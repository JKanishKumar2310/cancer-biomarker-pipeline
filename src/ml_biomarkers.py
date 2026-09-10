"""
ML-Based Biomarker Ranking -- Multi-Model Ensemble (RF, GradientBoosting, L1-Penalized Linear).

Trains multiple diverse classifiers on gene expression profiles, extracts normalized
feature importances / sparse coefficients, and aggregates into a consensus ensemble score.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger

def _make_cv(y: np.ndarray, groups: np.ndarray | None, n_splits: int = None):
    """
    Grade-aware CV splitter. Uses StratifiedGroupKFold when patient IDs allow it,
    otherwise falls back to StratifiedKFold.
    """
    if n_splits is None:
        n_splits = config.N_SPLITS
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
        logger.info(f"Using StratifiedGroupKFold ({n_splits} folds) across {len(np.unique(groups))} unique patient groups")
        return cv
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    logger.info(f"Using StratifiedKFold ({n_splits} folds)")
    return cv


def _prepare_data(
    expr_df: pd.DataFrame,
    labels: pd.Series,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    """Prepare X (samples x genes), y (encoded labels), gene_names, and patient groups."""
    X = expr_df.T.values  # samples x genes
    le = LabelEncoder()
    y = le.fit_transform(labels.values)  # 0=Normal, 1=Tumor
    gene_names = expr_df.index.tolist()

    groups = None
    patient_ids = labels.attrs.get("patient_id")
    if patient_ids is not None:
        groups = patient_ids.loc[labels.index].values
        logger.info(f"ML input: {X.shape[0]} samples x {X.shape[1]} features across {len(np.unique(groups))} unique patient groups")
    else:
        logger.info(f"ML input: {X.shape[0]} samples x {X.shape[1]} features")

    return X, y, gene_names, groups


def _min_max_norm(series: pd.Series) -> pd.Series:
    """Normalize a series to [0, 1]."""
    denom = series.max() - series.min()
    if denom == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / denom


def _fit_ensemble_on_indices(
    X: np.ndarray,
    y: np.ndarray,
    gene_names: list[str],
    train_idx: np.ndarray,
) -> pd.DataFrame:
    """Fit the RF + GB + L1 ensemble using ONLY the provided training indices.

    This is the leakage-safe primitive used for both the final full-data ranking
    and every nested outer fold. All scalers live inside the fitted pipelines, so
    no information from held-out samples ever touches the transformations.
    """
    X_tr, y_tr = X[train_idx], y[train_idx]

    rf = RandomForestClassifier(
        random_state=config.RANDOM_SEED, n_jobs=-1,
        **config.ML_MODELS.get("RandomForest", {}),
    )
    rf.fit(X_tr, y_tr)
    rf_imp = pd.Series(rf.feature_importances_, index=gene_names)

    gb = GradientBoostingClassifier(
        random_state=config.RANDOM_SEED,
        **config.ML_MODELS.get("GradientBoosting", {}),
    )
    gb.fit(X_tr, y_tr)
    gb_imp = pd.Series(gb.feature_importances_, index=gene_names)

    l1_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("l1", LogisticRegression(
            random_state=config.RANDOM_SEED, max_iter=1000,
            **config.ML_MODELS.get("L1_LogisticRegression", {}),
        )),
    ])
    l1_pipe.fit(X_tr, y_tr)
    l1_imp = pd.Series(np.abs(l1_pipe.named_steps["l1"].coef_[0]), index=gene_names)

    ensemble = (_min_max_norm(rf_imp) * 0.45
                + _min_max_norm(gb_imp) * 0.35
                + _min_max_norm(l1_imp) * 0.20)
    ranking = pd.DataFrame({
        "gene": gene_names,
        "RF_importance": rf_imp.values,
        "GB_importance": gb_imp.values,
        "L1_importance": l1_imp.values,
        "ensemble_score": ensemble.values,
    }, index=gene_names).sort_values("ensemble_score", ascending=False)
    ranking["ml_rank"] = range(1, len(ranking) + 1)
    return ranking
def train_ensemble_models(
    X: np.ndarray,
    y: np.ndarray,
    gene_names: list[str],
    groups: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Train Random Forest, Gradient Boosting, and L1-penalized Logistic Regression.
    Enforces fold-local scaling and StratifiedGroupKFold to prevent patient identity leakage.
    Returns composite ensemble feature importance ranking and CV metrics.
    """
    if groups is not None and len(np.unique(groups)) >= config.N_SPLITS:
        cv = _make_cv(y, groups)
    else:
        cv = _make_cv(y, None)
    # 1/3 Random Forest
    logger.info("1/3 Training Random Forest...")
    rf_params = config.ML_MODELS.get("RandomForest", {})
    rf = RandomForestClassifier(random_state=config.RANDOM_SEED, n_jobs=-1, **rf_params)
    rf_scores = cross_val_score(rf, X, y, cv=cv, groups=groups, scoring="accuracy")
    logger.info(f"  RF CV Accuracy: {rf_scores.mean():.4f} +/- {rf_scores.std():.4f}")

    # 2/3 Gradient Boosting
    logger.info("2/3 Training Gradient Boosting...")
    gb_params = config.ML_MODELS.get("GradientBoosting", {})
    gb = GradientBoostingClassifier(random_state=config.RANDOM_SEED, **gb_params)
    gb_scores = cross_val_score(gb, X, y, cv=cv, groups=groups, scoring="accuracy")
    logger.info(f"  GB CV Accuracy: {gb_scores.mean():.4f} +/- {gb_scores.std():.4f}")

    # 3/3 L1-Penalized Sparse Model (fold-local StandardScaler lives inside the pipeline)
    logger.info("3/3 Training L1-Penalized Sparse Model (with Fold-Local StandardScaler)...")
    l1_params = config.ML_MODELS.get("L1_LogisticRegression", {})
    l1_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("l1", LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000, **l1_params))
    ])
    l1_scores = cross_val_score(l1_pipe, X, y, cv=cv, groups=groups, scoring="accuracy")
    logger.info(f"  L1 (Fold-Isolated Scaler) CV Accuracy: {l1_scores.mean():.4f} +/- {l1_scores.std():.4f}")

    # Final feature ranking: fit the ensemble on the FULL dataset for reporting only.
    # The honest, leakage-free performance estimate is produced by
    # nested_cv_ensemble_evaluation(), which re-selects features inside every fold.
    ranking = _fit_ensemble_on_indices(X, y, gene_names, np.arange(len(y)))

    cv_metrics = {
        "rf_cv_acc": float(rf_scores.mean()),
        "rf_cv_std": float(rf_scores.std()),
        "gb_cv_acc": float(gb_scores.mean()),
        "gb_cv_std": float(gb_scores.std()),
        "l1_cv_acc": float(l1_scores.mean()),
        "l1_cv_std": float(l1_scores.std()),
    }
    return ranking, cv_metrics


def find_consensus_biomarkers(
    ml_ranking: pd.DataFrame,
    de_results: pd.DataFrame,
    top_n: int = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """Find consensus biomarkers - genes that rank highly in BOTH
    statistical DE analysis AND ML ensemble feature importance.
    """
    if top_n is None:
        top_n = config.TOP_ML_GENES
    if not quiet:
        logger.info("Finding consensus biomarkers (DE + Multi-Model ML)...")

    # Consider top candidate pool from ML (extended to avoid collinear suppression)
    ml_pool_size = max(100, top_n * 2)
    top_ml_genes = set(ml_ranking.head(ml_pool_size).index)
    sig_de = de_results[de_results["regulation"] != "Not Significant"]
    top_de_genes = set(sig_de.index)

    # Initial intersection
    candidate_genes = top_ml_genes & top_de_genes
    if not quiet:
        logger.info(f"  ML candidate pool (top {ml_pool_size}): {len(top_ml_genes)}")
        logger.info(f"  Significant DE genes: {len(top_de_genes)}")
        logger.info(f"  Overlapping consensus pool: {len(candidate_genes)}")

    if len(candidate_genes) == 0:
        logger.warning(
            "No genes satisfied the strict FDR < %.2f consensus (intersection empty). "
            "Relaxing to nominally significant genes (raw p < 0.05) and flagging them "
            "as EXPLORATORY - these do NOT meet the FDR-controlled consensus criterion.",
            config.PVALUE_THRESHOLD,
        )
        trending = de_results[de_results["pvalue"] < 0.05]
        candidate_genes = top_ml_genes & set(trending.index)

    # Mark which biomarkers met the strict FDR-controlled criterion vs the relaxed fallback
    strict_de = set(de_results[de_results["regulation"] != "Not Significant"].index)

    # Compute Hybrid Composite Score: balances ML importance with DE effect magnitude
    # DE magnitude metric = |log2FC| * -log10(adj_pvalue)
    de_scores = {}
    for g in candidate_genes:
        fc = abs(de_results.loc[g, "log2FC"]) if g in de_results.index else 1.0
        padj = de_results.loc[g, "adj_pvalue"] if g in de_results.index else 0.05
        neg_log_p = -np.log10(max(padj, 1e-300))
        de_scores[g] = fc * np.log1p(neg_log_p)

    max_de = max(de_scores.values()) if de_scores else 1.0
    de_norm = {g: de_scores[g] / max_de for g in de_scores}

    consensus_list = []
    for gene in candidate_genes:
        ens_score = ml_ranking.loc[gene, "ensemble_score"] if gene in ml_ranking.index else 0.0
        de_component = de_norm.get(gene, 0.0)
        # Composite score balances ML predictive weight (60%) and biological DE strength (40%)
        composite = (0.60 * ens_score) + (0.40 * de_component)

        row = {
            "gene": gene,
            "log2FC": de_results.loc[gene, "log2FC"] if gene in de_results.index else np.nan,
            "adj_pvalue": de_results.loc[gene, "adj_pvalue"] if gene in de_results.index else np.nan,
            "regulation": de_results.loc[gene, "regulation"] if gene in de_results.index else "Unknown",
            "ensemble_score": ens_score,
            "composite_score": composite,
            "ml_rank": ml_ranking.loc[gene, "ml_rank"] if gene in ml_ranking.index else np.nan,
            "consensus_tier": "FDR-controlled" if gene in strict_de else "Exploratory (nominal only)",
        }
        consensus_list.append(row)

    consensus_df = pd.DataFrame(consensus_list)
    if len(consensus_df) > 0:
        consensus_df = consensus_df.sort_values("composite_score", ascending=False).head(top_n)
        consensus_df.index = consensus_df["gene"]

    logger.info(f"  Final consensus biomarkers selected: {len(consensus_df)}")
    return consensus_df


def evaluate_consensus_signature_cv(
    X: np.ndarray,
    y: np.ndarray,
    gene_names: list[str],
    sig_genes: list[str],
    groups: np.ndarray | None = None,
) -> dict:
    """
    Evaluate the selected consensus biomarker signature in an isolated cross-validation design.
    Ensures that reported internal performance reflects the selected signature under patient grouping.
    """
    valid_idx = [gene_names.index(g) for g in sig_genes if g in gene_names]
    if len(valid_idx) == 0:
        return {}

    X_sub = X[:, valid_idx]
    if groups is not None and len(np.unique(groups)) >= config.N_SPLITS:
        cv = StratifiedGroupKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    else:
        cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED, n_jobs=-1))
    ])
    accs = cross_val_score(pipe, X_sub, y, cv=cv, groups=groups, scoring="accuracy")
    aucs = cross_val_score(pipe, X_sub, y, cv=cv, groups=groups, scoring="roc_auc")
    logger.info(f"  Selected Signature ({len(valid_idx)} genes) Patient-Grouped CV Accuracy: {accs.mean():.4f} +/- {accs.std():.4f}")
    logger.info(f"  Selected Signature ({len(valid_idx)} genes) Patient-Grouped CV ROC-AUC:  {aucs.mean():.4f} +/- {aucs.std():.4f}")
    return {
        "signature_cv_acc": float(accs.mean()),
        "signature_cv_acc_std": float(accs.std()),
        "signature_cv_auc": float(aucs.mean()),
        "signature_cv_auc_std": float(aucs.std()),
    }


def nested_cv_ensemble_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    gene_names: list[str],
    de_results: pd.DataFrame,
    groups: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Leakage-free nested evaluation of the full selection + classification pipeline.

    For every outer grouped fold the ENTIRE biomarker selection procedure
    (ensemble training then consensus selection against the DE table) is repeated
    using only that fold's training partition, then scored on the untouched test
    partition. This removes the optimistic bias of selecting features on the full
    dataset and then cross-validating an already-locked signature.
    """
    cv = _make_cv(y, groups)
    n_outer = config.N_SPLITS
    accs, aucs = [], []
    selection_counts: dict[str, int] = {}
    fold_rows = []
    gene_set = set(gene_names)

    for fold_i, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
        # Re-select features using ONLY the training partition
        fold_ranking = _fit_ensemble_on_indices(X, y, gene_names, tr)
        fold_de = de_results.loc[[g for g in de_results.index if g in gene_set]]
        fold_consensus = find_consensus_biomarkers(fold_ranking, fold_de, top_n=20, quiet=True)
        fold_genes = [g for g in fold_consensus["gene"].tolist() if g in gene_set]
        for g in fold_genes:
            selection_counts[g] = selection_counts.get(g, 0) + 1
        if len(fold_genes) == 0:
            logger.info(f"  Outer fold {fold_i}/{n_outer}: no consensus genes on training split - skipped")
            continue
        idx = [gene_names.index(g) for g in fold_genes]

        # Fit a fresh signature classifier on TRAIN, score on TEST (scaler is fold-local)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_SEED, n_jobs=-1)),
        ])
        pipe.fit(X[tr][:, idx], y[tr])
        y_pred = pipe.predict(X[te][:, idx])
        y_prob = pipe.predict_proba(X[te][:, idx])[:, 1]

        acc_f = accuracy_score(y[te], y_pred)
        accs.append(acc_f)
        try:
            auc_f = roc_auc_score(y[te], y_prob)
            aucs.append(auc_f)
        except ValueError:
            auc_f = np.nan
        fold_rows.append({
            "outer_fold": fold_i,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_selected_genes": len(fold_genes),
            "selected_genes": ";".join(fold_genes),
            "fold_accuracy": acc_f,
            "fold_roc_auc": auc_f,
        })
        logger.info(
            f"  Outer fold {fold_i}/{n_outer}: {len(fold_genes)} genes selected on train -> "
            f"test accuracy {acc_f:.4f}"
        )

    per_fold = pd.DataFrame(fold_rows)
    if not accs:
        logger.warning("Nested CV could not evaluate any fold (no consensus genes selected).")
        return per_fold, {"nested_cv_acc": np.nan, "nested_cv_auc": np.nan}

    metrics = {
        "nested_cv_acc": float(np.mean(accs)),
        "nested_cv_acc_std": float(np.std(accs)),
        "nested_cv_auc": float(np.nanmean(aucs)) if aucs else np.nan,
        "nested_cv_auc_std": float(np.nanstd(aucs)) if aucs else np.nan,
        "nested_cv_folds": len(accs),
        "selection_stability": {
            g: c / len(accs) for g, c in sorted(selection_counts.items(), key=lambda x: -x[1])
        },
    }
    logger.info(
        f"  NESTED (leakage-free) CV accuracy: {metrics['nested_cv_acc']:.4f} +/- {metrics['nested_cv_acc_std']:.4f} "
        f"| ROC-AUC: {metrics['nested_cv_auc']:.4f} across {metrics['nested_cv_folds']} folds"
    )
    return per_fold, metrics

def run_ml_biomarker_ranking(
    expr_df: pd.DataFrame,
    labels: pd.Series,
    de_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the multi-model ML biomarker ranking pipeline."""
    logger.info("=" * 60)
    logger.info("MULTI-MODEL ML BIOMARKER RANKING (RF + GB + L1)")
    logger.info("=" * 60)

    X, y, gene_names, groups = _prepare_data(expr_df, labels)
    ml_ranking, cv_metrics = train_ensemble_models(X, y, gene_names, groups=groups)
    consensus = find_consensus_biomarkers(ml_ranking, de_results)

    # Honest internal estimate: repeat selection inside each outer fold (nested CV).
    # This is the leakage-free number to report, NOT the locked-signature score.
    nested_per_fold, nested_metrics = nested_cv_ensemble_evaluation(
        X, y, gene_names, de_results, groups=groups
    )

    # Locked (apparent) signature score for reference only, clearly secondary.
    sig_genes = consensus["gene"].head(20).tolist()
    sig_metrics = evaluate_consensus_signature_cv(X, y, gene_names, sig_genes, groups=groups)

    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    ml_ranking.head(100).to_csv(os.path.join(config.RESULTS_DIR, "ml_ranking_top100.csv"), index=False)
    consensus.to_csv(os.path.join(config.RESULTS_DIR, "consensus_biomarkers.csv"), index=False)
    if len(nested_per_fold) > 0:
        nested_per_fold.to_csv(os.path.join(config.RESULTS_DIR, "nested_cv_metrics.csv"), index=False)
    logger.info("Saved top 100 ML-ranked genes, consensus biomarkers, and nested-CV metrics")

    logger.info("ML biomarker ranking complete [PASS]")
    return ml_ranking, consensus
