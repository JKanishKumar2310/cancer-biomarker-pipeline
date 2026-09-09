"""
ML-Based Biomarker Ranking — Uses Random Forest to rank gene importance.

Trains Random Forest on the expression data, extracts feature importances, 
and cross-references with DE results to find consensus biomarkers.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger


def _prepare_data(
    expr_df: pd.DataFrame,
    labels: pd.Series,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare X (samples × genes), y (encoded labels), gene_names."""
    X = expr_df.T.values  # samples × genes
    le = LabelEncoder()
    y = le.fit_transform(labels.values)  # 0=Normal, 1=Tumor
    gene_names = expr_df.index.tolist()
    logger.info(f"ML input: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y, gene_names


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    gene_names: list[str],
) -> pd.DataFrame:
    """Train Random Forest and return normalized feature importances."""
    logger.info("Training Random Forest...")

    params = config.ML_MODELS["RandomForest"]
    rf = RandomForestClassifier(
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        **params,
    )

    # Cross-validated accuracy
    cv = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED)
    scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    logger.info(f"  RF CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # Fit on all data for importances
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=gene_names, name="RF_importance")
    
    # Normalize
    rf_norm = (imp - imp.min()) / (imp.max() - imp.min() + 1e-10)
    
    ranking = pd.DataFrame({
        "gene": imp.index,
        "RF_importance": imp.values,
        "ensemble_score": rf_norm.values, # using same column name for compat
    }, index=imp.index)

    ranking = ranking.sort_values("ensemble_score", ascending=False)
    ranking["ml_rank"] = range(1, len(ranking) + 1)
    return ranking


def find_consensus_biomarkers(
    ml_ranking: pd.DataFrame,
    de_results: pd.DataFrame,
    top_n: int = None,
) -> pd.DataFrame:
    """
    Find consensus biomarkers — genes that rank highly in BOTH
    statistical DE analysis AND ML feature importance.
    """
    if top_n is None:
        top_n = config.TOP_ML_GENES

    logger.info("Finding consensus biomarkers (DE + ML)...")

    top_ml = set(ml_ranking.head(top_n).index)
    sig_de = de_results[de_results["regulation"] != "Not Significant"]
    top_de = set(sig_de.index)

    consensus_genes = top_ml & top_de
    logger.info(f"  Top {top_n} ML genes: {len(top_ml)}")
    logger.info(f"  Significant DE genes: {len(top_de)}")
    logger.info(f"  Consensus biomarkers: {len(consensus_genes)}")

    if len(consensus_genes) == 0:
        logger.warning("No strict consensus found, relaxing criteria...")
        trending = de_results[de_results["pvalue"] < 0.1]
        consensus_genes = top_ml & set(trending.index)
        logger.info(f"  Relaxed consensus: {len(consensus_genes)}")

    consensus_list = []
    for gene in consensus_genes:
        row = {
            "gene": gene,
            "log2FC": de_results.loc[gene, "log2FC"] if gene in de_results.index else np.nan,
            "adj_pvalue": de_results.loc[gene, "adj_pvalue"] if gene in de_results.index else np.nan,
            "regulation": de_results.loc[gene, "regulation"] if gene in de_results.index else "Unknown",
            "ensemble_score": ml_ranking.loc[gene, "ensemble_score"] if gene in ml_ranking.index else 0,
            "ml_rank": ml_ranking.loc[gene, "ml_rank"] if gene in ml_ranking.index else np.nan,
        }
        consensus_list.append(row)

    consensus_df = pd.DataFrame(consensus_list)
    if len(consensus_df) > 0:
        consensus_df = consensus_df.sort_values("ensemble_score", ascending=False)
        consensus_df.index = consensus_df["gene"]

    return consensus_df


def run_ml_biomarker_ranking(
    expr_df: pd.DataFrame,
    labels: pd.Series,
    de_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the ML biomarker ranking pipeline."""
    logger.info("=" * 60)
    logger.info("ML-BASED BIOMARKER RANKING")
    logger.info("=" * 60)

    X, y, gene_names = _prepare_data(expr_df, labels)
    ml_ranking = train_random_forest(X, y, gene_names)
    consensus = find_consensus_biomarkers(ml_ranking, de_results)

    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    ml_ranking.head(100).to_csv(os.path.join(config.RESULTS_DIR, "ml_ranking_top100.csv"), index=False)
    consensus.to_csv(os.path.join(config.RESULTS_DIR, "consensus_biomarkers.csv"), index=False)
    logger.info("Saved top 100 ML-ranked genes and consensus biomarkers")

    logger.info("ML biomarker ranking complete ✓")
    return ml_ranking, consensus
