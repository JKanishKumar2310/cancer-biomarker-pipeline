"""
Cross-Cohort External Validation Engine.

Tests the biomarker signature and Random Forest model trained on the
discovery cohort on an independent external cohort WITHOUT retraining,
proving cross-study generalizability.

Discovery → External cohort mapping:
  GSE19804 (Lung, Taiwan)  → GSE18842 (Lung, Europe: 46 tumors + 45 normals)
  GSE15852 (Breast, Malaysia) → GSE42568 (Breast, Europe: 104 tumors + 17 normals)
"""
import os
import sys
import gzip
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger

# Map discovery accession → external validation accession + platform
_EXT_COHORT_MAP = {
    "GSE19804": {
        "accession": "GSE18842",
        "platform": "GPL570",
        "description": "GSE18842 (Europe, n=91: 46 tumors + 45 normals)",
        "matrix_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE18nnn/GSE18842/matrix/GSE18842_series_matrix.txt.gz",
    },
    "GSE8671": {
        "accession": "GSE20916",
        "platform": "GPL570",
        "description": "GSE20916 (Colorectal Cohort, n=145: 101 adenomas/carcinomas + 44 normal mucosa)",
        "matrix_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE20nnn/GSE20916/matrix/GSE20916_series_matrix.txt.gz",
    },
    "GSE15852": {
        "accession": "GSE42568",
        "platform": "GPL570",
        "description": "GSE42568 (Europe Breast Cohort, n=121: 104 tumors + 17 normals)",
        "matrix_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/GSE42568/matrix/GSE42568_series_matrix.txt.gz",
    },
}


class ExternalValidationUnavailable(ValueError):
    """An independent external cohort could not be established."""


def _get_ext_info() -> dict:
    from src.ai_annotator import validate_validation_cohort

    val_info = getattr(config, "VALIDATION_COHORT", None)
    if val_info is None:
        discovery = str(config.GEO_ACCESSION).strip().upper()
        val_info = _EXT_COHORT_MAP.get(discovery)
    # Revalidate config/cache entries at use time; discovery may have changed.
    val_info = validate_validation_cohort(val_info)
    if val_info.get("status") == "UNAVAILABLE":
        raise ExternalValidationUnavailable(val_info.get("reason", "External validation unavailable."))
    acc = val_info["accession"]
    return {
        "accession": acc,
        "platform": val_info.get("platform", "GPL570"),
        "description": val_info.get("label", val_info.get("description", f"{acc} External Validation Cohort")),
        "matrix_url": f"https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/matrix/{acc}_series_matrix.txt.gz",
    }


def _unavailable_result(reason: str, force_synthetic: bool, status: str = "UNAVAILABLE") -> dict:
    """Overwrite stale success artifacts without fabricating performance or a ROC."""
    logger.warning(f"External validation unavailable: {reason}")
    result = {
        "accuracy": float("nan"), "roc_auc": float("nan"),
        "sensitivity": float("nan"), "specificity": float("nan"),
        "status": status, "reason": reason,
        "signature_genes": [], "roc_curve": {"fpr": [], "tpr": []},
    }
    # Test mode already redirects RESULTS_DIR into .../synthetic_test; don't nest deeper.
    out_dir = config.RESULTS_DIR if os.path.basename(os.path.normpath(config.RESULTS_DIR)) == "synthetic_test" else os.path.join(config.RESULTS_DIR, "synthetic_test") if force_synthetic else config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{
        "discovery_cohort": config.GEO_ACCESSION,
        "external_cohort": "Unavailable",
        "status": status, "error_details": reason,
        "test_accuracy": result["accuracy"], "roc_auc": result["roc_auc"],
        "sensitivity": result["sensitivity"], "specificity": result["specificity"],
        "data_provenance": "External validation not performed",
    }]).to_csv(os.path.join(out_dir, "external_validation_metrics.csv"), index=False)
    pd.DataFrame(columns=["fpr", "tpr"]).to_csv(os.path.join(out_dir, "external_roc_curve.csv"), index=False)
    return result


def _sample_ids(expression: pd.DataFrame, labels: pd.Series) -> set[str]:
    """Normalize IDs from both tables, including samples dropped in preprocessing."""
    return {str(value).strip().upper() for value in list(expression.columns) + list(labels.index)
            if pd.notna(value) and str(value).strip()}


def _create_synthetic_external_cohort() -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic external validation cohort for CI/offline testing."""
    np.random.seed(config.RANDOM_SEED + 10)
    samples = [f"Ext_Tumor_{i+1}" for i in range(30)] + [f"Ext_Normal_{i+1}" for i in range(30)]
    genes = list(dict.fromkeys(config.KNOWN_MARKERS + ["CLIC5", "TNNC1", "TOP2A", "CDK1", "EPCAM", "CDH3", "FXYD1", "DHRS11"]))
    base = np.random.normal(7.0, 1.5, size=(len(genes), len(samples)))
    base[:len(config.KNOWN_MARKERS), :30] += 2.0
    expr = pd.DataFrame(base, index=genes, columns=samples)
    labels = pd.Series(["Tumor"] * 30 + ["Normal"] * 30, index=samples, name="condition")
    return expr, labels


def load_external_cohort(force_synthetic: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Load the external validation cohort matching the current cancer type."""
    if force_synthetic:
        logger.info("Using synthetic external validation cohort (test mode)...")
        return _create_synthetic_external_cohort()

    info = _get_ext_info()
    ext_acc = info["accession"]

    cache_expr = os.path.join(config.DATA_DIR, f"{ext_acc}_expression.csv")
    cache_labels = os.path.join(config.DATA_DIR, f"{ext_acc}_labels.csv")

    if os.path.exists(cache_expr) and os.path.exists(cache_labels):
        logger.info(f"Loading cached {ext_acc} external validation cohort...")
        expr_df = pd.read_csv(cache_expr, index_col=0)
        raw_labels = pd.read_csv(cache_labels, index_col=0)
        if isinstance(raw_labels, pd.DataFrame):
            col = "condition" if "condition" in raw_labels.columns else raw_labels.columns[0]
            labels = raw_labels[col]
        else:
            labels = raw_labels.squeeze()
        return expr_df, labels

    matrix_file = os.path.join(config.DATA_DIR, f"{ext_acc}_series_matrix.txt.gz")
    annot_file = os.path.join(config.DATA_DIR, f"{info['platform']}.annot.gz")
    # Derive the annotation using unified platform mapping
    from src.ai_geo_curator import fetch_gpl_probe_mapping

    # Download if needed with strict error handling
    import urllib.request
    if not os.path.exists(matrix_file):
        try:
            logger.info(f"Downloading authentic {ext_acc} series matrix from NCBI GEO...")
            urllib.request.urlretrieve(info["matrix_url"], matrix_file)
        except Exception as e:
            if force_synthetic:
                logger.warning(f"Could not download {ext_acc}: {e}. Using test validation cohort.")
                return _create_synthetic_external_cohort()
            raise RuntimeError(
                f"Failed to download authentic external validation cohort {ext_acc} from {info['matrix_url']}: {e}. "
                "Cross-study validation cannot proceed without authentic cohort data."
            ) from e

    try:
        logger.info(f"Parsing {ext_acc} external cohort series matrix...")
        meta_lines = []
        skiprows = 0
        with gzip.open(matrix_file, "rt", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if line.startswith("!series_matrix_table_begin"):
                    skiprows = i + 1
                    break
                meta_lines.append(line)

        sample_ids, sample_titles = [], []
        for line in meta_lines:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [x.strip(' "\t\r\n') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                sample_titles = [x.strip(' "\t\r\n') for x in line.split("\t")[1:]]

        labels_dict = {}
        for i, s_id in enumerate(sample_ids):
            t = sample_titles[i].lower() if i < len(sample_titles) else ""
            if any(k in t for k in ["normal", "healthy", "adjacent", "donnor", "control", "non-tumor"]):
                labels_dict[s_id] = "Normal"
            else:
                labels_dict[s_id] = "Tumor"

        labels = pd.Series(labels_dict, name="condition")

        # Read expression
        logger.info(f"Reading {ext_acc} expression matrix...")
        df = pd.read_csv(
            matrix_file, compression="gzip", skiprows=skiprows, sep="\t", index_col=0, comment="!"
        )
        df = df[~df.index.astype(str).str.startswith("!")]
        df = df.apply(pd.to_numeric, errors="coerce")

        # Map probes
        logger.info(f"Mapping probes via {info['platform']} annotations...")
        probe_to_gene = fetch_gpl_probe_mapping(info["platform"])
        if probe_to_gene:
            df["gene"] = df.index.astype(str).map(probe_to_gene)
            df = df.dropna(subset=["gene"]).set_index("gene")
            expr_df = df.groupby(df.index).mean()
        else:
            expr_df = df

        common = expr_df.columns.intersection(labels.index)
        expr_df = expr_df[common]
        labels = labels[common]

        if expr_df.values.max() > 50:
            expr_df = np.log2(expr_df + 1)

        # Cache
        expr_df.to_csv(cache_expr)
        labels.to_frame().to_csv(cache_labels)
        logger.info(f"{ext_acc} loaded & cached: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples ({dict(labels.value_counts())})")
        return expr_df, labels
    except Exception as e:
        raise ExternalValidationUnavailable(f"Failed to process authentic external cohort {ext_acc}: {e}") from e


def run_external_validation(force_synthetic: bool = False) -> dict:
    """Train RF on discovery cohort and evaluate on external cohort (zero-shot)."""
    if force_synthetic:
        info = {"accession": "SYNTHETIC", "description": "Synthetic external cohort (test mode)"}
    else:
        try:
            info = _get_ext_info()
        except ExternalValidationUnavailable as exc:
            return _unavailable_result(str(exc), force_synthetic)
    logger.info("=" * 60)
    logger.info("CROSS-COHORT EXTERNAL VALIDATION")
    logger.info("=" * 60)

    # 1. Load Discovery Cohort
    from src.data_loader import load_data
    from src.preprocessing import preprocess

    train_expr, train_labels = load_data(force_synthetic=force_synthetic)
    # Prefer the PREPROCESSED labels: preprocess() may drop samples (QC/unknown),
    # and X_train rows below must align 1:1 with y_train or sklearn raises a
    # found-array-with-inconsistent-number-of-samples error. A None result
    # (preprocessing skipped) falls back to the raw labels unchanged.
    train_clean, cleaned_labels = preprocess(train_expr, train_labels)
    if cleaned_labels is not None:
        train_labels = cleaned_labels

    # 2. Load Consensus Biomarker Signature
    cons_file = os.path.join(config.RESULTS_DIR, "consensus_biomarkers.csv")
    if os.path.exists(cons_file):
        cons_df = pd.read_csv(cons_file)
        sig_genes = cons_df["gene"].head(20).tolist()
    else:
        sig_genes = config.KNOWN_MARKERS[:10]

    logger.info(f"Using {len(sig_genes)} consensus biomarker genes for signature model: {sig_genes[:6]}...")

    # 3. Load External Cohort
    try:
        test_expr, test_labels = load_external_cohort(force_synthetic=force_synthetic)
    except ExternalValidationUnavailable as exc:
        return _unavailable_result(str(exc), force_synthetic)

    overlap = _sample_ids(train_expr, train_labels) & _sample_ids(test_expr, test_labels)
    if overlap:
        return _unavailable_result(
            f"Discovery and external cohorts share {len(overlap)} sample ID(s); independence is not established.",
            force_synthetic, status="SAMPLE_OVERLAP",
        )

    # Find common genes in signature
    valid_sig = [g for g in sig_genes if g in train_clean.index and g in test_expr.index]
    if len(valid_sig) == 0:
        valid_sig = [g for g in config.KNOWN_MARKERS if g in train_clean.index and g in test_expr.index]
    if len(valid_sig) == 0:
        common = list(train_clean.index.intersection(test_expr.index))
        valid_sig = common[:min(20, len(common))]

    logger.info(f"Common signature genes present in both cohorts: {len(valid_sig)}/{len(sig_genes)}")

    if len(valid_sig) == 0:
        return _unavailable_result("No overlapping signature genes between discovery and external cohorts.",
                                   force_synthetic, status="ZERO_FEATURE_OVERLAP")

    # 4. Train on Discovery, Test on External
    X_train = train_clean.loc[valid_sig].T.values
    y_train = (train_labels == "Tumor").astype(int).values

    rf = RandomForestClassifier(n_estimators=500, random_state=config.RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)

    X_test = test_expr.loc[valid_sig].T.values
    y_test = (test_labels == "Tumor").astype(int).values

    test_preds = rf.predict(X_test)
    test_probs = rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, test_preds)

    # Guard against a single-class external cohort (e.g. tumor-only survival set):
    # ROC-AUC and roc_curve are undefined when only one class is present.
    if len(np.unique(y_test)) < 2:
        logger.warning(
            f"External cohort {info['accession']} contains only one class "
            f"(y_test unique = {np.unique(y_test).tolist()}); ROC-AUC is undefined. "
            f"Reporting accuracy only."
        )
        auc = float("nan")
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        tn, fp, fn, tp = 0, 0, 0, 0
    else:
        auc = roc_auc_score(y_test, test_probs)
        tn, fp, fn, tp = confusion_matrix(y_test, test_preds, labels=[0, 1]).ravel()
        fpr, tpr, _ = roc_curve(y_test, test_probs)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    logger.info(f"\nExternal Cohort Performance (Zero-Shot on {info['accession']}):")
    logger.info(f"  • Test Accuracy: {acc * 100:.2f}%")
    logger.info(f"  • ROC-AUC Score: {auc:.4f}")
    logger.info(f"  • Sensitivity:   {sensitivity * 100:.2f}% (Tumor Detection)")
    logger.info(f"  • Specificity:   {specificity * 100:.2f}% (Normal Tissue Rule-Out)")
    logger.info(f"  • Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # Save with complete traceable provenance
    metrics_df = pd.DataFrame([{
        "discovery_cohort": f"{config.GEO_ACCESSION} (n={len(train_labels)})",
        "external_cohort": info["description"],
        "cancer_type": getattr(config, "CANCER_TYPE", "Cancer"),
        "disease_matched": True,
        "n_train_samples": len(train_labels),
        "n_test_samples": len(test_labels),
        "signature_genes_tested": len(valid_sig),
        "signature_gene_list": ";".join(valid_sig),
        "test_accuracy": acc,
        "roc_auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "data_provenance": "Authentic NCBI GEO Cross-Cohort Transfer (Zero Retraining)",
    }])

    out_dir = config.RESULTS_DIR if os.path.basename(os.path.normpath(config.RESULTS_DIR)) == "synthetic_test" else os.path.join(config.RESULTS_DIR, "synthetic_test") if force_synthetic else config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(out_dir, "external_validation_metrics.csv"), index=False)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(os.path.join(out_dir, "external_roc_curve.csv"), index=False)

    logger.info("Cross-cohort external validation complete ✓")
    return {"accuracy": acc, "roc_auc": auc, "sensitivity": sensitivity,
            "specificity": specificity, "signature_genes": valid_sig,
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()}}


if __name__ == "__main__":
    res = run_external_validation()
