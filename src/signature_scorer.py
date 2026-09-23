"""Exploratory, same-discovery-cohort weighted signature scoring.

Gene selection, weights, standardization and the Youden threshold use the discovery
cohort. Metrics are apparent training performance, not held-out clinical validation.
"""
import os
import sys
import json

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger


def calculate_composite_risk_score(
    expr_df: pd.DataFrame,
    labels: pd.Series,
    consensus_df: pd.DataFrame,
    top_n: int = 15,
    out_dir: str = None,
) -> dict:
    """Describe a signature on its discovery data; do not estimate clinical utility.

    ``expr_df`` is genes x samples and labels must explicitly identify every sample
    as Tumor or Normal. Use the producer's ``composite_score`` (preferred), or
    ``ensemble_score`` only if the composite column is absent, signed by ``log2FC``.
    Missing/invalid weights and directions are errors, never equal-weight defaults.

    No post-selection split or bootstrap clinical confidence interval is reported:
    selection and threshold fitting have already seen these samples. The saved
    weights, scaling parameters and provenance document this apparent evaluation,
    but do not constitute an independently validated model.
    """
    if not isinstance(top_n, (int, np.integer)) or isinstance(top_n, bool) or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")
    if expr_df.empty or expr_df.shape[1] < 2:
        raise ValueError("Signature scoring requires expression data for at least two samples.")
    if not expr_df.index.is_unique or not expr_df.columns.is_unique or not labels.index.is_unique:
        raise ValueError("Gene and sample identifiers must be unique.")
    aligned_labels = labels.reindex(expr_df.columns)
    if not aligned_labels.isin(["Tumor", "Normal"]).all():
        raise ValueError("Every expression sample needs an explicit Tumor or Normal label.")
    if aligned_labels.nunique() != 2:
        raise ValueError("Apparent diagnostic performance requires both Tumor and Normal samples.")
    if consensus_df.empty or not {"gene", "log2FC"}.issubset(consensus_df.columns):
        raise ValueError("Consensus ranking must contain genes and log2FC directions.")
    if consensus_df["gene"].isna().any() or consensus_df["gene"].duplicated().any():
        raise ValueError("Consensus gene identifiers must be nonmissing and unique.")

    weight_column = next((c for c in ("composite_score", "ensemble_score")
                          if c in consensus_df.columns), None)
    if weight_column is None:
        raise ValueError("Consensus ranking requires composite_score or ensemble_score weights.")
    selected = consensus_df.head(top_n)
    available = selected[selected["gene"].isin(expr_df.index)]
    available_genes = available["gene"].tolist()
    if not available_genes:
        raise ValueError("None of the consensus signature genes were found in the expression matrix.")

    strengths = pd.to_numeric(available[weight_column], errors="coerce").to_numpy(dtype=float)
    directions = pd.to_numeric(available["log2FC"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(strengths).all() or (strengths < 0).any():
        raise ValueError(f"{weight_column} weights must be finite and nonnegative.")
    if not np.isfinite(directions).all():
        raise ValueError("Signature log2FC directions must be finite.")
    weights = strengths * np.sign(directions)
    # Rescale first to avoid overflow even for unusually large finite scores.
    maximum = np.max(np.abs(weights))
    if maximum == 0:
        raise ValueError("Signature weights must have a nonzero signed total magnitude.")
    weights /= maximum
    weights /= np.sum(np.abs(weights))

    expr_sub = expr_df.loc[available_genes].apply(pd.to_numeric, errors="coerce")
    values = expr_sub.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Selected signature expression values must be finite numeric data.")
    means = values.mean(axis=1)
    stds = values.std(axis=1, ddof=1)
    # Constant genes contribute zero; no imputation from other samples/cohorts.
    scales = np.where(stds == 0, 1.0, stds)
    scores = weights @ ((values - means[:, None]) / scales[:, None])
    if not np.isfinite(scores).all():
        raise ValueError("Signature standardization produced nonfinite scores.")

    y_true = (aligned_labels == "Tumor").to_numpy(dtype=int)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = float(auc(fpr, tpr))
    # sklearn includes an infinite sentinel for 'predict none'; not a deployable cutoff.
    finite_indices = np.flatnonzero(np.isfinite(thresholds))
    best_idx = finite_indices[np.argmax((tpr - fpr)[finite_indices])]
    optimal_cutoff = float(thresholds[best_idx])
    y_pred = (scores >= optimal_cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    scores_df = pd.DataFrame({
        "sample_id": expr_df.columns,
        "actual_class": aligned_labels.to_numpy(),
        "predicted_class": np.where(y_pred == 1, "Tumor", "Normal"),
        # Preserve precision so the saved cutoff reproduces the saved classifications.
        "risk_score": scores,
        "classification_result": [
            ("True Positive" if actual else "False Positive") if pred
            else ("False Negative" if actual else "True Negative")
            for actual, pred in zip(y_true, y_pred)
        ],
        "evaluation_scope": "apparent_training",
    })
    perf_metrics = {
        "validation_status": "not_independently_validated",
        "evaluation_scope": "apparent_training",
        "evaluation_type": "apparent_training",
        "performance_label": "Apparent training performance (exploratory, same discovery cohort)",
        "clinical_use_supported": False,
        "confidence_intervals_status": "not_reported_selection_and_threshold_fit_on_evaluation_data",
        "provenance": {
            "selection_data": "same_discovery_cohort",
            "weight_source": weight_column,
            "direction_source": "log2FC",
            "standardization_fit_data": "same_discovery_cohort",
            "threshold_fit_data": "same_discovery_cohort",
            "evaluation_data": "same_discovery_cohort",
            "independent_validation_performed": False,
            "sample_ids": [str(s) for s in expr_df.columns],
        },
        "n_samples": len(y_true),
        "signature_size": len(available_genes),
        "signature_genes": available_genes,
        "missing_signature_genes": selected.loc[~selected["gene"].isin(expr_df.index), "gene"].tolist(),
        "signature_weights": dict(zip(available_genes, weights.tolist())),
        "standardization": {
            "method": "gene_z_score",
            "ddof": 1,
            "means": dict(zip(available_genes, means.tolist())),
            "scales": dict(zip(available_genes, scales.tolist())),
        },
        "optimal_decision_threshold": optimal_cutoff,
        "roc_auc": roc_auc,
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

    out_dir = out_dir if out_dir is not None else config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    scores_df.to_csv(os.path.join(out_dir, "patient_risk_scores.csv"), index=False)
    with open(os.path.join(out_dir, "diagnostic_performance.json"), "w", encoding="utf-8") as f:
        json.dump(perf_metrics, f, indent=2, allow_nan=False)
    logger.info("Apparent training signature performance (exploratory; not independently validated): "
                "%d genes, ROC-AUC = %.4f", len(available_genes), roc_auc)
    return {"scores_df": scores_df, "metrics": perf_metrics,
            "fpr": fpr.tolist(), "tpr": tpr.tolist()}
