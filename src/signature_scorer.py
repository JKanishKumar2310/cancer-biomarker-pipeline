"""
Composite Biomarker Signature Risk Score & Diagnostic Index Calculator.

Computes a patient-level multi-gene diagnostic risk score using consensus feature weights.
Determines optimal clinical decision threshold via Youden's J-statistic
and reports clinical utility metrics (Sensitivity, Specificity, PPV, NPV) with 95% Confidence Intervals.
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
    out_dir: str = None
) -> dict:
    """
    Calculate patient-level diagnostic risk scores and diagnostic metrics.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Genes (rows) x Samples (columns).
    labels : pd.Series
        Sample labels ('Tumor' vs 'Normal').
    consensus_df : pd.DataFrame
        Consensus biomarker ranking with 'gene' and 'consensus_score'.
    top_n : int
        Number of top genes to include in the diagnostic signature.
    out_dir : str, optional
        Results output directory.

    Returns
    -------
    dict with patient scores DataFrame, optimal cutoff, and performance metrics.
    """
    if out_dir is None:
        out_dir = config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    sig_genes = consensus_df.head(top_n)["gene"].tolist()
    available_genes = [g for g in sig_genes if g in expr_df.index]

    if not available_genes:
        raise ValueError("None of the consensus signature genes were found in the expression matrix.")

    # Extract weights from consensus score or log2FC direction
    weights = []
    for g in available_genes:
        sub = consensus_df[consensus_df["gene"] == g]
        score = float(sub["consensus_score"].values[0]) if "consensus_score" in sub.columns else 1.0
        # Determine direction: positive if upregulated, negative if downregulated
        fc = float(sub["log2FC"].values[0]) if "log2FC" in sub.columns else 1.0
        w = score * np.sign(fc)
        weights.append(w)

    weights = np.array(weights)
    if np.sum(np.abs(weights)) > 0:
        weights = weights / np.sum(np.abs(weights))

    # Standardize gene expression across samples (Z-scores)
    expr_sub = expr_df.loc[available_genes]
    z_scores = (expr_sub - expr_sub.mean(axis=1).values[:, None]) / (expr_sub.std(axis=1).values[:, None] + 1e-8)

    # Compute composite risk score for each sample
    scores = np.dot(weights, z_scores.values)

    # Binary ground truth (Tumor = 1, Normal = 0)
    samples = expr_df.columns.tolist()
    y_true = np.array([1 if labels.get(s, "Tumor") == "Tumor" else 0 for s in samples])

    # ROC analysis & Youden's J threshold
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal_cutoff = float(thresholds[best_idx])

    # Predicted classes at optimal cutoff
    y_pred = (scores >= optimal_cutoff).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, len(y_true))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # 95% Confidence Intervals via Bootstrap (200 iterations)
    np.random.seed(42)
    boot_sens, boot_spec, boot_auc = [], [], []
    for _ in range(200):
        boot_idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[boot_idx])) < 2:
            continue
        b_true = y_true[boot_idx]
        b_pred = y_pred[boot_idx]
        b_scores = scores[boot_idx]

        b_tp = np.sum((b_true == 1) & (b_pred == 1))
        b_fn = np.sum((b_true == 1) & (b_pred == 0))
        b_tn = np.sum((b_true == 0) & (b_pred == 0))
        b_fp = np.sum((b_true == 0) & (b_pred == 1))

        boot_sens.append(b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0)
        boot_spec.append(b_tn / (b_tn + b_fp) if (b_tn + b_fp) > 0 else 0.0)
        try:
            b_fpr, b_tpr, _ = roc_curve(b_true, b_scores)
            boot_auc.append(auc(b_fpr, b_tpr))
        except Exception:
            pass

    ci_sens = (round(float(np.percentile(boot_sens, 2.5)), 3), round(float(np.percentile(boot_sens, 97.5)), 3)) if boot_sens else (0.0, 1.0)
    ci_spec = (round(float(np.percentile(boot_spec, 2.5)), 3), round(float(np.percentile(boot_spec, 97.5)), 3)) if boot_spec else (0.0, 1.0)
    ci_auc = (round(float(np.percentile(boot_auc, 2.5)), 3), round(float(np.percentile(boot_auc, 97.5)), 3)) if boot_auc else (0.0, 1.0)

    # Patient Scores Table
    patient_records = []
    for i, s_id in enumerate(samples):
        sc = float(scores[i])
        actual = labels.get(s_id, "Tumor")
        pred_label = "Tumor" if sc >= optimal_cutoff else "Normal"
        patient_records.append({
            "sample_id": s_id,
            "actual_class": actual,
            "predicted_class": pred_label,
            "risk_score": round(sc, 3),
            "classification_result": "True Positive" if actual == "Tumor" and pred_label == "Tumor" else (
                "True Negative" if actual == "Normal" and pred_label == "Normal" else (
                    "False Positive" if actual == "Normal" and pred_label == "Tumor" else "False Negative"
                )
            )
        })

    scores_df = pd.DataFrame(patient_records)
    scores_df.to_csv(os.path.join(out_dir, "patient_risk_scores.csv"), index=False)

    perf_metrics = {
        "signature_size": len(available_genes),
        "signature_genes": available_genes,
        "optimal_decision_threshold": round(optimal_cutoff, 3),
        "roc_auc": round(float(roc_auc), 4),
        "roc_auc_95ci": [ci_auc[0], ci_auc[1]],
        "sensitivity": round(float(sensitivity), 4),
        "sensitivity_95ci": [ci_sens[0], ci_sens[1]],
        "specificity": round(float(specificity), 4),
        "specificity_95ci": [ci_spec[0], ci_spec[1]],
        "ppv": round(float(ppv), 4),
        "npv": round(float(npv), 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

    with open(os.path.join(out_dir, "diagnostic_performance.json"), "w", encoding="utf-8") as f:
        json.dump(perf_metrics, f, indent=2)

    logger.info(f"Composite signature risk scoring complete ({len(available_genes)} genes): ROC-AUC = {roc_auc:.4f}, Sensitivity = {sensitivity:.3f}, Specificity = {specificity:.3f}")

    return {
        "scores_df": scores_df,
        "metrics": perf_metrics,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


if __name__ == "__main__":
    np.random.seed(42)
    genes = ["TOP2A", "CDK1", "MKI67", "PCNA", "EPCAM", "CDH1", "CLDN1"]
    samples = [f"GSM{i}" for i in range(40)]
    lbls = pd.Series(["Tumor" if i < 25 else "Normal" for i in range(40)], index=samples)

    expr_m = pd.DataFrame(np.random.randn(len(genes), 40), index=genes, columns=samples)
    for i, g in enumerate(genes[:4]):
        expr_m.iloc[i, :25] += 2.5  # Upregulated in Tumor
    for i, g in enumerate(genes[4:], start=4):
        expr_m.iloc[i, :25] -= 2.0  # Downregulated in Tumor

    cons_m = pd.DataFrame({
        "gene": genes,
        "consensus_score": [0.95, 0.90, 0.88, 0.82, 0.79, 0.75, 0.71],
        "log2FC": [2.5, 2.2, 1.9, 1.8, -1.7, -1.5, -1.4]
    })

    res = calculate_composite_risk_score(expr_m, lbls, cons_m, top_n=7)
    print("Risk Scoring Test Metrics:")
    print(json.dumps(res["metrics"], indent=2))
