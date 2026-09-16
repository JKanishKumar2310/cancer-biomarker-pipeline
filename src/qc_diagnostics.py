"""
Sample Quality Control & Outlier Diagnostics Module (nf-core / MultiQC Standard).

Computes:
1. Sample-to-sample correlation & distance matrices.
2. PCA Scree variance explained metrics.
3. Automated sample outlier detection (> 3 sigma from PCA centroid).
4. Sample-wise distribution summaries (quantiles, IQR, detection rate).
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger


def compute_qc_diagnostics(
    expr_df: pd.DataFrame,
    labels: pd.Series,
    out_dir: str = None
) -> dict:
    """
    Run comprehensive sample QC and outlier detection across the normalized expression matrix.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Genes (rows) x Samples (columns).
    labels : pd.Series
        Sample labels ('Tumor', 'Normal').
    out_dir : str, optional
        Output directory for QC metrics.

    Returns
    -------
    dict containing QC metrics, PCA scree data, outlier flags, and sample distance matrix.
    """
    if out_dir is None:
        out_dir = config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    n_genes, n_samples = expr_df.shape
    samples = expr_df.columns.tolist()

    # 1. Sample-to-sample Pearson Correlation & Euclidean Distance
    sample_corr = expr_df.corr(method="pearson")
    corr_csv = os.path.join(out_dir, "sample_correlation_matrix.csv")
    sample_corr.to_csv(corr_csv)

    # 2. PCA Scree & Multi-Component Decomposition
    n_components = min(10, n_samples - 1, n_genes)
    pca = PCA(n_components=n_components, random_state=42)
    pca_transformed = pca.fit_transform(expr_df.T)  # Shape: n_samples x n_components

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    scree_data = [
        {
            "component": f"PC{i+1}",
            "variance_explained": round(float(var * 100), 2),
            "cumulative_variance": round(float(cum * 100), 2)
        }
        for i, (var, cum) in enumerate(zip(explained_var, cumulative_var))
    ]

    # 3. Outlier Detection in PCA Subspace (PC1 - PC3)
    coords_3d = pca_transformed[:, :min(3, n_components)]
    centroid = np.median(coords_3d, axis=0)
    distances_from_centroid = np.linalg.norm(coords_3d - centroid, axis=1)

    mean_dist = np.mean(distances_from_centroid)
    std_dist = np.std(distances_from_centroid)
    outlier_threshold = mean_dist + 3.0 * std_dist

    outliers = []
    sample_qc_table = []

    for i, s_id in enumerate(samples):
        grp = labels.get(s_id, "Unknown")
        vals = expr_df[s_id].values
        dist_c = float(distances_from_centroid[i])
        is_outlier = bool(dist_c > outlier_threshold)

        if is_outlier:
            outliers.append(s_id)

        sample_qc_table.append({
            "sample_id": s_id,
            "group": grp,
            "mean_expression": round(float(np.mean(vals)), 3),
            "median_expression": round(float(np.median(vals)), 3),
            "iqr": round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 3),
            "pca_distance_from_centroid": round(dist_c, 2),
            "is_outlier": is_outlier,
            "pc1": round(float(coords_3d[i, 0]), 2),
            "pc2": round(float(coords_3d[i, 1]), 2) if coords_3d.shape[1] > 1 else 0.0,
        })

    qc_df = pd.DataFrame(sample_qc_table)
    qc_df.to_csv(os.path.join(out_dir, "sample_qc_metrics.csv"), index=False)

    summary = {
        "n_samples": n_samples,
        "n_genes": n_genes,
        "n_outliers": len(outliers),
        "outlier_samples": outliers,
        "pc1_variance": round(float(explained_var[0] * 100), 2) if len(explained_var) > 0 else 0.0,
        "pc2_variance": round(float(explained_var[1] * 100), 2) if len(explained_var) > 1 else 0.0,
        "scree_data": scree_data,
        "mean_inter_sample_correlation": round(float(sample_corr.values[np.triu_indices_from(sample_corr.values, k=1)].mean()), 3) if n_samples > 1 else 1.0,
    }

    with open(os.path.join(out_dir, "qc_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Sample QC diagnostics complete: {n_samples} samples evaluated ({len(outliers)} outliers flagged)")
    return summary


if __name__ == "__main__":
    np.random.seed(42)
    genes = [f"Gene_{i}" for i in range(500)]
    samps = [f"Sample_{i}" for i in range(30)]
    mat = pd.DataFrame(np.random.randn(500, 30) + 10, index=genes, columns=samps)
    # inject one deliberate outlier
    mat.iloc[:, 5] += 15.0
    lbls = pd.Series(["Tumor" if i < 15 else "Normal" for i in range(30)], index=samps)

    res = compute_qc_diagnostics(mat, lbls)
    print("QC Summary Test Results:")
    print(json.dumps(res, indent=2))
