"""
Differential Expression Analysis — Statistical identification of DEGs.

Performs Welch's t-test (tumor vs normal) with Benjamini-Hochberg FDR correction
to identify differentially expressed genes (DEGs).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger


def compute_fold_change(
    expr_df: pd.DataFrame,
    labels: pd.Series,
) -> pd.Series:
    """
    Compute log2 fold change: mean(Tumor) - mean(Normal).
    Positive = upregulated in tumor, Negative = downregulated.
    """
    tumor_samples = labels[labels == "Tumor"].index
    normal_samples = labels[labels == "Normal"].index

    mean_tumor = expr_df[tumor_samples].mean(axis=1)
    mean_normal = expr_df[normal_samples].mean(axis=1)

    log2fc = mean_tumor - mean_normal  # Already in log2 space
    return log2fc


def compute_pvalues(
    expr_df: pd.DataFrame,
    labels: pd.Series,
) -> pd.Series:
    """
    Compute per-gene p-values using Welch's t-test (unequal variance).
    """
    tumor_samples = labels[labels == "Tumor"].index
    normal_samples = labels[labels == "Normal"].index

    tumor_expr = expr_df[tumor_samples]
    normal_expr = expr_df[normal_samples]

    pvalues = []
    for gene in expr_df.index:
        t_stat, p_val = stats.ttest_ind(
            tumor_expr.loc[gene].values,
            normal_expr.loc[gene].values,
            equal_var=False,  # Welch's t-test
            nan_policy="omit",
        )
        pvalues.append(p_val)

    return pd.Series(pvalues, index=expr_df.index, name="pvalue")


def adjust_pvalues(pvalues: pd.Series) -> pd.Series:
    """
    Apply Benjamini-Hochberg FDR correction for multiple testing.
    """
    # Handle NaN p-values
    valid_mask = ~pvalues.isna()
    adjusted = np.full(len(pvalues), np.nan)

    if valid_mask.sum() > 0:
        reject, adj_pvals, _, _ = multipletests(
            pvalues[valid_mask].values,
            alpha=config.PVALUE_THRESHOLD,
            method="fdr_bh",
        )
        adjusted[valid_mask.values] = adj_pvals

    return pd.Series(adjusted, index=pvalues.index, name="adj_pvalue")


def categorize_genes(log2fc: pd.Series, adj_pvalue: pd.Series) -> pd.Series:
    """
    Categorize each gene as Upregulated, Downregulated, or Not Significant.
    """
    categories = pd.Series("Not Significant", index=log2fc.index, name="regulation")

    sig_mask = adj_pvalue < config.PVALUE_THRESHOLD

    up_mask = sig_mask & (log2fc > config.FC_THRESHOLD)
    down_mask = sig_mask & (log2fc < -config.FC_THRESHOLD)

    categories[up_mask] = "Upregulated"
    categories[down_mask] = "Downregulated"

    return categories


def run_differential_expression(
    expr_df: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    """
    Run the full differential expression analysis pipeline.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Preprocessed expression matrix (genes × samples).
    labels : pd.Series
        Sample labels ('Tumor' or 'Normal').

    Returns
    -------
    de_results : pd.DataFrame
        Columns: gene, log2FC, pvalue, adj_pvalue, regulation,
                 mean_tumor, mean_normal, neg_log10_pval
    """
    logger.info("=" * 60)
    logger.info("DIFFERENTIAL EXPRESSION ANALYSIS")
    logger.info("=" * 60)

    tumor_samples = labels[labels == "Tumor"].index
    normal_samples = labels[labels == "Normal"].index
    logger.info(f"Comparing {len(tumor_samples)} Tumor vs {len(normal_samples)} Normal samples")

    # Step 1: Fold change
    logger.info("Computing log2 fold changes...")
    log2fc = compute_fold_change(expr_df, labels)

    # Step 2: P-values
    logger.info("Running Welch's t-test for each gene...")
    pvalues = compute_pvalues(expr_df, labels)

    # Step 3: FDR correction
    logger.info("Applying Benjamini-Hochberg FDR correction...")
    adj_pvalues = adjust_pvalues(pvalues)

    # Step 4: Categorize
    regulation = categorize_genes(log2fc, adj_pvalues)

    # Build results DataFrame
    de_results = pd.DataFrame({
        "gene": expr_df.index,
        "log2FC": log2fc.values,
        "pvalue": pvalues.values,
        "adj_pvalue": adj_pvalues.values,
        "regulation": regulation.values,
        "mean_tumor": expr_df[tumor_samples].mean(axis=1).values,
        "mean_normal": expr_df[normal_samples].mean(axis=1).values,
    })
    de_results.index = expr_df.index

    # Add -log10(adj_pvalue) for volcano plot
    de_results["neg_log10_pval"] = -np.log10(
        de_results["adj_pvalue"].clip(lower=1e-300)
    )

    # Biological mechanism annotation for hallmark cancer genes
    mutation_drivers = {"KRAS", "TP53", "APC", "BRAF", "PIK3CA", "PTEN", "NRAS", "EGFR", "SMAD4"}
    post_translational = {"CTNNB1", "AKT1", "GSK3B", "SRC"}
    lineage_markers = {"EPCAM", "CDX2", "CEACAM5", "KRT20", "CDH1"}

    def get_mech_note(row):
        g = str(row["gene"]).upper()
        reg = row["regulation"]
        if reg != "Not Significant":
            return "Transcriptional Driver"
        if g in mutation_drivers:
            return "Somatic Mutation Driver (Non-Transcriptomic Activation)"
        if g in post_translational:
            return "Post-Translational Driver (Phosphorylation/Nuclear Translocation)"
        if g in lineage_markers:
            return "Epithelial Lineage Marker (High Baseline in Normal & Adenoma)"
        return "Not Significant"

    de_results["mechanistic_note"] = de_results.apply(get_mech_note, axis=1)

    # Sort by adjusted p-value
    de_results = de_results.sort_values("adj_pvalue")

    # Summary
    counts = regulation.value_counts()
    logger.info(f"Results: {counts.get('Upregulated', 0)} upregulated, "
                f"{counts.get('Downregulated', 0)} downregulated, "
                f"{counts.get('Not Significant', 0)} not significant")
    logger.info(f"Thresholds: |log2FC| > {config.FC_THRESHOLD}, adj_p < {config.PVALUE_THRESHOLD}")

    # Sanity check: are known markers detected?
    _sanity_check(de_results)

    # Save
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    de_results.to_csv(os.path.join(config.RESULTS_DIR, "de_results.csv"))
    logger.info("Saved differential expression results")

    logger.info("Differential expression analysis complete ✓")
    return de_results


def _sanity_check(de_results: pd.DataFrame) -> None:
    """Check known cancer markers across major biological pathways."""
    logger.info("Running verification across known cancer biomarkers...")
    found = []
    missing = []
    up_markers = []
    down_markers = []
    other_markers = []

    for marker in config.KNOWN_MARKERS:
        if marker in de_results.index:
            row = de_results.loc[marker]
            status = row["regulation"]
            fc = row["log2FC"]
            p = row["adj_pvalue"]
            found.append(marker)
            if status == "Upregulated":
                up_markers.append(f"{marker} (+{fc:.2f}, p={p:.1e})")
            elif status == "Downregulated":
                down_markers.append(f"{marker} ({fc:.2f}, p={p:.1e})")
            else:
                other_markers.append(f"{marker} ({fc:+.2f}, p={p:.1e})")
        else:
            missing.append(marker)

    logger.info(f"  Verified markers in dataset: {len(found)}/{len(config.KNOWN_MARKERS)}")
    if up_markers:
        logger.info(f"  ⬆ Significantly Upregulated: {', '.join(up_markers)}")
    if down_markers:
        logger.info(f"  ⬇ Significantly Downregulated: {', '.join(down_markers)}")
    if other_markers:
        logger.info(f"  • Other Verified Markers: {', '.join(other_markers[:8])}...")
    if missing:
        logger.warning(f"  ⚠ Not detected in array: {missing}")


def get_top_degs(de_results: pd.DataFrame, n: int = None) -> pd.DataFrame:
    """Get the top N differentially expressed genes (by adjusted p-value)."""
    if n is None:
        n = config.TOP_DE_GENES

    sig = de_results[de_results["regulation"] != "Not Significant"]
    return sig.head(n)


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.preprocessing import preprocess

    expr, labels = load_data()
    expr_clean, labels_clean = preprocess(expr, labels)
    de_results = run_differential_expression(expr_clean, labels_clean)
    print(f"\nTop 10 DEGs:")
    print(de_results.head(10)[["gene", "log2FC", "adj_pvalue", "regulation"]])
