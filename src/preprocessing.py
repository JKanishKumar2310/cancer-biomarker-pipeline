"""
Preprocessing — Normalization, filtering, and preparation of expression data.

Handles:
- Microarray: log2 transformation, quantile normalization, probe-to-gene collapsing
- RNA-seq: TMM/median-of-ratios normalization, voom transformation, count filtering
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger


def log2_transform(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Apply log2(x + 1) transformation to handle raw expression values."""
    if config.LOG2_TRANSFORM:
        # Check if data appears already log-transformed (max < 30 suggests log scale)
        if expr_df.max().max() > 100:
            logger.info("Applying log2(x + 1) transformation...")
            expr_df = np.log2(expr_df + 1)
        else:
            logger.info("Data appears already log-transformed, skipping log2")
    return expr_df


def normalize_rnaseq_counts(counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize RNA-seq raw counts using TMM or median-of-ratios method,
    followed by voom transformation (log2 CPM with precision weights).
    """
    data_type = getattr(config, "DATA_TYPE", "microarray")
    if data_type != "rnaseq":
        return counts_df
    
    norm_method = getattr(config, "RNASEQ_NORMALIZATION", "tmm")
    
    if norm_method == "tmm":
        logger.info("Applying TMM normalization for RNA-seq...")
        normalized = _tmm_normalize(counts_df)
    elif norm_method == "median_of_ratios":
        logger.info("Applying median-of-ratios (DESeq2) normalization for RNA-seq...")
        normalized = _median_of_ratios_normalize(counts_df)
    else:
        logger.info("Using simple CPM normalization for RNA-seq...")
        lib_sizes = counts_df.sum(axis=0)
        normalized = counts_df.div(lib_sizes, axis=1) * 1e6
    
    # Voom transformation: log2 CPM. TMM and median-of-ratios both output
    # normalized counts on the original library scale; they must still be
    # converted to CPM before the log2 transform. Previously the TMM branch
    # divided by the RAW library sizes (double-scaling), and the
    # median-of-ratios branch was logged as CPM without any conversion at all.
    logger.info("Applying voom transformation (log2 CPM)...")
    lib_sizes = normalized.sum(axis=0)
    cpm = normalized.div(lib_sizes, axis=1) * 1e6
    
    log_cpm = np.log2(cpm + 0.5)
    
    # Filter low-count genes (already done in data_loader, but double-check)
    min_counts = getattr(config, "RNASEQ_MIN_COUNTS", 10)
    min_samples = getattr(config, "RNASEQ_MIN_SAMPLES", 3)
    keep = (counts_df >= min_counts).sum(axis=1) >= min_samples
    log_cpm = log_cpm[keep]
    
    logger.info(f"RNA-seq preprocessing: {log_cpm.shape[0]} genes × {log_cpm.shape[1]} samples")
    return log_cpm


def _tmm_normalize(counts: pd.DataFrame) -> pd.DataFrame:
    """
    TMM (Trimmed Mean of M-values) normalization - edgeR algorithm.
    Pure Python implementation.
    """
    logger.info("  Computing TMM size factors...")
    
    # Reference: sample with upper quartile closest to median
    upper_quartiles = counts.quantile(0.75, axis=0)
    ref_idx = upper_quartiles.sub(upper_quartiles.median()).abs().idxmin()
    ref = counts[ref_idx]
    
    tmm_factors = []
    for col in counts.columns:
        sample = counts[col]
        # Avoid division by zero
        ref_safe = ref + 0.5
        sample_safe = sample + 0.5
        
        # M and A values
        m = np.log2(sample_safe / ref_safe)
        a = 0.5 * np.log2(sample_safe * ref_safe)
        
        # Trim extreme M and A values (default: 30% M, 5% A)
        m_finite = m[np.isfinite(m)]
        if len(m_finite) < 5:
            tmm_factors.append(1.0)
            continue
        
        m_low, m_high = np.percentile(m_finite, 30), np.percentile(m_finite, 70)
        a_finite = a[np.isfinite(a)]
        a_low, a_high = np.percentile(a_finite, 5), np.percentile(a_finite, 95)
        
        keep = (m >= m_low) & (m <= m_high) & (a >= a_low) & (a <= a_high) & np.isfinite(m) & np.isfinite(a)
        
        if keep.sum() > 10:
            tmm = 2 ** np.mean(m[keep])
        else:
            tmm = 1.0
        tmm_factors.append(tmm)
    
    tmm_factors = np.array(tmm_factors)
    # Normalize so geometric mean of factors = 1
    tmm_factors = tmm_factors / np.exp(np.mean(np.log(tmm_factors)))
    
    # Apply normalization
    normalized = counts.div(tmm_factors, axis=1)
    logger.info(f"  TMM factors computed: mean={np.mean(tmm_factors):.4f}")
    return normalized


def _median_of_ratios_normalize(counts: pd.DataFrame) -> pd.DataFrame:
    """
    DESeq2 median-of-ratios normalization.
    Pure Python implementation.
    """
    logger.info("  Computing DESeq2 size factors...")
    
    # Only use genes with counts > 1 for geometric mean
    gene_means = counts.mean(axis=1)
    usable = counts[gene_means > 1]
    
    # Geometric mean per gene
    log_counts = np.log(usable + 1)
    geo_means = np.exp(log_counts.mean(axis=1))
    
    # Ratios
    ratios = usable.div(geo_means, axis=0)
    
    # Size factors = median of ratios per sample
    size_factors = ratios.median(axis=0)
    size_factors = size_factors / np.exp(np.mean(np.log(size_factors)))
    
    # Apply normalization
    normalized = counts.div(size_factors, axis=1)
    logger.info(f"  Size factors computed: mean={np.mean(size_factors):.4f}")
    return normalized


def remove_low_variance_genes(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Remove genes in the bottom percentile of variance across samples."""
    variances = expr_df.var(axis=1)
    threshold = np.percentile(variances, config.LOW_VARIANCE_PERCENTILE)
    mask = variances > threshold

    # Preserve known markers so they can always be verified
    for marker in config.KNOWN_MARKERS:
        if marker in mask.index:
            mask.loc[marker] = True

    n_before = expr_df.shape[0]
    expr_df = expr_df[mask]
    n_removed = n_before - expr_df.shape[0]
    logger.info(
        f"Variance filtering: {n_before} → {expr_df.shape[0]} genes "
        f"(removed {n_removed} below {config.LOW_VARIANCE_PERCENTILE}th percentile, preserved known markers)"
    )
    return expr_df


def quantile_normalize(expr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast, memory-efficient Quantile Normalization across samples.
    Ensures all samples have the same distribution of expression values.
    """
    logger.info("Applying quantile normalization...")
    # Impute missing values with row median if any exist
    if expr_df.isna().any().any():
        expr_df = expr_df.apply(lambda row: row.fillna(row.median()), axis=1).fillna(0.0)

    vals = expr_df.values.astype(float)
    sorted_vals = np.sort(vals, axis=0)
    rank_means = np.mean(sorted_vals, axis=1)
    ranks = np.argsort(np.argsort(vals, axis=0), axis=0)
    normalized_vals = rank_means[ranks]

    return pd.DataFrame(normalized_vals, index=expr_df.index, columns=expr_df.columns)


def collapse_probes_to_genes(expr_df: pd.DataFrame) -> pd.DataFrame:
    """
    If index contains probe IDs (numeric-looking), try to keep gene-symbol-like
    indices and collapse duplicates by taking the mean.
    """
    # Check if index looks like gene symbols (alphabetic) or probe IDs (numeric)
    sample_idx = expr_df.index[:10].tolist()
    has_gene_names = any(
        isinstance(idx, str) and idx[0].isalpha() for idx in sample_idx
    )

    if has_gene_names:
        # Collapse duplicate gene symbols by taking the mean
        n_before = expr_df.shape[0]
        expr_df = expr_df.groupby(expr_df.index).mean()
        n_after = expr_df.shape[0]
        if n_before != n_after:
            logger.info(
                f"Collapsed duplicate gene symbols: {n_before} → {n_after} unique genes"
            )
    else:
        logger.info("Index appears to be probe IDs (will be used as-is)")

    return expr_df


def remove_constant_genes(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Remove genes with zero variance (constant across all samples)."""
    mask = expr_df.var(axis=1) > 0
    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.info(f"Removed {n_removed} constant genes")
    return expr_df[mask]


def preprocess(
    expr_df: pd.DataFrame,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Raw expression matrix (genes × samples).
    labels : pd.Series
        Sample labels ('Tumor' or 'Normal').

    Returns
    -------
    expr_clean : pd.DataFrame
        Preprocessed expression matrix (genes × samples).
    labels : pd.Series
        Aligned labels (unchanged).
    """
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # Ensure alignment
    common = expr_df.columns.intersection(labels.index)
    expr_df = expr_df[common]
    labels = labels[common]
    logger.info(f"Starting with {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")

    data_type = getattr(config, "DATA_TYPE", "microarray")
    
    if data_type == "rnaseq":
        # RNA-seq pipeline
        logger.info("RNA-seq preprocessing pipeline selected")
        # Step 1: Normalize counts (TMM or median-of-ratios) + voom transform
        expr_df = normalize_rnaseq_counts(expr_df)
        # Step 2: Remove constant genes
        expr_df = remove_constant_genes(expr_df)
        # Step 3: Collapse duplicate gene symbols
        expr_df = collapse_probes_to_genes(expr_df)
        # Step 4: Remove low-variance genes
        expr_df = remove_low_variance_genes(expr_df)
    else:
        # Microarray pipeline
        logger.info("Microarray preprocessing pipeline selected")
        # Step 1: Log2 transform
        expr_df = log2_transform(expr_df)
        # Step 2: Remove constant genes
        expr_df = remove_constant_genes(expr_df)
        # Step 3: Collapse probes to genes
        expr_df = collapse_probes_to_genes(expr_df)
        # Step 4: Quantile normalization
        expr_df = quantile_normalize(expr_df)
        # Step 5: Remove low-variance genes
        expr_df = remove_low_variance_genes(expr_df)

    logger.info(f"Final shape: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")
    logger.info("Preprocessing complete ✓")

    return expr_df, labels


if __name__ == "__main__":
    from src.data_loader import load_data

    expr, labels = load_data()
    expr_clean, labels_clean = preprocess(expr, labels)
    print(f"\nCleaned: {expr_clean.shape}")
    print(f"Labels: {labels_clean.value_counts().to_dict()}")
