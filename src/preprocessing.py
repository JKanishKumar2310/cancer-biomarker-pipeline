"""
Preprocessing — Normalization, filtering, and preparation of expression data.

Handles log2 transformation, low-variance gene removal, quantile normalization,
and probe-to-gene symbol collapsing.
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
    Quantile normalization across samples.
    Ensures all samples have the same distribution of expression values.
    """
    logger.info("Applying quantile normalization...")

    rank_mean = expr_df.stack().groupby(
        expr_df.rank(method="first").stack().astype(int)
    ).mean()

    normalized = expr_df.rank(method="min").stack().astype(int).map(rank_mean).unstack()
    normalized.index = expr_df.index
    normalized.columns = expr_df.columns

    return normalized


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
