"""
Shared test fixtures for the Cancer Biomarker Discovery Pipeline.

Provides synthetic expression data, known DE genes, and known HR genes
that tests can use without network access.
"""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def synthetic_expression_data():
    """
    Generate synthetic gene expression data with known DE genes.

    - 100 genes total
    - 30 tumor samples, 20 normal samples
    - 'EGFR', 'KRAS', 'MYC' are upregulated in tumor
    - 'TP53', 'CDH1' are downregulated in tumor
    """
    rng = np.random.RandomState(42)
    n_genes = 100
    n_tumor = 30
    n_normal = 20

    genes = [f"GENE_{i:03d}" for i in range(n_genes)]
    de_genes_up = ["EGFR", "KRAS", "MYC"]
    de_genes_down = ["TP53", "CDH1"]

    # Replace first few genes with known markers
    for i, gene in enumerate(de_genes_up + de_genes_down):
        if i < n_genes:
            genes[i] = gene

    # Generate base expression
    expr_tumor = rng.randn(n_genes, n_tumor) + 5
    expr_normal = rng.randn(n_genes, n_normal) + 5

    # Make DE genes differentially expressed
    for i, gene in enumerate(de_genes_up):
        if gene in genes:
            idx = genes.index(gene)
            expr_tumor[idx] += 3.0  # Upregulate

    for i, gene in enumerate(de_genes_down):
        if gene in genes:
            idx = genes.index(gene)
            expr_tumor[idx] -= 2.5  # Downregulate

    all_expr = np.hstack([expr_tumor, expr_normal])
    all_labels = ["Tumor"] * n_tumor + ["Normal"] * n_normal

    expr_df = pd.DataFrame(all_expr, index=genes, columns=[f"Sample_{i}" for i in range(len(all_labels))])
    labels = pd.Series(all_labels, index=expr_df.columns)

    return expr_df, labels


@pytest.fixture
def synthetic_rnaseq_data():
    """
    Generate synthetic RNA-seq count data with known DE genes.

    Returns log2CPM-normalized expression (as output of normalize_rnaseq_counts)
    """
    rng = np.random.RandomState(123)
    n_genes = 80
    n_tumor = 15
    n_normal = 15

    genes = [f"GENE_{i:03d}" for i in range(n_genes)]
    de_genes_up = ["EGFR", "KRAS"]
    de_genes_down = ["TP53"]

    for i, gene in enumerate(de_genes_up + de_genes_down):
        if i < n_genes:
            genes[i] = gene

    # Generate count-like data (then apply log2 transform to simulate normalized)
    counts_tumor = rng.negative_binomial(n=10, p=0.5, size=(n_genes, n_tumor)) + 10
    counts_normal = rng.negative_binomial(n=10, p=0.5, size=(n_genes, n_normal)) + 10

    for gene in de_genes_up:
        if gene in genes:
            idx = genes.index(gene)
            counts_tumor[idx] *= 3

    for gene in de_genes_down:
        if gene in genes:
            idx = genes.index(gene)
            counts_tumor[idx] //= 3

    all_counts = np.hstack([counts_tumor, counts_normal]).astype(float)
    log_cpm = np.log2(all_counts + 0.5)

    all_labels = ["Tumor"] * n_tumor + ["Normal"] * n_normal

    expr_df = pd.DataFrame(log_cpm, index=genes, columns=[f"S{i}" for i in range(len(all_labels))])
    labels = pd.Series(all_labels, index=expr_df.columns)

    return expr_df, labels


@pytest.fixture
def known_de_genes():
    """Genes that should be detected as significantly differentially expressed."""
    return ["EGFR", "KRAS", "MYC", "TP53", "CDH1"]


@pytest.fixture
def known_upregulated():
    return ["EGFR", "KRAS", "MYC"]


@pytest.fixture
def known_downregulated():
    return ["TP53", "CDH1"]


@pytest.fixture
def synthetic_de_results():
    """
    Synthetic DE results DataFrame with known markers.

    Returns gene-indexed DataFrame with log2FC, pvalue, adj_pvalue columns.
    """
    genes = ["EGFR", "KRAS", "MYC", "TP53", "CDH1", "GENE_005", "GENE_006", "GENE_007"]
    return pd.DataFrame({
        "gene": genes,
        "log2FC": [3.1, 2.8, 3.2, -2.6, -2.3, 0.2, -0.1, 0.3],
        "pvalue": [1e-7, 1e-6, 1e-8, 1e-5, 1e-4, 0.5, 0.8, 0.3],
        "adj_pvalue": [1e-6, 1e-5, 1e-7, 1e-4, 1e-3, 0.6, 0.9, 0.4],
        "mean_tumor": [10, 9, 11, 5, 4, 3, 5, 2],
        "mean_normal": [5, 7, 3, 8, 7, 3, 5, 2],
        "regulation": ["Upregulated", "Upregulated", "Upregulated", "Downregulated", "Downregulated", "Not Significant", "Not Significant", "Not Significant"],
    }).set_index("gene")


@pytest.fixture
def synthetic_survival_data():
    """
    Synthetic survival data with a known HR gene.

    'EGFR' high expression -> HR = 2.5 (worse survival)
    """
    rng = np.random.RandomState(456)
    n_samples = 50

    egfr_high = rng.randn(n_samples) + 5
    egfr_low = rng.randn(n_samples) - 2

    # Simulate survival times
    time_high = rng.exponential(scale=100 / 2.5, size=n_samples) + 50
    time_low = rng.exponential(scale=100, size=n_samples) + 50

    # Events (~70% have events)
    event_high = rng.binomial(1, 0.7, n_samples)
    event_low = rng.binomial(1, 0.7, n_samples)

    df = pd.DataFrame({
        "gene": "EGFR",
        "high_expr": egfr_high[:n_samples],
        "low_expr": egfr_low[:n_samples],
        "time_high": time_high,
        "time_low": time_low,
        "event_high": event_high,
        "event_low": event_low,
    })

    return df


@pytest.fixture
def known_hr_genes():
    """Genes with known hazard ratios in synthetic data."""
    return {"EGFR": 2.5}
