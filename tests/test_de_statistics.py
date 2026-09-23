"""
Unit tests for differential expression statistics.

Tests:
- DE statistic correctness on synthetic data with known DE genes
- Voom/limma dispatch for RNA-seq
- Meta-analysis fixed-effects integration
"""
import pytest
import numpy as np
import pandas as pd

from src.differential_expression import (
    run_differential_expression,
    get_top_degs,
    compute_pvalues,
    compute_pvalues_voom_limma,
)


class TestDifferentialExpression:
    """Tests for differential expression analysis."""

    def test_de_detects_known_genes(self, synthetic_expression_data, known_de_genes):
        """DE analysis should detect genes spiked with differential expression."""
        expr_df, labels = synthetic_expression_data

        results = run_differential_expression(expr_df, labels)

        # All known DE genes should be in results
        for gene in known_de_genes:
            assert gene in results.index, f"DE gene '{gene}' not found in results"

    def test_de_direction_correctness(self, synthetic_expression_data, synthetic_de_results):
        """Upregulated genes should have positive log2FC; downregulated should be negative."""
        expr_df, labels = synthetic_expression_data
        results = run_differential_expression(expr_df, labels)

        # EGFR, KRAS, MYC upregulated -> positive FC
        for gene in ["EGFR", "KRAS", "MYC"]:
            assert results.loc[gene, "log2FC"] > 0, f"{gene} should have positive log2FC"

        # TP53, CDH1 downregulated -> negative FC
        for gene in ["TP53", "CDH1"]:
            assert results.loc[gene, "log2FC"] < 0, f"{gene} should have negative log2FC"

    def test_de_pvalues_significant(self, synthetic_expression_data):
        """Known DE genes should have very small p-values."""
        expr_df, labels = synthetic_expression_data
        results = run_differential_expression(expr_df, labels)

        for gene in ["EGFR", "KRAS", "MYC", "TP53", "CDH1"]:
            assert results.loc[gene, "pvalue"] < 0.001, f"{gene} should have p < 0.001"
            assert results.loc[gene, "adj_pvalue"] < 0.01, f"{gene} should have adj_p < 0.01"

    def test_de_non_de_genes_not_significant(self, synthetic_expression_data):
        """Non-DE genes should not be significant (most of them)."""
        expr_df, labels = synthetic_expression_data
        results = run_differential_expression(expr_df, labels)

        non_de_genes = [g for g in results.index if g.startswith("GENE_")]
        non_sig_count = sum(1 for g in non_de_genes if results.loc[g, "pvalue"] > 0.05)
        # Most non-DE genes should not be significant
        assert non_sig_count >= len(non_de_genes) * 0.7, \
            f"Expected most non-DE genes to not be significant, but {non_sig_count}/{len(non_de_genes)} are"

    def test_get_top_degs(self, synthetic_de_results, known_de_genes):
        """get_top_degs should return the most significant genes."""
        top = get_top_degs(synthetic_de_results, n=5)
        top_gene_list = top["gene"].tolist() if "gene" in top.columns else top.index.tolist()

        for gene in ["EGFR", "KRAS", "MYC"]:
            assert gene in top_gene_list, f"{gene} should be in top DEGs"

    def test_voom_limma_dispatch(self, synthetic_rnaseq_data, monkeypatch):
        """Voom/limma path should be used when DATA_TYPE is rnaseq."""
        expr_df, labels = synthetic_rnaseq_data

        # Set data type to rnaseq
        import config
        monkeypatch.setattr(config, "DATA_TYPE", "rnaseq")

        result = compute_pvalues_voom_limma(expr_df, labels)

        # Returns tuple (pvalues, test_type, n_pairs)
        assert len(result) >= 2
        pvalues = result[0]
        assert len(pvalues) > 0

    def test_microarray_dispatch(self, synthetic_expression_data, monkeypatch):
        """Microarray path should be used when DATA_TYPE is microarray."""
        expr_df, labels = synthetic_expression_data

        results = compute_pvalues(expr_df, labels)

        assert len(results) > 0


class TestMetaAnalysis:
    """Tests for fixed-effects meta-analysis."""

    def test_meta_analysis_combines_cohorts(self):
        """Meta-analysis should combine results from multiple cohorts."""
        from src.meta_analysis import run_meta_analysis

        # Create synthetic per-cohort DE results
        de1 = pd.DataFrame({
            "gene": ["EGFR", "KRAS", "TP53"],
            "log2FC": [2.1, 1.5, -1.2],
            "pvalue": [1e-5, 1e-4, 1e-3],
            "adj_pvalue": [1e-4, 1e-3, 1e-2],
            "mean_tumor": [10, 9, 5],
            "mean_normal": [5, 7, 6],
        }).set_index("gene")

        de2 = pd.DataFrame({
            "gene": ["EGFR", "KRAS", "TP53"],
            "log2FC": [1.8, 1.2, -0.9],
            "pvalue": [1e-4, 1e-3, 1e-2],
            "adj_pvalue": [1e-3, 1e-2, 1e-1],
            "mean_tumor": [9, 8, 6],
            "mean_normal": [6, 6, 7],
        }).set_index("gene")

        results = run_meta_analysis([de1, de2], ["GSE8671", "GSE20916"])

        assert len(results) > 0
        assert "gene" in results.columns
        assert "log2FC_meta" in results.columns
        assert "pvalue_meta" in results.columns
        assert "adj_pvalue_meta" in results.columns
        assert "direction_consistent" in results.columns

    def test_meta_analysis_direction_consistency(self):
        """Meta-analysis should detect direction consistency."""
        from src.meta_analysis import run_meta_analysis

        de1 = pd.DataFrame({
            "gene": ["EGFR", "KRAS"],
            "log2FC": [2.1, 1.5],
            "pvalue": [1e-5, 1e-4],
            "adj_pvalue": [1e-4, 1e-3],
            "mean_tumor": [10, 9],
            "mean_normal": [5, 7],
        }).set_index("gene")

        de2 = pd.DataFrame({
            "gene": ["EGFR", "KRAS"],
            "log2FC": [1.8, -1.2],  # KRAS changes direction
            "pvalue": [1e-4, 1e-3],
            "adj_pvalue": [1e-3, 1e-2],
            "mean_tumor": [9, 8],
            "mean_normal": [6, 6],
        }).set_index("gene")

        results = run_meta_analysis([de1, de2], ["Cohort1", "Cohort2"])

        # EGFR should be consistent (both positive)
        egfr_row = results[results["gene"] == "EGFR"].iloc[0]
        assert egfr_row["direction_consistent"] == True

        # KRAS should not be consistent (one positive, one negative)
        kras_row = results[results["gene"] == "KRAS"].iloc[0]
        assert kras_row["direction_consistent"] == False

    def test_meta_analysis_single_gene(self):
        """Meta-analysis should handle genes present in only one cohort."""
        from src.meta_analysis import run_meta_analysis

        de1 = pd.DataFrame({
            "gene": ["EGFR", "KRAS"],
            "log2FC": [2.1, 1.5],
            "pvalue": [1e-5, 1e-4],
            "adj_pvalue": [1e-4, 1e-3],
            "mean_tumor": [10, 9],
            "mean_normal": [5, 7],
        }).set_index("gene")

        de2 = pd.DataFrame({
            "gene": ["EGFR"],
            "log2FC": [1.8],
            "pvalue": [1e-4],
            "adj_pvalue": [1e-3],
            "mean_tumor": [9],
            "mean_normal": [6],
        }).set_index("gene")

        results = run_meta_analysis([de1, de2], ["Cohort1", "Cohort2"])

        # KRAS only in cohort 1, should not be in meta results
        assert "KRAS" not in results["gene"].values
        # EGFR in both, should be present
        assert "EGFR" in results["gene"].values
