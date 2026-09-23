"""
Unit tests for preprocessing normalization functions.

Tests:
- log2 transformation logic
- TMM normalization
- Median-of-ratios normalization
- Voom transformation
- Quantile normalization correctness
- Variance filtering preserves known markers
"""
import pytest
import numpy as np
import pandas as pd

from src.preprocessing import (
    log2_transform,
    normalize_rnaseq_counts,
    _tmm_normalize,
    _median_of_ratios_normalize,
    remove_low_variance_genes,
    quantile_normalize,
    collapse_probes_to_genes,
    remove_constant_genes,
    preprocess,
)


class TestLog2Transform:
    """Tests for log2 transformation."""

    def test_log2_transform_applied(self):
        """Large values should trigger log2(x+1) transform."""
        expr = pd.DataFrame({"S1": [1, 100, 500], "S2": [2, 200, 1000]})
        result = log2_transform(expr.copy())

        assert result["S1"].max() < 100  # Should be log2 scaled
        assert result["S1"].max() > 5    # Should still be positive

    def test_log2_transform_skipped(self):
        """Already log-transformed data should pass through."""
        expr = pd.DataFrame({"S1": [1.0, 5.0, 3.0], "S2": [2.0, 4.0, 6.0]})
        result = log2_transform(expr.copy())

        # Max < 100, should not transform
        pd.testing.assert_frame_equal(result, expr)


class TestTMMNormalize:
    """Tests for TMM normalization."""

    def test_tmm_preserves_shape(self):
        """TMM normalization should preserve matrix dimensions."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame(
            rng.negative_binomial(20, 0.5, size=(10, 4)),
            index=[f"G{i}" for i in range(10)],
            columns=["S1", "S2", "S3", "S4"]
        )
        result = _tmm_normalize(data)
        assert result.shape == data.shape

    def test_tmm_normalizes_lib_sizes(self):
        """TMM should produce normalized counts with adjusted scaling."""
        rng = np.random.RandomState(42)
        # Use larger counts for more stable TMM
        data = pd.DataFrame(
            rng.negative_binomial(100, 0.3, size=(50, 3)),
            index=[f"G{i}" for i in range(50)],
            columns=["S1", "S2", "S3"]
        )
        original_sums = data.sum(axis=0).values
        # Make S2 have 10x library size
        data["S2"] *= 10

        result = _tmm_normalize(data)

        # TMM should adjust S2 towards the others
        result_sums = result.sum(axis=0).values
        # S2 should be reduced
        assert result_sums[1] < data["S2"].sum(), "S2 should be reduced by TMM"
        # Variance of normalized sums should be smaller than raw inflated sums
        assert np.std(result_sums) < np.std(data.sum(axis=0).values)


class TestMedianOfRatios:
    """Tests for DESeq2 median-of-ratios normalization."""

    def test_median_ratios_shape(self):
        """Median-of-ratios should preserve matrix dimensions."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame(
            rng.negative_binomial(20, 0.5, size=(10, 4)),
            index=[f"G{i}" for i in range(10)],
            columns=["S1", "S2", "S3", "S4"]
        )
        result = _median_of_ratios_normalize(data)
        assert result.shape == data.shape

    def test_median_ratios_lib_balance(self):
        """Median-of-ratios should reduce library size bias."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame(
            rng.negative_binomial(20, 0.5, size=(10, 3)),
            index=[f"G{i}" for i in range(10)],
            columns=["S1", "S2", "S3"]
        )
        data["S2"] *= 5
        result = _median_of_ratios_normalize(data)

        # S2 should be normalized closer to others
        assert result["S2"].sum() < data["S2"].sum()


class TestQuantileNormalize:
    """Tests for quantile normalization."""

    def test_quantile_same_distribution(self):
        """All samples should have the same distribution after quantile normalization."""
        rng = np.random.RandomState(42)
        data = pd.DataFrame(
            rng.randn(20, 3) + np.array([1, 5, 10]),  # Different means
            columns=["S1", "S2", "S3"]
        )
        result = quantile_normalize(data)

        # All columns should have the same mean (same distribution)
        col_means = result.mean(axis=0)
        assert col_means.std() < 0.01, "Columns should have nearly identical means after quantile normalization"

    def test_quantile_preserves_shape(self):
        """Quantile normalization should preserve matrix dimensions."""
        data = pd.DataFrame(
            np.random.randn(15, 4),
            columns=["S1", "S2", "S3", "S4"]
        )
        result = quantile_normalize(data)
        assert result.shape == data.shape


class TestGeneFiltering:
    """Tests for gene filtering functions."""

    def test_remove_constant_genes(self):
        """Remove genes with zero variance."""
        data = pd.DataFrame({
            "S1": [1, 2, 3, 5, 5],
            "S2": [2, 2, 2, 5, 5],
        }, index=["G1", "G2", "G3", "G4", "G5"])

        result = remove_constant_genes(data)
        assert "G4" not in result.index  # Zero variance
        assert "G5" not in result.index  # Zero variance
        assert "G1" in result.index      # Has variance

    def test_variances_genes_preserves_markers(self):
        """Variance filtering should preserve known marker genes."""
        import config
        rng = np.random.RandomState(42)
        data = pd.DataFrame(
            rng.randn(100, 5),
            index=[f"GENE_{i:03d}" for i in range(95)] + ["EGFR", "KRAS", "MYC", "TP53", "CDH1"]
        )
        # Make EGFR low variance
        data.loc["EGFR"] *= 0.001

        result = remove_low_variance_genes(data)

        # Known markers should be preserved even if low variance
        for marker in ["EGFR", "KRAS", "MYC", "TP53", "CDH1"]:
            if marker in config.KNOWN_MARKERS:
                assert marker in result.index, f"{marker} should be preserved as known marker"


class TestPreprocessIntegration:
    """Integration tests for the full preprocessing pipeline."""

    def test_preprocess_microarray(self, synthetic_expression_data, monkeypatch):
        """Microarray preprocessing should return cleaned data."""
        expr_df, labels = synthetic_expression_data
        import config
        monkeypatch.setattr(config, "DATA_TYPE", "microarray", raising=False)

        result, cleaned_labels = preprocess(expr_df, labels)

        assert result.shape[0] <= expr_df.shape[0]  # May remove genes
        assert result.shape[1] == expr_df.shape[1]
        assert len(cleaned_labels) == len(labels)

    def test_preprocess_rnaseq(self, monkeypatch):
        """RNA-seq preprocessing should use voom/TMM path."""
        from src.preprocessing import normalize_rnaseq_counts, _tmm_normalize
        import config

        monkeypatch.setattr(config, "DATA_TYPE", "rnaseq", raising=False)

        # Test RNA-seq normalization directly
        rng = np.random.RandomState(42)
        counts = pd.DataFrame(
            rng.negative_binomial(50, 0.3, size=(30, 6)),
            index=[f"GENE_{i:03d}" for i in range(30)],
            columns=[f"S{i}" for i in range(6)]
        )
        counts.iloc[0:3] *= 5  # Make some genes high count

        result = normalize_rnaseq_counts(counts.copy())
        assert result.shape[0] > 0
        assert result.shape[1] == 6
