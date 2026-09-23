"""
Regression tests for the 2026-09 bug-fix pass.

Each test pins a specific bug that was found during the project review:
1. L1 tuning wrote the tuned C into "LogisticRegression" (never read) instead of
   "L1_LogisticRegression" (the key the ensemble actually trains with).
2. L1 LogisticRegression used the deprecated penalty="l1" kwarg (sklearn >= 1.8).
3. normalize_rnaseq_counts double-scaled TMM output by RAW library sizes and
   never converted median-of-ratios output to CPM at all.
4. _combat_correct crashed when attrs["cohort"] was a scalar accession string.
5. external_validation trained on preprocessed features but labels from BEFORE
   preprocessing (row misalignment when preprocess drops samples).
6. Synthetic/test output paths nested a second "synthetic_test" directory when
   RESULTS_DIR already pointed at the synthetic_test folder.
"""
import numpy as np
import pandas as pd
import pytest

import config
from src.preprocessing import normalize_rnaseq_counts, _median_of_ratios_normalize
from src.data_loader import _combat_correct
from src.meta_analysis import _find_gene_in_df, run_meta_analysis


class TestL1TuningRegression:
    """Bug 1 + 2: L1 tuning must update the key that is actually used."""

    def test_tuned_c_written_to_l1_key(self, monkeypatch):
        from src.ml_biomarkers import run_ml_biomarker_ranking

        rng = np.random.RandomState(42)
        genes = [f"GENE_{i:03d}" for i in range(30)]
        n_t, n_n = 12, 12
        expr = pd.DataFrame(
            rng.randn(30, n_t + n_n) + 5,
            index=genes,
            columns=[f"T{i}" for i in range(n_t)] + [f"N{i}" for i in range(n_n)],
        )
        expr.iloc[:5, :n_t] += 3.0  # some DE genes
        labels = pd.Series(["Tumor"] * n_t + ["Normal"] * n_n,
                           index=expr.columns)

        monkeypatch.setattr(config, "ENABLE_L1_TUNING", True, raising=False)
        monkeypatch.setattr(config, "N_SPLITS", 3, raising=False)
        # Drop the bogus "LogisticRegression" key if a stale run added it
        monkeypatch.setattr(
            config, "ML_MODELS", dict(config.ML_MODELS), raising=False
        )

        _, consensus = run_ml_biomarker_ranking(expr, labels, _fake_de(expr, labels))

        # The tuned C must land on the key the ensemble reads...
        assert "C" in config.ML_MODELS["L1_LogisticRegression"]
        # ...and no stray wrongly-keyed entry may be created.
        assert "LogisticRegression" not in config.ML_MODELS

    def test_l1_model_config_uses_l1_ratio_not_penalty(self):
        """penalty='l1' is deprecated (removed in sklearn 1.10); use l1_ratio."""
        l1_params = config.ML_MODELS.get("L1_LogisticRegression", {})
        assert "penalty" not in l1_params, (
            "penalty='l1' triggers FutureWarning on sklearn>=1.8 and breaks on 1.10"
        )
        assert l1_params.get("l1_ratio") == 1
        assert l1_params.get("solver") == "liblinear"

    def test_l1_ratio_one_is_sparse(self):
        """l1_ratio=1 with liblinear must actually produce sparse coefficients."""
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(0)
        X = rng.randn(60, 20)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(C=0.05, l1_ratio=1, solver="liblinear",
                                   max_iter=1000, random_state=42)
        model.fit(X, y)
        assert (np.abs(model.coef_) > 0).sum() < 20, (
            "l1_ratio=1 must keep L1 sparsity (fewer nonzero coefs than features)"
        )


class TestRnaseqCpmRegression:
    """Bug 3: voom must operate on CPM of the NORMALIZED counts, once."""

    def _counts(self):
        rng = np.random.RandomState(42)
        counts = pd.DataFrame(
            rng.negative_binomial(50, 0.3, size=(30, 6)).astype(float),
            index=[f"GENE_{i:03d}" for i in range(30)],
            columns=[f"S{i}" for i in range(6)],
        )
        counts.iloc[0:3] *= 5
        return counts

    def test_median_of_ratios_output_is_log2_cpm(self, monkeypatch):
        monkeypatch.setattr(config, "DATA_TYPE", "rnaseq", raising=False)
        monkeypatch.setattr(config, "RNASEQ_NORMALIZATION", "median_of_ratios",
                            raising=False)
        counts = self._counts()

        result = normalize_rnaseq_counts(counts.copy())

        size_factors = _median_of_ratios_normalize(counts.copy())
        expected_cpm = size_factors.div(size_factors.sum(axis=0), axis=1) * 1e6
        expected = np.log2(expected_cpm + 0.5)
        pd.testing.assert_frame_equal(result, expected)

    def test_tmm_output_not_double_scaled(self, monkeypatch):
        """TMM branch must divide by the NORMALIZED library sizes, not raw ones.

        If the raw library sizes were used, samples whose TMM factor deviates
        from 1 would be scaled a second time and the recovered per-sample totals
        (in CPM space) would drift from 1e6 by that factor.
        """
        monkeypatch.setattr(config, "DATA_TYPE", "rnaseq", raising=False)
        monkeypatch.setattr(config, "RNASEQ_NORMALIZATION", "tmm", raising=False)
        counts = self._counts()
        # Give sample S0 a strong compositional shift so its TMM factor != 1
        counts["S0"] *= 3

        result = normalize_rnaseq_counts(counts.copy())

        # Recover CPM-space column sums after reversing the log2(+0.5) shift.
        # They must all be ~1e6 (not 1e6 * tmm_factor).
        recovered = (2 ** result - 0.5).sum(axis=0)
        np.testing.assert_allclose(recovered, 1e6, rtol=0.02)


class TestCombatRegression:
    """Bug 4: ComBat must tolerate a scalar cohort attr."""

    def test_combat_scalar_cohort_attr(self):
        rng = np.random.RandomState(42)
        expr = pd.DataFrame(
            rng.randn(20, 8),
            index=[f"G{i}" for i in range(20)],
            columns=[f"S{i}" for i in range(8)],
        )
        labels = pd.Series(["Tumor", "Normal"] * 4, index=expr.columns)
        labels.attrs["cohort"] = "GSE8671"  # scalar, as load_multi_cohort sets

        corrected = _combat_correct(expr, labels)
        assert corrected.shape == expr.shape
        assert np.isfinite(corrected.values).all()


class TestMetaGeneLookupRegression:
    """Gene lookup must work for both indexed and 'gene'-column DE tables."""

    def test_lookup_in_gene_column(self):
        de = pd.DataFrame({
            "gene": ["EGFR", "KRAS"],
            "log2FC": [2.0, -1.0],
            "pvalue": [1e-5, 1e-3],
        })  # default RangeIndex, gene in a column
        row = _find_gene_in_df(de, "KRAS")
        assert row is not None
        assert float(row["log2FC"]) == -1.0

    def test_lookup_in_index(self):
        de = pd.DataFrame(
            {"log2FC": [2.0, -1.0], "pvalue": [1e-5, 1e-3]},
            index=["EGFR", "KRAS"],
        )
        row = _find_gene_in_df(de, "KRAS")
        assert row is not None
        assert float(row["log2FC"]) == -1.0

    def test_lookup_gene_column_takes_precedence_when_index_generic(self):
        """Index is a RangeIndex; 'gene' column carries the symbols."""
        de = pd.DataFrame({
            "gene": ["EGFR", "TP53"],
            "log2FC": [2.0, -3.0],
            "pvalue": [1e-5, 1e-9],
        })
        row = _find_gene_in_df(de, "TP53")
        assert row is not None
        assert float(row["log2FC"]) == -3.0


class TestExternalValidationAlignment:
    """Bug 5: labels must come from AFTER preprocessing."""

    def test_preprocess_returns_aligned_labels(self, synthetic_expression_data):
        """Pin the contract the fixed external validation relies on: the second
        return value of preprocess() must align with the returned matrix."""
        from src.preprocessing import preprocess

        expr_df, labels = synthetic_expression_data
        # Add a sample whose label will not survive alignment (unlabeled column)
        expr_df = expr_df.copy()
        expr_df["phantom_sample"] = expr_df.iloc[:, 0]
        labels_ext = pd.concat(
            [labels, pd.Series(["Tumor"], index=["phantom_sample"])]
        )
        labels_ext.name = labels.name

        # preprocess() intersects columns with labels; the returned pair must
        # always be row/column aligned.
        expr_clean, labels_clean = preprocess(expr_df, labels_ext)
        assert list(expr_clean.columns) == list(labels_clean.index)


class TestSyntheticTestDirNesting:
    """Bug 6: no synthetic_test/synthetic_test/ nesting in test mode."""

    def test_external_validation_output_dir_guard(self, monkeypatch, tmp_path):
        """When RESULTS_DIR is already .../synthetic_test, force_synthetic must
        NOT append another synthetic_test segment."""
        from src.external_validation import _unavailable_result

        results_root = tmp_path / "results"
        test_dir = results_root / "synthetic_test"
        test_dir.mkdir(parents=True)
        monkeypatch.setattr(config, "RESULTS_DIR", str(test_dir), raising=False)

        _unavailable_result("regression probe", force_synthetic=True)

        # Metrics landed in the synthetic_test dir itself, not a nested copy
        assert (test_dir / "external_validation_metrics.csv").exists()
        assert not (test_dir / "synthetic_test").exists()


class TestMultiCohortCohortMap:
    """load_multi_cohort must expose a per-sample cohort map that survives
    pd.concat (attrs are dropped by concat)."""

    def test_cohort_map_survives_concat(self, monkeypatch):
        from src.data_loader import load_multi_cohort

        cohorts = ["GSE0001", "GSE0002"]
        monkeypatch.setattr(config, "DISCOVERY_COHORTS", cohorts, raising=False)

        def fake_load_data(force_synthetic=False):
            # Config.GEO_ACCESSION is swapped by load_multi_cohort per cohort
            acc = config.GEO_ACCESSION
            rng = np.random.RandomState(hash(acc) % (2 ** 32))
            expr = pd.DataFrame(
                rng.randn(50, 6),
                index=[f"GENE_{i:03d}" for i in range(50)],
                columns=[f"{acc}_S{i}" for i in range(6)],
            )
            labels = pd.Series(["Tumor"] * 3 + ["Normal"] * 3,
                               index=expr.columns, name="condition")
            return expr, labels

        monkeypatch.setattr("src.data_loader.load_data", fake_load_data)

        combined_expr, combined_labels = load_multi_cohort()

        cohort_map = combined_labels.attrs.get("cohort")
        assert cohort_map is not None, (
            "combined labels must carry a per-sample cohort map"
        )
        assert isinstance(cohort_map, pd.Series)
        assert cohort_map.index.equals(combined_labels.index)
        assert set(cohort_map.unique()) == set(cohorts)

    def test_meta_analysis_end_to_end_with_mixed_tables(self):
        """Meta-analysis accepts per-cohort DE tables with gene-named index
        AND a 'gene' column (the shape run_differential_expression emits)."""
        genes = ["EGFR", "KRAS", "TP53"]
        de1 = pd.DataFrame({
            "gene": genes,
            "log2FC": [2.1, 1.5, -1.2],
            "pvalue": [1e-5, 1e-4, 1e-3],
            "adj_pvalue": [1e-4, 1e-3, 1e-2],
        }, index=genes)
        de2 = pd.DataFrame({
            "gene": genes,
            "log2FC": [1.8, 1.2, -0.9],
            "pvalue": [1e-4, 1e-3, 1e-2],
            "adj_pvalue": [1e-3, 1e-2, 1e-1],
        }, index=genes)

        results = run_meta_analysis([de1, de2], ["C1", "C2"])
        assert set(results["gene"]) == set(genes)
        assert results["direction_consistent"].all()


def _fake_de(expr_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Minimal DE table shaped like run_differential_expression output."""
    tumor = labels[labels == "Tumor"].index
    normal = labels[labels == "Normal"].index
    log2fc = expr_df[tumor].mean(axis=1) - expr_df[normal].mean(axis=1)
    return pd.DataFrame({
        "gene": expr_df.index,
        "log2FC": log2fc.values,
        "pvalue": 1e-6,
        "adj_pvalue": 1e-5,
        "regulation": np.where(log2fc > 0, "Upregulated", "Downregulated"),
    }, index=expr_df.index)
