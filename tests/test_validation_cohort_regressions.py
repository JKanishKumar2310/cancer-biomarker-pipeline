"""Offline regressions for discovery/external cohort independence."""
import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import config
from src import ai_annotator as annotator
from src import external_validation as external


class ValidationCohortRegressions(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / "data"
        self.results = self.root / "results"
        self.data.mkdir()
        self.results.mkdir()
        for name, value in {
            "GEO_ACCESSION": " gse30784 ",
            "VALIDATION_COHORT": None,
            "DATA_DIR": str(self.data),
            "RESULTS_DIR": str(self.results),
            "KNOWN_MARKERS": ["TP53", "EGFR"],
        }.items():
            self.enterContext(patch.object(config, name, value, create=True))
        # Block transport even if an implementation accidentally reaches it.
        self.urlopen = self.enterContext(patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")))
        self.retrieve = self.enterContext(patch("urllib.request.urlretrieve", side_effect=AssertionError("network forbidden")))
        self.metadata = self.enterContext(patch("src.ai_geo_curator.fetch_geo_metadata", side_effect=AssertionError("metadata fetch forbidden")))
        self.addCleanup(self.assert_no_network)

    def assert_no_network(self):
        self.urlopen.assert_not_called()
        self.retrieve.assert_not_called()
        self.metadata.assert_not_called()

    def assert_unavailable(self, cohort):
        self.assertEqual(cohort["status"], "UNAVAILABLE")
        self.assertIsNone(cohort["accession"])
        self.assertIn("unavailable", cohort["label"].lower())

    def test_oral_aliases_without_api_fail_closed(self):
        with patch.object(annotator, "get_api_key", return_value=None):
            for cancer in ["Oral Squamous Cell Carcinoma (OSCC)", "head and neck", "OSCC"]:
                with self.subTest(cancer=cancer):
                    cohorts = annotator.fetch_cohorts_for_cancer(cancer)
                    self.assert_unavailable(cohorts["validation"])
                    self.assertIn("GSE30784", cohorts["validation"]["reason"])
                    self.assertTrue(cohorts["survival"]["accession"])
        self.assertEqual(annotator._VALIDATION_FALLBACKS["oral"]["accession"], "GSE30784")

    def test_llm_collision_rejected_before_platform_fetch(self):
        for accession in ["GSE30784", " gse30784\n"]:
            with self.subTest(accession=accession), patch.object(
                annotator, "_llm_query_cohort", return_value={"accession": accession, "label": "Wrong"}
            ) as query, patch.object(annotator, "_resolve_platform") as platform:
                self.assert_unavailable(annotator.fetch_validation_cohort_from_llm("oral", "unused"))
                platform.assert_not_called()
                self.assertIn(config.GEO_ACCESSION, query.call_args.args[0])

    def test_api_failure_fallback_rejected(self):
        with patch.object(annotator, "_llm_query_cohort", return_value=None):
            self.assert_unavailable(annotator.fetch_validation_cohort_from_llm("OSCC", "unused"))

    def test_default_fallback_cannot_be_discovery(self):
        config.GEO_ACCESSION = "gse18842"
        with patch.object(annotator, "get_api_key", return_value=None):
            self.assert_unavailable(annotator.fetch_cohorts_for_cancer("unknown")["validation"])

    def test_independent_llm_selection_is_normalized(self):
        candidate = {"accession": " gse18842 ", "label": "Existing selection", "platform": "GPL570"}
        with patch.object(annotator, "_llm_query_cohort", return_value=candidate), patch.object(annotator, "_resolve_platform") as platform:
            result = annotator.fetch_validation_cohort_from_llm("lung", "unused")
        self.assertEqual(result["accession"], "GSE18842")
        platform.assert_called_once_with(result)
        self.assertEqual(candidate["accession"], " gse18842 ")

    def test_invalid_or_unavailable_selection_never_acquires_accession(self):
        for candidate in [None, {}, {"accession": ""}, {"accession": "../GSE30784"},
                          {"accession": "GSE18842", "status": "UNAVAILABLE"}]:
            with self.subTest(candidate=candidate):
                self.assert_unavailable(annotator.validate_validation_cohort(candidate))

    def test_loader_rejects_cached_discovery_before_read(self):
        config.VALIDATION_COHORT = {"accession": " GSE30784 ", "label": "Cached selection"}
        (self.data / "GSE30784_expression.csv").write_text("cached discovery")
        (self.data / "GSE30784_labels.csv").write_text("cached discovery")
        with patch.object(external.pd, "read_csv") as read:
            with self.assertRaises(external.ExternalValidationUnavailable):
                external.load_external_cohort()
            read.assert_not_called()

    def test_unknown_discovery_does_not_default_to_breast(self):
        with self.assertRaises(external.ExternalValidationUnavailable):
            external.load_external_cohort()

    def test_explicit_unavailable_does_not_fall_back_to_mapping(self):
        config.GEO_ACCESSION = "GSE19804"
        config.VALIDATION_COHORT = {"accession": None, "status": "UNAVAILABLE", "reason": "Rejected"}
        with self.assertRaisesRegex(external.ExternalValidationUnavailable, "Rejected"):
            external.load_external_cohort()

    def test_builtin_mapping_and_cache_still_load(self):
        config.GEO_ACCESSION = " gse19804 "
        expr = pd.DataFrame([[1., 2.]], index=["TP53"], columns=["GSM1", "GSM2"])
        labels = pd.Series(["Tumor", "Normal"], index=expr.columns, name="condition")
        expr.to_csv(self.data / "GSE18842_expression.csv")
        labels.to_frame().to_csv(self.data / "GSE18842_labels.csv")
        loaded_expr, loaded_labels = external.load_external_cohort()
        pd.testing.assert_frame_equal(loaded_expr, expr)
        pd.testing.assert_series_equal(loaded_labels, labels)

    def test_pipeline_returns_unavailable_and_overwrites_stale_success(self):
        config.VALIDATION_COHORT = {"accession": "GSE30784"}
        (self.results / "external_validation_metrics.csv").write_text("test_accuracy,roc_auc\n1,1\n")
        (self.results / "external_roc_curve.csv").write_text("fpr,tpr\n0,1\n")
        with patch("src.data_loader.load_data") as discovery, patch.object(external, "load_external_cohort") as loader:
            result = external.run_external_validation()
        discovery.assert_not_called()
        loader.assert_not_called()
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertTrue(np.isnan(result["accuracy"]))  # run_analysis's unavailable contract
        metrics = pd.read_csv(self.results / "external_validation_metrics.csv")
        self.assertEqual(metrics.iloc[0]["status"], "UNAVAILABLE")
        self.assertTrue(np.isnan(metrics.iloc[0]["test_accuracy"]))
        self.assertTrue(pd.read_csv(self.results / "external_roc_curve.csv").empty)

    def test_sample_overlap_rejected_before_model_fit(self):
        config.VALIDATION_COHORT = {"accession": "GSE18842"}
        train = pd.DataFrame([[1., 2.], [3., 4.]], index=["TP53", "EGFR"], columns=["GSM1", "GSM2"])
        labels = pd.Series(["Tumor", "Normal"], index=train.columns)
        test = train.rename(columns={"GSM1": " gsm1 ", "GSM2": "GSM3"})
        test_labels = pd.Series(["Tumor", "Normal"], index=test.columns)
        with patch("src.data_loader.load_data", return_value=(train, labels)), patch(
            "src.preprocessing.preprocess", return_value=(train, None)
        ), patch.object(external, "load_external_cohort", return_value=(test, test_labels)), patch.object(external, "RandomForestClassifier") as rf:
            result = external.run_external_validation()
        rf.assert_not_called()
        self.assertEqual(result["status"], "SAMPLE_OVERLAP")
        self.assertTrue(np.isnan(result["accuracy"]))

    def test_independent_cached_cohort_can_be_evaluated(self):
        config.VALIDATION_COHORT = {"accession": "GSE18842"}
        train = pd.DataFrame([[1., 2.], [3., 4.]], index=["TP53", "EGFR"], columns=["GSM1", "GSM2"])
        labels = pd.Series(["Tumor", "Normal"], index=train.columns)
        test = train.rename(columns={"GSM1": "GSM3", "GSM2": "GSM4"})
        test_labels = pd.Series(["Tumor", "Normal"], index=test.columns, name="condition")
        test.to_csv(self.data / "GSE18842_expression.csv")
        test_labels.to_frame().to_csv(self.data / "GSE18842_labels.csv")
        with patch("src.data_loader.load_data", return_value=(train, labels)), patch(
            "src.preprocessing.preprocess", return_value=(train, None)
        ), patch.object(external, "RandomForestClassifier") as rf:
            rf.return_value.predict.return_value = np.array([1, 0])
            rf.return_value.predict_proba.return_value = np.array([[0.1, 0.9], [0.9, 0.1]])
            result = external.run_external_validation()
        rf.return_value.fit.assert_called_once()
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["roc_auc"], 1.0)

    def test_overlap_ids_include_both_expression_and_labels(self):
        expr = pd.DataFrame(columns=[" GSM1 ", "gsm2"])
        labels = pd.Series(["Tumor"], index=["gsm3"])
        self.assertEqual(external._sample_ids(expr, labels), {"GSM1", "GSM2", "GSM3"})

    def test_parse_failure_cannot_turn_into_authentic_synthetic_result(self):
        config.VALIDATION_COHORT = {"accession": "GSE18842", "platform": "GPL570"}
        with gzip.open(self.data / "GSE18842_series_matrix.txt.gz", "wt") as handle:
            handle.write("")
        with patch.object(external, "_create_synthetic_external_cohort") as synthetic:
            with self.assertRaises(external.ExternalValidationUnavailable):
                external.load_external_cohort()
        synthetic.assert_not_called()

    def test_explicit_synthetic_mode_remains_offline(self):
        config.VALIDATION_COHORT = {"accession": "GSE30784"}
        expr, labels = external.load_external_cohort(force_synthetic=True)
        self.assertEqual(expr.shape[1], len(labels))
        self.assertEqual(set(labels), {"Tumor", "Normal"})


if __name__ == "__main__":
    unittest.main()
