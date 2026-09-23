"""Offline regressions for signature scoring and truthful saved-artifact reports.

Run only this module: python -m pytest tests/test_signature_report_regressions.py
"""
import json
import socket
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.signature_scorer import calculate_composite_risk_score
from src.report_generator import generate_html_report


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("These regression tests must not use the network")
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)


@pytest.fixture
def signature():
    expr = pd.DataFrame([[0., 1., 2., 3.], [0., 3., 2., 5.]],
                        index=["A", "B"], columns=["n1", "n2", "t1", "t2"])
    labels = pd.Series(["Normal", "Normal", "Tumor", "Tumor"], index=expr.columns)
    consensus = pd.DataFrame({"gene": ["A", "B"], "composite_score": [.9, .1],
                              "ensemble_score": [.2, .8], "log2FC": [2., -1.]})
    return expr, labels, consensus


def score(signature, tmp_path):
    return calculate_composite_risk_score(*signature, out_dir=str(tmp_path))


def test_actual_composite_weights_signed_and_not_equal(signature, tmp_path):
    expr, _, _ = signature
    result = score(signature, tmp_path)
    expected = np.array([.9, -.1]) @ ((expr.values - expr.values.mean(axis=1)[:, None])
                                             / expr.values.std(axis=1, ddof=1)[:, None])
    np.testing.assert_allclose(result["scores_df"]["risk_score"], expected)
    assert result["metrics"]["signature_weights"] == pytest.approx({"A": .9, "B": -.1})
    assert result["metrics"]["provenance"]["weight_source"] == "composite_score"
    assert not np.allclose(expected, np.array([.5, -.5]) @ ((expr - expr.mean(axis=1).values[:, None])
                                                         / expr.std(axis=1).values[:, None]))


def test_exact_two_gene_review_reproducer(tmp_path):
    expr = pd.DataFrame([[0., 1., 3., 4.], [0., 2., 1., 3.]],
                        index=["A", "B"], columns=["n1", "n2", "t1", "t2"])
    labels = pd.Series(["Normal", "Normal", "Tumor", "Tumor"], index=expr.columns)
    consensus = pd.DataFrame({"gene": ["A", "B"], "composite_score": [.9, .1], "log2FC": [2., 1.]})
    result = score((expr, labels, consensus), tmp_path)
    assert result["metrics"]["signature_weights"] == pytest.approx({"A": .9, "B": .1})
    np.testing.assert_array_equal(result["scores_df"]["risk_score"].round(3), [-1.102, -.454, .454, 1.102])


def test_ensemble_only_schema_is_explicit(signature, tmp_path):
    expr, labels, consensus = signature
    result = score((expr, labels, consensus.drop(columns="composite_score")), tmp_path)
    assert result["metrics"]["signature_weights"] == pytest.approx({"A": .2, "B": -.8})
    assert result["metrics"]["provenance"]["weight_source"] == "ensemble_score"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1., "not-a-weight"])
def test_invalid_primary_weight_does_not_fall_back_to_ensemble(signature, tmp_path, bad):
    expr, labels, consensus = signature
    consensus["composite_score"] = [bad, .1]
    with pytest.raises(ValueError, match="weights must be finite"):
        score((expr, labels, consensus), tmp_path)
    assert not (tmp_path / "diagnostic_performance.json").exists()


def test_missing_weight_column_rejects_obsolete_consensus_score(signature, tmp_path):
    expr, labels, consensus = signature
    consensus = consensus.drop(columns=["ensemble_score", "composite_score"])
    consensus["consensus_score"] = [.9, .1]
    with pytest.raises(ValueError, match="composite_score or ensemble_score"):
        score((expr, labels, consensus), tmp_path)


@pytest.mark.parametrize("kind", ["missing_label", "unknown_label", "one_class", "duplicate_label",
                                  "duplicate_gene", "duplicate_sample", "empty_expression", "empty_consensus",
                                  "missing_gene_column", "missing_direction", "nan_direction", "zero_weights",
                                  "zero_directions", "nan_expression", "inf_expression", "no_overlap"])
def test_invalid_signature_inputs_fail_closed(signature, tmp_path, kind):
    expr, labels, consensus = signature
    if kind == "missing_label":
        labels = labels.iloc[:-1]
    elif kind == "unknown_label":
        labels.iloc[0] = "Unknown"
    elif kind == "one_class":
        labels[:] = "Tumor"
    elif kind == "duplicate_label":
        labels.index = ["n1", "n1", "t1", "t2"]
    elif kind == "duplicate_gene":
        consensus["gene"] = ["A", "A"]
    elif kind == "duplicate_sample":
        expr.columns = ["n1", "n1", "t1", "t2"]
    elif kind == "empty_expression":
        expr = expr.iloc[:, :0]
    elif kind == "empty_consensus":
        consensus = consensus.iloc[:0]
    elif kind == "missing_gene_column":
        consensus = consensus.drop(columns="gene")
    elif kind == "missing_direction":
        consensus = consensus.drop(columns="log2FC")
    elif kind == "nan_direction":
        consensus.loc[0, "log2FC"] = np.nan
    elif kind == "zero_weights":
        consensus["composite_score"] = 0.
    elif kind == "zero_directions":
        consensus["log2FC"] = 0.
    elif kind in ("nan_expression", "inf_expression"):
        expr.iloc[0, 0] = np.nan if kind == "nan_expression" else np.inf
    elif kind == "no_overlap":
        expr.index = ["C", "D"]
    with pytest.raises(ValueError):
        score((expr, labels, consensus), tmp_path)


@pytest.mark.parametrize("top_n", [0, -1, 1.5, True])
def test_invalid_signature_size(signature, tmp_path, top_n):
    with pytest.raises(ValueError, match="positive integer"):
        calculate_composite_risk_score(*signature, top_n=top_n, out_dir=str(tmp_path))


def test_labels_are_aligned_by_identifier(signature, tmp_path):
    expr, labels, consensus = signature
    forward = score(signature, tmp_path)
    reverse = score((expr, labels.iloc[::-1], consensus), tmp_path)
    pd.testing.assert_frame_equal(forward["scores_df"], reverse["scores_df"])
    assert forward["metrics"] == reverse["metrics"]


def test_missing_genes_disclosed_and_available_weights_renormalized(signature, tmp_path):
    expr, labels, consensus = signature
    result = score((expr.loc[["B"]], labels, consensus), tmp_path)
    assert result["metrics"]["signature_genes"] == ["B"]
    assert result["metrics"]["missing_signature_genes"] == ["A"]
    assert result["metrics"]["signature_weights"] == {"B": -1.}


def test_no_bootstrap_ci_and_full_apparent_training_provenance(signature, tmp_path):
    # Scoring must not reseed or consume the caller's global RNG.
    np.random.seed(782)
    state = np.random.get_state()
    result = score(signature, tmp_path)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    metrics = result["metrics"]
    assert metrics["validation_status"] == "not_independently_validated"
    assert metrics["evaluation_type"] == metrics["evaluation_scope"] == "apparent_training"
    assert metrics["clinical_use_supported"] is False
    assert not any("95ci" in key for key in metrics)
    assert metrics["confidence_intervals_status"].startswith("not_reported")
    provenance = metrics["provenance"]
    for key in ("selection_data", "standardization_fit_data", "threshold_fit_data", "evaluation_data"):
        assert provenance[key] == "same_discovery_cohort"
    assert provenance["sample_ids"] == signature[0].columns.tolist()
    assert provenance["independent_validation_performed"] is False
    disk = json.loads((tmp_path / "diagnostic_performance.json").read_text())
    assert disk == metrics
    assert set(result["scores_df"]["evaluation_scope"]) == {"apparent_training"}
    reconstructed = []
    for gene in metrics["signature_genes"]:
        reconstructed.append(metrics["signature_weights"][gene] *
                             (signature[0].loc[gene] - metrics["standardization"]["means"][gene]) /
                             metrics["standardization"]["scales"][gene])
    np.testing.assert_allclose(np.sum(reconstructed, axis=0), result["scores_df"]["risk_score"])
    saved_scores = pd.read_csv(tmp_path / "patient_risk_scores.csv")
    predicted = np.where(saved_scores["risk_score"] >= disk["optimal_decision_threshold"], "Tumor", "Normal")
    assert predicted.tolist() == saved_scores["predicted_class"].tolist()


def test_constant_signature_has_no_fabricated_npv_or_infinite_cutoff(signature, tmp_path):
    expr, labels, consensus = signature
    expr.iloc[:, :] = 3.
    metrics = score((expr, labels, consensus), tmp_path)["metrics"]
    assert metrics["roc_auc"] == .5
    assert metrics["npv"] is None  # No negative predictions: denominator is zero.
    assert metrics["specificity"] == 0.
    assert np.isfinite(metrics["optimal_decision_threshold"])
    assert "NaN" not in (tmp_path / "diagnostic_performance.json").read_text()


class ReportParser(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.cells, self.cards, self.text = [], [], []
        self.capture = None
        self.parts = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == "td" or (tag == "div" and dict(attrs).get("class") == "meta-val"):
            self.capture, self.parts = tag, []

    def handle_data(self, data):
        self.text.append(data)
        if self.capture:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if self.capture == tag:
            (self.cells if tag == "td" else self.cards).append("".join(self.parts).strip())
            self.capture = None


def render(tmp_path):
    html = Path(generate_html_report(str(tmp_path))).read_text(encoding="utf-8")
    return html, ReportParser(html)


def test_missing_artifacts_show_unavailable_not_success_or_zero(tmp_path):
    html, parsed = render(tmp_path)
    assert parsed.cards == ["Unavailable"] * 8
    assert "Unavailable: no consensus biomarker rows" in html
    assert "Unavailable: no pathway enrichment rows" in html
    assert "PIPELINE VERIFIED" not in html
    assert "Zero Data Leakage" not in html
    assert "95.0%" not in html and "0.980" not in html and "100.0%" not in html
    assert "Recorded validation status: <strong>Unavailable</strong>" in html


@pytest.mark.parametrize("content", ["", "\n", "irrelevant\n", "irrelevant\nNaN\n"])
def test_empty_and_wrong_schema_csvs_do_not_break_report(tmp_path, content):
    for name in ("de_results", "consensus_biomarkers", "ml_ranking_top100", "nested_cv_metrics", "pathway_enrichment"):
        (tmp_path / f"{name}.csv").write_text(content)
    _, parsed = render(tmp_path)
    assert parsed.cards == ["Unavailable"] * 8
    assert not any(cell == "0.000" for cell in parsed.cells)


@pytest.mark.parametrize("content", ["", "{broken", "null", "[]", '{"metrics": null}'])
def test_missing_empty_and_invalid_json_metadata(tmp_path, content):
    (tmp_path / "qc_summary.json").write_text(content)
    (tmp_path / "diagnostic_performance.json").write_text(content)
    _, parsed = render(tmp_path)
    assert parsed.cards == ["Unavailable"] * 8


def test_normal_producer_schema_and_gene_join(tmp_path):
    pd.DataFrame({"regulation": ["Upregulated", "Downregulated", "Not Significant"]}).to_csv(tmp_path / "de_results.csv", index=False)
    pd.DataFrame({"gene": ["A", "B"], "log2FC": [2., -1.5], "adj_pvalue": [.001, .00002],
                  "composite_score": [.9, .3], "ensemble_score": [.8, .2],
                  "consensus_tier": ["FDR-controlled", "Exploratory (nominal only)"]}).to_csv(tmp_path / "consensus_biomarkers.csv", index=False)
    pd.DataFrame({"gene": ["B", "A"], "RF_importance": [.0123, .4567],
                  "GB_importance": [.0234, .5678]}).to_csv(tmp_path / "ml_ranking_top100.csv", index=False)
    pd.DataFrame({"outer_fold": [1, 2], "fold_accuracy": [.7, .9],
                  "fold_roc_auc": [.8, .9]}).to_csv(tmp_path / "nested_cv_metrics.csv", index=False)
    pd.DataFrame({"pathway": ["Cell cycle"], "database": ["GO"], "overlap_count": [2],
                  "pathway_size": [10], "fold_enrichment": [3.2], "adjusted_p_value": [.0003],
                  "genes": ["A;B"]}).to_csv(tmp_path / "pathway_enrichment.csv", index=False)
    (tmp_path / "diagnostic_performance.json").write_text(json.dumps({"sensitivity": .75,
        "specificity": .6, "ppv": .5, "optimal_decision_threshold": -.125,
        "evaluation_scope": "apparent_training", "validation_status": "not_independently_validated",
        "provenance": {"weight_source": "composite_score", "evaluation_data": "same_discovery_cohort"}}))
    (tmp_path / "qc_summary.json").write_text(json.dumps({"n_samples": 20, "n_genes": 300, "n_outliers": 0}))
    html, parsed = render(tmp_path)
    assert parsed.cards == ["2 (1 Up / 1 Down)", "2 Genes", "80.0%", "0.850", "75.0%", "60.0%", "50.0%", "-0.125"]
    assert parsed.cells[:7] == ["A", "2.00", "1.00e-03", "0.4567", "0.5678", "0.900", "FDR-controlled"]
    assert parsed.cells[7:14] == ["B", "-1.50", "2.00e-05", "0.0123", "0.0234", "0.300", "Exploratory (nominal only)"]
    assert parsed.cells[14:] == ["Cell cycle", "GO", "2 / 10", "3.20x", "3.00e-04", "A;B"]
    assert "not_independently_validated" in html and "apparent_training" in html
    assert "Apparent Training Performance" in html
    assert "not held-out performance" in html
    assert "Weight source: composite_score" in html
    assert "Samples: 20" in html and "flagged outliers: 0" in html


def test_nan_missing_values_and_out_of_range_metrics_stay_unavailable(tmp_path):
    pd.DataFrame({"gene": ["A"], "log2FC": [np.nan], "adj_pvalue": [np.nan],
                  "RF_importance": [np.nan], "GB_importance": [np.nan],
                  "composite_score": [np.nan]}).to_csv(tmp_path / "consensus_biomarkers.csv", index=False)
    pd.DataFrame({"fold_accuracy": [np.nan, np.inf, -1, 2],
                  "fold_roc_auc": [np.nan, np.inf, -1, 2]}).to_csv(tmp_path / "nested_cv_metrics.csv", index=False)
    (tmp_path / "diagnostic_performance.json").write_text(json.dumps({"sensitivity": float("nan"),
        "specificity": None, "ppv": "invalid", "optimal_decision_threshold": float("inf")}))
    html, parsed = render(tmp_path)
    assert parsed.cards[2:] == ["Unavailable"] * 6
    assert parsed.cells[1:7] == ["Unavailable"] * 6
    assert ">nan<" not in html.lower() and ">inf<" not in html.lower()


def test_real_zero_values_not_marked_missing(tmp_path):
    pd.DataFrame({"regulation": ["Not Significant"]}).to_csv(tmp_path / "de_results.csv", index=False)
    pd.DataFrame({"fold_accuracy": [0], "fold_roc_auc": [0]}).to_csv(tmp_path / "nested_cv_metrics.csv", index=False)
    (tmp_path / "diagnostic_performance.json").write_text(json.dumps({"metrics": {
        "sensitivity": 0, "specificity": 0, "ppv": 0, "optimal_decision_threshold": 0}}))
    _, parsed = render(tmp_path)
    assert parsed.cards == ["0 (0 Up / 0 Down)", "Unavailable", "0.0%", "0.000", "0.0%", "0.0%", "0.0%", "0.000"]


def test_partial_fold_availability_is_disclosed(tmp_path):
    pd.DataFrame({"fold_accuracy": [.6, np.nan, .8], "fold_roc_auc": [np.nan, .9, np.nan]}).to_csv(tmp_path / "nested_cv_metrics.csv", index=False)
    html, parsed = render(tmp_path)
    assert parsed.cards[2:4] == ["70.0%", "0.900"]
    assert "valid accuracy folds: 2/3; valid AUC folds: 1/3" in html


def test_duplicate_ml_genes_not_arbitrarily_joined(tmp_path):
    pd.DataFrame({"gene": ["A", "B"]}).to_csv(tmp_path / "consensus_biomarkers.csv", index=False)
    pd.DataFrame({"gene": ["A", "A"], "RF_importance": [.3, .4], "GB_importance": [.2, .5]}).to_csv(tmp_path / "ml_ranking_top100.csv", index=False)
    _, parsed = render(tmp_path)
    assert parsed.cells[3:5] == ["Unavailable", "Unavailable"]
    assert parsed.cells[10:12] == ["Unavailable", "Unavailable"]


def test_untrusted_artifact_text_is_html_escaped(tmp_path, monkeypatch):
    import src.report_generator as reports
    monkeypatch.setattr(reports.config, "CANCER_TYPE", "<script>alert(1)</script>")
    pd.DataFrame({"gene": ["<img src=x onerror=alert(1)>"]}).to_csv(tmp_path / "consensus_biomarkers.csv", index=False)
    (tmp_path / "diagnostic_performance.json").write_text(json.dumps({"validation_status": "<script>bad</script>"}))
    html, _ = render(tmp_path)
    assert "<script>" not in html and "<img " not in html
    assert "&lt;script&gt;" in html and "&lt;img " in html


def test_score_report_integration_retains_apparent_status(signature, tmp_path):
    score(signature, tmp_path)
    html, _ = render(tmp_path)
    assert "not_independently_validated" in html and "apparent_training" in html
    assert "same_discovery_cohort" in html
    assert "No clinical confidence intervals are reported" in html
    assert "PIPELINE VERIFIED" not in html and "Zero Data Leakage" not in html
