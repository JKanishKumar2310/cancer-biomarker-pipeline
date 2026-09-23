"""Offline survival regressions: no real cohort fetching or LLM calls."""
import gzip
import socket
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import survival_analysis as survival


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden in survival regression tests")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(survival.config, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(survival.config, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(survival.config, "GEO_ACCESSION", "GSE99999")
    monkeypatch.setattr(survival.config, "CANCER_TYPE", "Unspecified")
    monkeypatch.setattr(survival.config, "KNOWN_MARKERS", ["GENE", "MYC"])
    monkeypatch.setattr(survival.config, "SURVIVAL_COHORT", None, raising=False)
    (tmp_path / "data").mkdir()
    # Avoid importing the curator (and its LLM clients) even for a mocked mapping.
    fake = types.ModuleType("src.ai_geo_curator")
    fake.fetch_gpl_probe_mapping = forbidden
    fake.fetch_geo_metadata = forbidden
    monkeypatch.setitem(sys.modules, "src.ai_geo_curator", fake)


@pytest.fixture
def cohort():
    rng = np.random.default_rng(74)
    ids = [f"S{i}" for i in range(60)]
    expression = pd.DataFrame(rng.normal(size=(2, 60)), index=["GENE", "MYC"], columns=ids)
    clinical = pd.DataFrame({"SURV_DEATH": rng.uniform(.1, 12, 60), "DEATH": rng.integers(0, 2, 60),
                             "SURV_RELAPSE": rng.uniform(.1, 10, 60), "RELAPSE": rng.integers(0, 2, 60)}, index=ids)
    return expression, clinical


def test_km_removes_only_observations_at_current_time():
    timeline, probabilities = survival.compute_kaplan_meier([1, 2, 3], [1, 1, 1])
    assert timeline == [0, 1, 2, 3]
    np.testing.assert_allclose(probabilities, [1, 2/3, 1/3, 0])


def test_km_ties_censoring_and_invalid_records():
    timeline, probabilities = survival.compute_kaplan_meier([1, 1, 2, 3, np.nan, -1, np.inf, 5], [1, 0, 1, 0, 1, 1, 1, 2])
    assert timeline == [0, 1, 2, 3]
    np.testing.assert_allclose(probabilities, [1, .75, .375, .375])


def test_hr_is_binary_cox_estimate_and_reciprocal():
    th, tl = np.arange(1, 11), np.arange(2, 12)
    event = np.ones(10)
    _, p, hr = survival.log_rank_test(th, event, tl, event)
    reference = PHReg(np.r_[th, tl], np.r_[np.ones(10), np.zeros(10)][:, None], status=np.ones(20), ties="breslow").fit(disp=False)
    assert hr == pytest.approx(np.exp(reference.params[0]))
    assert hr > 1
    _, reverse_p, reverse_hr = survival.log_rank_test(tl, event, th, event)
    assert reverse_hr == pytest.approx(1 / hr)
    assert reverse_p == pytest.approx(p)


def test_logrank_known_statistic_and_stable_extreme_tail():
    stat, p, _ = survival.log_rank_test([1, 2], [1, 1], [3, 4], [1, 1])
    assert stat == pytest.approx(49 / 17)
    assert p == pytest.approx(.089555074413642)
    _, tiny_p, _ = survival.log_rank_test(np.arange(1, 101), np.ones(100), np.arange(101, 201), np.ones(100))
    assert 0 < tiny_p < 1e-20


@pytest.mark.parametrize("high_event,low_event", [(np.zeros(10), np.zeros(10)), (np.ones(10), np.zeros(10))])
def test_unestimable_hr_is_missing_not_one_or_clamped(high_event, low_event):
    _, _, hr = survival.log_rank_test(np.arange(1, 11), high_event, np.arange(11, 21), low_event)
    assert np.isnan(hr)


def test_exact_endpoint_parsing_and_year_conversion():
    metadata = {
        "A": {"characteristics": ["os.delay (months): 24", "os.event: 1", "rfs.delay: 12", "rfs.event: 0"]},
        "B": {"characteristics": ["overall survival days: 365.25", "vital_status: alive", "rfs_years: 3", "recurrence: yes"]},
    }
    os = survival.extract_clinical_survival_data(metadata, accession="GSE39582")
    rfs = survival.extract_clinical_survival_data(metadata, "relapse", "GSE39582")
    np.testing.assert_allclose(os["time"], [2, 1])
    np.testing.assert_allclose(os["event"], [1, 0])
    np.testing.assert_allclose(rfs["time"], [1, 3])
    np.testing.assert_allclose(rfs["event"], [0, 1])


@pytest.mark.parametrize("fields", [
    ["survival_months: 24"],
    ["survival_months: 24", "vital_status: unknown"],
    ["survival_months: 24", "vital_status: 10"],
    ["survival_months: 24", "vital_status: not dead"],
    ["survival_months: 24", "recurrence: yes"],
    ["pfs_months: 24", "vital_status: dead"],
    ["dfs_months: 24", "vital_status: dead"],
    ["bcr_free_months: 24", "vital_status: dead"],
    ["time: 24", "status: 1"],
    ["os: 24", "os.event: 1"],
    ["survival_months: 24", "er_status: positive"],
    ["survival_months: -2", "vital_status: dead"],
    ["survival_months: 24", "vital_status: alive", "death: yes"],
])
def test_no_missing_events_fabricated_or_substring_endpoint_matches(fields):
    meta = {"S": {"title": "Patient dead survival_months: 120", "characteristics": fields}}
    assert survival.extract_clinical_survival_data(meta).empty


def test_dashboard_contract_and_exclusion_of_invalid_expression(cohort):
    expression, clinical = cohort
    expression.loc["GENE", "S0"] = np.nan
    expression.loc["GENE", "S1"] = np.inf
    clinical.loc["S2", "DEATH"] = np.nan
    clinical.loc["S3", "DEATH"] = 2
    clinical.loc["S4", "SURV_DEATH"] = -1
    result = survival.evaluate_biomarker_survival("GENE", expr_df=expression, clinical_df=clinical)
    assert "error" not in result
    assert result["n_total"] == 55
    assert result["n_high"] + result["n_low"] == 55
    assert result["median_cutoff"] == pytest.approx(expression.loc["GENE", "S5":].median())
    for group in ("km_high", "km_low"):
        assert set(result[group]) == {"timeline", "survival"}
        assert len(result[group]["timeline"]) == len(result[group]["survival"])
        assert max(result[group]["timeline"]) <= 12


def test_duplicate_gene_rows_are_aggregated_and_samples_are_aligned(cohort):
    expression, clinical = cohort
    duplicate = pd.concat([expression, expression.loc[["GENE"]] + 2])
    result = survival.evaluate_biomarker_survival("GENE", expr_df=duplicate, clinical_df=clinical.iloc[::-1])
    expected = survival.evaluate_biomarker_survival("GENE", expr_df=expression + 1, clinical_df=clinical)
    assert result["median_cutoff"] == pytest.approx(expected["median_cutoff"])
    assert result["p_value"] == pytest.approx(expected["p_value"])


def test_degenerate_expression_and_unavailable_cohort_are_errors(cohort):
    expression, clinical = cohort
    expression.loc["GENE"] = 1
    assert "error" in survival.evaluate_biomarker_survival("GENE", expr_df=expression, clinical_df=clinical)
    assert "error" in survival.evaluate_biomarker_survival("GENE")
    assert "error" in survival.evaluate_biomarker_survival("GENE", "pfs", expression, clinical)


def test_five_year_survival_not_extrapolated(cohort):
    expression, clinical = cohort
    clinical["SURV_DEATH"] = 2
    clinical["DEATH"] = 0
    result = survival.evaluate_biomarker_survival("GENE", expr_df=expression, clinical_df=clinical)
    assert np.isnan(result["surv_5yr_high"])
    assert np.isnan(result["surv_5yr_low"])
    assert np.isnan(result["hazard_ratio"])


def test_selected_cohort_cache_independent_of_discovery(monkeypatch, cohort):
    expression, clinical = cohort
    monkeypatch.setattr(survival.config, "SURVIVAL_COHORT", {"accession": "GSE12345", "platform": "GPL570"})
    data = Path(survival.config.DATA_DIR)
    expression.to_csv(data / "GSE12345_expression.csv")
    clinical.to_csv(data / "GSE12345_clinical.csv")
    result_expression, result_clinical = survival.load_survival_cohort()
    pd.testing.assert_frame_equal(result_expression, expression)
    pd.testing.assert_frame_equal(result_clinical, clinical)
    assert survival.config.GEO_ACCESSION == "GSE99999"


def test_selected_local_matrix_parsing_preserves_patients_and_missing_events(monkeypatch):
    monkeypatch.setattr(survival.config, "SURVIVAL_COHORT", {"accession": "GSE39582", "platform": "GPL570"})
    seen = []
    def mapping(platform):
        seen.append(platform)
        return {"probe1": "GENE", "probe2": "GENE"}
    monkeypatch.setattr(sys.modules["src.ai_geo_curator"], "fetch_gpl_probe_mapping", mapping)
    matrix = ('\n!Series_platform_id\t"GPL570"\n\n'
              '!Sample_geo_accession\t"S1"\t"S2"\n'
              '!Sample_characteristics_ch1\t"os.delay (months): 24"\t"os.delay (months): 36"\n'
              '!Sample_characteristics_ch1\t"os.event: 1"\t"os.event: NA"\n'
              '!Sample_characteristics_ch1\t"rfs.delay: 12"\t"rfs.delay: 18"\n'
              '!Sample_characteristics_ch1\t"rfs.event: 0"\t"rfs.event: 1"\n'
              '!series_matrix_table_begin\n'
              '"ID_REF"\t"S1"\t"S2"\n"probe1"\t1\t2\n"probe2"\t3\t4\n'
              '!series_matrix_table_end\n')
    with gzip.open(Path(survival.config.DATA_DIR) / "GSE39582_series_matrix.txt.gz", "wt") as handle:
        handle.write(matrix)
    expression, clinical = survival.load_survival_cohort()
    assert seen == ["GPL570"]
    assert list(expression.columns) == ["S1", "S2"]
    np.testing.assert_allclose(expression.loc["GENE"], [2, 3])
    np.testing.assert_allclose(clinical["SURV_DEATH"], [2, 3])
    np.testing.assert_allclose(clinical["SURV_RELAPSE"], [1, 1.5])
    assert np.isnan(clinical.loc["S2", "DEATH"])


def test_pipeline_restores_os_rfs_consumer_schema_and_fdr(monkeypatch, cohort):
    monkeypatch.setattr(survival, "load_survival_cohort", lambda force_synthetic=False: cohort)
    monkeypatch.setattr(survival.config, "SURVIVAL_COHORT", {"accession": "GSE12345"})
    result = survival.run_survival_pipeline()
    assert set(survival._VALIDATION_COLUMNS) <= set(result)
    assert set(result["cohort_accession"]) == {"GSE12345"}
    assert set(result["data_provenance"]) == {"authentic_clinical_geo"}
    assert len(result) == 2
    for ep in ("os", "rfs"):
        np.testing.assert_allclose(result[f"{ep}_fdr"], multipletests(result[f"{ep}_pvalue"], method="fdr_bh")[1])
    np.testing.assert_allclose(result["p_value"], result["os_pvalue"])
    assert (Path(survival.config.RESULTS_DIR) / "survival_validation.csv").exists()


def test_rfs_only_cohort_is_not_dropped_or_mislabeled_as_os(monkeypatch, cohort):
    expression, clinical = cohort
    clinical = clinical.drop(columns=["SURV_DEATH", "DEATH"])
    monkeypatch.setattr(survival, "load_survival_cohort", lambda force_synthetic=False: (expression, clinical))
    result = survival.run_survival_pipeline()
    assert len(result) == 2
    assert result["os_pvalue"].isna().all()
    assert result["rfs_pvalue"].notna().all()


def test_synthetic_outputs_isolated_and_rng_unchanged():
    out = Path(survival.config.RESULTS_DIR)
    out.mkdir()
    for name in ("survival_analysis.csv", "survival_validation.csv"):
        (out / name).write_text("authentic data remains untouched")
    np.random.seed(88)
    before = np.random.get_state()
    result = survival.run_survival_pipeline(force_synthetic=True)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert len(result)
    assert set(result["data_provenance"]) == {"synthetic_simulation"}
    assert set(result["cohort_accession"]) == {"SYNTHETIC"}
    for name in ("survival_analysis.csv", "survival_validation.csv"):
        assert (out / name).read_text() == "authentic data remains untouched"
        assert (out / "synthetic_test" / name).exists()


@pytest.mark.parametrize("mode", ["failure", "empty", "missing_gene"])
def test_stale_outputs_cleared_on_empty_or_failure(monkeypatch, cohort, mode):
    out = Path(survival.config.RESULTS_DIR)
    out.mkdir()
    for name in ("survival_analysis.csv", "survival_validation.csv"):
        (out / name).write_text("gene,os_pvalue\nSTALE,.0001\n")
    if mode == "failure":
        def load(force_synthetic=False):
            raise RuntimeError("cohort unavailable")
        monkeypatch.setattr(survival, "load_survival_cohort", load)
    elif mode == "empty":
        monkeypatch.setattr(survival, "load_survival_cohort", lambda force_synthetic=False: (cohort[0], cohort[1].iloc[:0]))
    else:
        monkeypatch.setattr(survival, "load_survival_cohort", lambda force_synthetic=False: cohort)
        (out / "consensus_biomarkers.csv").write_text("gene\nABSENT\n")
    result = survival.run_survival_pipeline()
    assert result.empty
    for name in ("survival_analysis.csv", "survival_validation.csv"):
        assert pd.read_csv(out / name).empty


def test_direct_analysis_keeps_summary_and_curve_contracts(cohort):
    expression, clinical = cohort
    metadata = {sample: {"characteristics": [f"survival_months: {row.SURV_DEATH * 12}", f"vital_status: {int(row.DEATH)}"]}
                for sample, row in clinical.iterrows()}
    result, curves = survival.run_survival_analysis(expression, metadata, ["GENE", "GENE", "MYC"])
    assert len(result) == 2
    assert set(survival._ANALYSIS_COLUMNS) == set(result)
    expected = survival.evaluate_biomarker_survival("GENE", expr_df=expression, clinical_df=clinical)
    assert curves["GENE"]["p_value"] == pytest.approx(expected["p_value"])
    np.testing.assert_allclose(curves["GENE"]["high"]["time"], expected["km_high"]["timeline"])
    validation = pd.read_csv(Path(survival.config.RESULTS_DIR) / "survival_validation.csv")
    assert len(validation) == 2
    assert validation["rfs_pvalue"].isna().all()


def test_direct_analysis_empty_and_missing_gene_are_safe(cohort):
    expression, _ = cohort
    result, curves = survival.run_survival_analysis(expression, {}, ["GENE"])
    assert result.empty and curves == {}
    meta = {sample: {"characteristics": [f"survival_months: {i+1}", "vital_status: dead"]}
            for i, sample in enumerate(expression.columns)}
    result, curves = survival.run_survival_analysis(expression, meta, ["ABSENT"])
    assert result.empty and curves == {}
