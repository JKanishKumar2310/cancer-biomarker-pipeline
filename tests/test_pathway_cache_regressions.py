"""Offline regressions for pathway statistics and missing-cohort-cache guards.

AST loading executes the actual production function bodies without importing the
Dash application, reading active CSVs at startup, or initializing API clients.
REGRESSION_SOURCE_ROOT optionally selects another checkout for verification.
"""
import ast
import logging
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


ROOT = Path(os.environ.get("REGRESSION_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
LEGACY_COLUMNS = {"Term", "Gene_set", "Adjusted P-value", "P-value", "Genes", "Overlap"}
QUICK_ACCESSIONS = ["GSE53757", "GSE30784", "GSE8671", "GSE19804", "GSE15852"]


def load_functions(relative_path, names, namespace, parent=None):
    """Compile selected functions verbatim, omitting only Dash decorators."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    body = tree.body
    if parent:
        body = next(node for node in body if isinstance(node, ast.FunctionDef)
                    and node.name == parent).body
    selected = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    for node in selected:
        node.decorator_list = []
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden in pathway/cache regressions")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.fixture
def pathway():
    namespace = {
        "pd": pd, "np": np, "os": os, "fisher_exact": fisher_exact,
        "logger": logging.getLogger("pathway-regression"),
        "config": SimpleNamespace(PVALUE_THRESHOLD=.05, FC_THRESHOLD=1.),
        "CANONICAL_PATHWAYS": {"Example (KEGG:test)": ["UP", "DOWN", "CONSENSUS"]},
    }
    functions = load_functions("src/pathway_enrichment.py",
                              ["run_pathway_enrichment", "analyze_and_export_pathways"], namespace)
    return functions, namespace


@pytest.mark.parametrize("annotations,expected", [
    ({"regulation": ["Upregulated", "Downregulated", "Not Significant", None]}, {"UP", "DOWN"}),
    ({"significant": [True, False, False, True]}, {"UP", "OTHER"}),
    ({"adj_pvalue": [.01, .01, .2, .01], "log2FC": [2., -2., 3., .1]}, {"UP", "DOWN"}),
    # Regulation is authoritative; neither a contradictory bool nor p-value overrides it.
    ({"regulation": ["Not Significant", "Downregulated", "unknown", None],
      "significant": [True, False, True, True],
      "adj_pvalue": [.001] * 4, "log2FC": [3.] * 4}, {"DOWN"}),
])
def test_export_selects_only_significant_degs_plus_consensus(pathway, tmp_path, annotations, expected):
    functions, namespace = pathway
    de = pd.DataFrame({"gene": ["UP", "DOWN", "NOISE", "OTHER"], **annotations})
    captured = Mock(return_value=pd.DataFrame(columns=sorted(LEGACY_COLUMNS)))
    namespace["run_pathway_enrichment"] = captured
    functions.analyze_and_export_pathways(de, pd.DataFrame({"gene": ["CONSENSUS", None]}), str(tmp_path))
    args, kwargs = captured.call_args
    assert set(args[0]) == expected | {"CONSENSUS"}
    assert kwargs["background_genes"] == de["gene"].tolist()
    assert (tmp_path / "pathway_enrichment.csv").is_file()


def test_numeric_significance_uses_strict_fdr_and_absolute_fc_thresholds(pathway, tmp_path):
    functions, namespace = pathway
    de = pd.DataFrame({
        "gene": ["UP", "DOWN", "P_BOUNDARY", "FC_BOUNDARY", "NEG_FC_BOUNDARY", "NAN"],
        "adj_pvalue": [.049, .049, .05, .001, .001, np.nan],
        "log2FC": [1.01, -1.01, 2., 1., -1., 4.],
        "significant": [True] * 6,
    })
    captured = Mock(return_value=pd.DataFrame())
    namespace["run_pathway_enrichment"] = captured
    functions.analyze_and_export_pathways(de, pd.DataFrame(), str(tmp_path))
    assert set(captured.call_args.args[0]) == {"UP", "DOWN"}


def test_export_rejects_unannotated_de_instead_of_using_all_genes(pathway, tmp_path):
    functions, _ = pathway
    with pytest.raises(ValueError, match="significance"):
        functions.analyze_and_export_pathways(pd.DataFrame({"gene": ["UP"]}), pd.DataFrame(), str(tmp_path))
    assert not (tmp_path / "pathway_enrichment.csv").exists()


def test_full_family_bh_keeps_zero_overlap_hypotheses_until_after_correction(pathway):
    functions, _ = pathway
    database = {
        "strong (KEGG:1)": ["A", "B"],
        "weak (GO:1)": ["A", "C", "D"],
        "no overlap": ["E", "F"],
        "outside universe": ["OUTSIDE"],
        "empty pathway": [],
    }
    result = functions.run_pathway_enrichment(["A", "B"], pathway_db=database,
                                             background_genes=list("ABCDEFGHIJ"))
    raw = [fisher_exact([[2, 0], [0, 8]], alternative="greater").pvalue,
           fisher_exact([[1, 1], [2, 6]], alternative="greater").pvalue, 1., 1., 1.]
    adjusted = multipletests(raw, method="fdr_bh")[1]
    indexed = result.set_index("pathway")
    assert set(indexed.index) == {"strong (KEGG:1)", "weak (GO:1)"}
    np.testing.assert_allclose(indexed.loc[list(database)[:2], "p_value"], raw[:2])
    np.testing.assert_allclose(indexed.loc[list(database)[:2], "adjusted_p_value"], adjusted[:2])
    assert indexed.loc["strong (KEGG:1)", "adjusted_p_value"] > multipletests(raw[:2], method="fdr_bh")[1][0]
    assert result["adjusted_p_value"].is_monotonic_increasing


def test_bh_reverse_cumulative_minimum_and_ties(pathway):
    functions, namespace = pathway
    pvalues = iter([.01, .011, .011, .2])
    namespace["fisher_exact"] = lambda *a, **kw: SimpleNamespace(pvalue=next(pvalues))
    database = {f"term{i}": ["A"] for i in range(4)}
    database["zero"] = ["B"]
    result = functions.run_pathway_enrichment(["A"], background_gene_count=10, pathway_db=database)
    np.testing.assert_allclose(result["adjusted_p_value"],
                               multipletests([.01, .011, .011, .2, 1.], method="fdr_bh")[1][:4])


def test_query_and_pathways_intersect_normalized_measured_universe(pathway):
    functions, _ = pathway
    result = functions.run_pathway_enrichment(
        [" a ", "A", "OUTSIDE", None, ""], background_gene_count=1,
        background_genes=["a", " B ", "C", "D", "a", None, " "],
        pathway_db={"Measured (GO:1)": ["A", "b", "OUTSIDE"], "Unmeasured": ["OUTSIDE"]},
    )
    row = result.iloc[0]
    assert len(result) == 1
    assert row["overlap_count"] == 1
    assert row["pathway_size"] == 2
    assert row["genes"] == "A"
    assert row["fold_enrichment"] == pytest.approx(2.)
    assert row["p_value"] == pytest.approx(.5)
    assert row["adjusted_p_value"] == pytest.approx(1.)


@pytest.mark.parametrize("kwargs", [
    {"background_gene_count": 0},
    {"background_gene_count": 2},
])
def test_impossible_background_is_rejected_instead_of_clamped(pathway, kwargs):
    functions, _ = pathway
    with pytest.raises(ValueError, match="[Bb]ackground"):
        functions.run_pathway_enrichment(["A", "B"], pathway_db={"term": ["B", "C"]}, **kwargs)


@pytest.mark.parametrize("genes,database,universe", [
    ([], {"term": ["A"]}, ["A", "B"]),
    (["A"], {}, ["A", "B"]),
    (["OUTSIDE"], {"term": ["A"]}, ["A", "B"]),
    (["A"], {"term": ["A"]}, []),
])
def test_empty_results_retain_readable_legacy_csv_schema(pathway, tmp_path, genes, database, universe):
    functions, _ = pathway
    result = functions.run_pathway_enrichment(genes, pathway_db=database, background_genes=universe)
    assert result.empty
    assert LEGACY_COLUMNS <= set(result.columns)
    destination = tmp_path / "empty.csv"
    result.to_csv(destination, index=False)
    assert LEGACY_COLUMNS <= set(pd.read_csv(destination).columns)


def test_export_legacy_dashboard_columns_round_trip(pathway, tmp_path):
    functions, namespace = pathway
    namespace["CANONICAL_PATHWAYS"] = {"Cycle (KEGG:1)": ["UP"], "Process (GO:1)": ["DOWN"]}
    de = pd.DataFrame({"gene": ["UP", "DOWN", "NOISE"],
                       "regulation": ["Upregulated", "Downregulated", "Not Significant"]})
    result = functions.analyze_and_export_pathways(de, pd.DataFrame(), str(tmp_path))
    saved = pd.read_csv(tmp_path / "pathway_enrichment.csv")
    pd.testing.assert_frame_equal(result, saved)
    for legacy, current in [("Term", "pathway"), ("Gene_set", "database"),
                            ("Adjusted P-value", "adjusted_p_value"), ("P-value", "p_value"),
                            ("Genes", "genes")]:
        pd.testing.assert_series_equal(saved[legacy], saved[current], check_names=False)
    assert saved["Overlap"].tolist() == ["1/1", "1/1"]
    # These are the exact column operations used to split/sort dashboard results.
    for category in ["GO", "KEGG"]:
        subset = saved[saved["Gene_set"].str.contains(category, case=False, na=False)]
        assert len(subset.sort_values("Adjusted P-value")["Term"].str[:50]) == 1


@pytest.fixture
def cache(tmp_path):
    data_dir, results_dir = tmp_path / "data", tmp_path / "results"
    data_dir.mkdir()
    results_dir.mkdir()
    (data_dir / "expression_matrix.csv").write_text("gene,old_sample\nOLD,10\n")
    (data_dir / "sample_labels.csv").write_text("sample,condition\nold_sample,Normal\n")
    (results_dir / "consensus_biomarkers.csv").write_text("gene\nOLD\n")
    config = SimpleNamespace(BASE_DIR=str(tmp_path), DATA_DIR=str(data_dir), RESULTS_DIR=str(results_dir),
                             GEO_ACCESSION="GSE_ACTIVE", CANCER_TYPE="Active cancer")
    active_frame = pd.DataFrame({"gene": ["OLD"]})
    namespace = {
        "os": os, "pd": pd, "config": config, "RESULTS": {"consensus": active_frame},
        "no_update": object(), "ctx": SimpleNamespace(triggered_id=None),
        "dbc": SimpleNamespace(Alert=lambda children, **kwargs: SimpleNamespace(children=children, **kwargs)),
        "load_results": Mock(side_effect=AssertionError("Missing cache must not reload results")),
    }
    functions = load_functions("dashboard/callbacks.py", ["activate_cached_cohort", "handle_cohort_actions"],
                              namespace, parent="register_callbacks")
    return functions, namespace, tmp_path


def snapshot_files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def make_incomplete_cache(root, accession, mode):
    if mode != "missing_directory":
        directory = root / "results_cache" / accession
        directory.mkdir(parents=True)
        (directory / "de_results.csv").write_text("gene\nSHOULD_NOT_COPY\n")
        if mode == "sentinel_directory":
            (directory / "consensus_biomarkers.csv").mkdir()
    # Cached expression must not be copied before checking analysis readiness.
    (root / "data" / f"{accession}_expression_matrix.csv").write_text("gene,new_sample\nNEW,99\n")
    (root / "data" / f"{accession}_sample_labels.csv").write_text("sample,condition\nnew_sample,Tumor\n")


def assert_active_unchanged(namespace, root, before_files, before_results, before_config):
    assert snapshot_files(root) == before_files
    assert vars(namespace["config"]) == before_config
    assert namespace["RESULTS"].keys() == before_results.keys()
    for key, value in before_results.items():
        assert namespace["RESULTS"][key] is value
        pd.testing.assert_frame_equal(value, pd.DataFrame({"gene": ["OLD"]}))
    namespace["load_results"].assert_not_called()


@pytest.mark.parametrize("accession", QUICK_ACCESSIONS)
@pytest.mark.parametrize("mode", ["missing_directory", "missing_sentinel", "sentinel_directory"])
def test_all_quick_load_missing_cache_guards_preserve_active_state(cache, accession, mode):
    functions, namespace, root = cache
    make_incomplete_cache(root, accession, mode)
    namespace["ctx"].triggered_id = f"btn-quick-{accession.lower()}"
    before_files = snapshot_files(root)
    before_results = namespace["RESULTS"].copy()
    before_config = vars(namespace["config"]).copy()
    store, subtitle, status, input_value = functions.handle_cohort_actions(*([0] * 7), "GSE_ACTIVE")
    assert store is namespace["no_update"]
    assert subtitle is namespace["no_update"]
    assert input_value is namespace["no_update"]
    assert status.color == "warning"
    assert accession in status.children and "unchanged" in status.children
    assert_active_unchanged(namespace, root, before_files, before_results, before_config)


@pytest.mark.parametrize("mode", ["missing_directory", "missing_sentinel", "sentinel_directory"])
def test_activate_cached_cohort_itself_preserves_state_on_missing_cache(cache, mode):
    functions, namespace, root = cache
    make_incomplete_cache(root, "GSE_MISSING", mode)
    before_files = snapshot_files(root)
    before_results = namespace["RESULTS"].copy()
    before_config = vars(namespace["config"]).copy()
    assert functions.activate_cached_cohort("GSE_MISSING", "Must not become active") is False
    assert_active_unchanged(namespace, root, before_files, before_results, before_config)


def test_complete_cache_activation_still_copies_and_reloads(cache):
    functions, namespace, root = cache
    accession = "GSE53757"
    directory = root / "results_cache" / accession
    directory.mkdir(parents=True)
    (directory / "consensus_biomarkers.csv").write_text("gene\nNEW\n")
    (root / "data" / f"{accession}_expression_matrix.csv").write_text("gene,new_sample\nNEW,99\n")
    (root / "data" / f"{accession}_sample_labels.csv").write_text("sample,condition\nnew_sample,Tumor\n")
    load_functions("dashboard/callbacks.py", ["load_results"], namespace)
    namespace["ctx"].triggered_id = "btn-quick-gse53757"
    store, _, status, input_value = functions.handle_cohort_actions(*([0] * 7), "GSE_ACTIVE")
    assert store["accession"] == accession == input_value == namespace["config"].GEO_ACCESSION
    assert status.color == "success"
    assert namespace["RESULTS"]["consensus"]["gene"].tolist() == ["NEW"]
    assert namespace["RESULTS"]["expression"].columns.tolist() == ["new_sample"]
    assert namespace["RESULTS"]["labels"].to_dict() == {"new_sample": "Tumor"}
