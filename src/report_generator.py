"""Generate a descriptive HTML report from saved discovery-analysis artifacts.

Missing results stay unavailable. Artifact presence alone does not verify pipeline
integrity, leakage control, cohort authenticity, or clinical validity.
"""
import os
import sys
import json
from html import escape

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger


UNAVAILABLE = "Unavailable"


def _read_table(out_dir, filename):
    try:
        return pd.read_csv(os.path.join(out_dir, filename))
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        logger.info("Report table unavailable (%s): %s", filename, exc)
        return pd.DataFrame()


def _read_metadata(out_dir, filename):
    try:
        with open(os.path.join(out_dir, filename), encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError) as exc:
        logger.info("Report metadata unavailable (%s): %s", filename, exc)
        return {}


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _format(value, spec=".3f", percent=False):
    value = _number(value)
    if value is None or (percent and not 0 <= value <= 1):
        return UNAVAILABLE
    return format(value * 100 if percent else value, spec) + ("%" if percent else "")


def _text(value):
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return UNAVAILABLE
    return escape(str(value)) if str(value).strip() else UNAVAILABLE


def _fold_mean(frame, column):
    """Average only finite, valid recorded fold values; never invent missing folds."""
    if column not in frame:
        return None, 0
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values[np.isfinite(values) & values.between(0, 1)]
    return (float(valid.mean()), len(valid)) if len(valid) else (None, 0)


def _metric_card(label, value):
    return f'<div class="meta-card"><div class="meta-lbl">{escape(label)}</div><div class="meta-val">{value}</div></div>'


def generate_html_report(out_dir: str = None) -> str:
    """Summarize producer-schema values, with explicit missingness and provenance."""
    out_dir = out_dir if out_dir is not None else config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    de_df = _read_table(out_dir, "de_results.csv")
    ml_df = _read_table(out_dir, "ml_ranking_top100.csv")
    cons_df = _read_table(out_dir, "consensus_biomarkers.csv")
    nested_df = _read_table(out_dir, "nested_cv_metrics.csv")
    path_df = _read_table(out_dir, "pathway_enrichment.csv")
    qc_meta = _read_metadata(out_dir, "qc_summary.json")
    risk_meta = _read_metadata(out_dir, "diagnostic_performance.json")

    # The saved diagnostic producer is flat; also accept its in-memory metrics wrapper.
    risk_metrics = risk_meta.get("metrics", risk_meta)
    if not isinstance(risk_metrics, dict):
        risk_metrics = {}
    provenance = risk_metrics.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}

    de_count = UNAVAILABLE
    if (not de_df.empty and "regulation" in de_df
            and de_df["regulation"].isin(["Upregulated", "Downregulated", "Not Significant"]).all()):
        n_up = int(de_df["regulation"].eq("Upregulated").sum())
        n_down = int(de_df["regulation"].eq("Downregulated").sum())
        de_count = f'{n_up + n_down:,} <span class="detail">({n_up} Up / {n_down} Down)</span>'
    consensus_count = UNAVAILABLE
    if not cons_df.empty and "gene" in cons_df and cons_df["gene"].notna().all():
        consensus_count = f'{cons_df["gene"].nunique():,} Genes'
    nested_acc, n_acc = _fold_mean(nested_df, "fold_accuracy")
    nested_auc, n_auc = _fold_mean(nested_df, "fold_roc_auc")
    overview_cards = "".join([
        _metric_card("Significant DEGs", de_count),
        _metric_card("Consensus Biomarkers", consensus_count),
        _metric_card("Recorded nested CV accuracy", _format(nested_acc, ".1f", percent=True)),
        _metric_card("Recorded nested CV ROC-AUC", _format(nested_auc)),
    ])

    # The consensus producer omits per-model importances: recover the exact ML
    # producer columns by gene, not row order. Ambiguous duplicate IDs stay missing.
    cons_table = cons_df.copy()
    if "gene" in cons_table and "gene" in ml_df:
        unique_ml = ml_df[ml_df["gene"].notna() & ~ml_df["gene"].duplicated(keep=False)].set_index("gene")
        for column in ("RF_importance", "GB_importance"):
            if column in unique_ml:
                mapped = cons_table["gene"].map(unique_ml[column])
                cons_table[column] = cons_table[column].combine_first(mapped) if column in cons_table else mapped

    cons_rows = []
    for _, row in cons_table.head(15).iterrows():
        cons_rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in (
            _text(row.get("gene")), _format(row.get("log2FC"), ".2f"),
            _format(row.get("adj_pvalue"), ".2e"),
            _format(row.get("RF_importance"), ".4f"),
            _format(row.get("GB_importance"), ".4f"),
            _format(row.get("composite_score")), _text(row.get("consensus_tier")),
        )) + "</tr>")
    cons_rows = "".join(cons_rows) or '<tr><td colspan="7">Unavailable: no consensus biomarker rows.</td></tr>'

    path_rows = []
    for _, row in path_df.head(8).iterrows():
        fold = _format(row.get("fold_enrichment"), ".2f")
        path_rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in (
            _text(row.get("pathway")), _text(row.get("database")),
            f'{_format(row.get("overlap_count"), ".0f")} / {_format(row.get("pathway_size"), ".0f")}',
            fold + ("x" if fold != UNAVAILABLE else ""),
            _format(row.get("adjusted_p_value"), ".2e"), _text(row.get("genes")),
        )) + "</tr>")
    path_rows = "".join(path_rows) or '<tr><td colspan="6">Unavailable: no pathway enrichment rows.</td></tr>'

    training_cards = "".join([
        _metric_card("Apparent sensitivity", _format(risk_metrics.get("sensitivity"), ".1f", True)),
        _metric_card("Apparent specificity", _format(risk_metrics.get("specificity"), ".1f", True)),
        _metric_card("Apparent PPV", _format(risk_metrics.get("ppv"), ".1f", True)),
        _metric_card("Training-fitted cutoff (Youden J)", _format(risk_metrics.get("optimal_decision_threshold"))),
    ])
    validation_status = _text(risk_metrics.get("validation_status"))
    evaluation_scope = _text(risk_metrics.get("evaluation_scope", risk_metrics.get("evaluation_type")))

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Biomarker Discovery Report - {_text(config.GEO_ACCESSION)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 32px; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 32px; border: 1px solid #334155; }}
        h1 {{ font-size: 28px; margin-top: 0; }}
        h2 {{ font-size: 20px; margin-top: 30px; }}
        p {{ line-height: 1.6; }}
        .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 16px; margin: 24px 0; }}
        .meta-card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; }}
        .meta-val {{ font-size: 24px; font-weight: 700; margin-top: 8px; overflow-wrap: anywhere; }}
        .meta-lbl, .detail {{ font-size: 13px; color: #cbd5e1; }}
        .detail {{ display: block; font-weight: 400; }}
        .table-wrap {{ overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
        th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; }}
        .notice {{ border-left: 3px solid #cbd5e1; padding-left: 16px; }}
        .footer {{ margin-top: 36px; padding-top: 16px; border-top: 1px solid #334155; color: #cbd5e1; font-size: 12px; }}
        @media (max-width: 600px) {{ body {{ padding: 12px; }} .container {{ padding: 16px; }} }}
    </style>
</head>
<body>
<main class="container">
    <h1>Cancer Biomarker Discovery Report</h1>
    <p>Configured cohort: <strong>{_text(config.GEO_ACCESSION)}</strong> — {_text(config.CANCER_TYPE)}.
    Configuration is not verification of the saved artifacts' cohort or data provenance.</p>
    <p class="notice">Exploratory research summary. Unavailable means the result is absent, empty, invalid,
    or not recorded; it does not mean zero. Pipeline integrity and clinical validity are not verified by this report.</p>
    <div class="meta-grid">{overview_cards}</div>
    <p>Nested metrics summarize recorded <code>fold_accuracy</code> and <code>fold_roc_auc</code> values
    (valid accuracy folds: {n_acc}/{len(nested_df)}; valid AUC folds: {n_auc}/{len(nested_df)}).
    These are internal CV results, not independent clinical validation of the weighted signature.
    Fold-local preprocessing, feature selection, and patient separation must be audited separately.</p>

    <h2>1. Top Consensus Biomarkers (Multi-Model ML + DE)</h2>
    <p>Composite scores and adjusted p-values come from the consensus table; RF/GB importances
    are joined by gene from the ML ranking when absent there. Missing importances remain unavailable.</p>
    <div class="table-wrap"><table>
        <thead><tr><th>Gene Symbol</th><th>log2 Fold-Change</th><th>Adj. p-value (FDR)</th>
        <th>Random Forest Importance</th><th>Gradient Boosting Importance</th><th>Composite Score</th><th>Consensus Tier</th></tr></thead>
        <tbody>{cons_rows}</tbody>
    </table></div>

    <h2>2. Functional Pathway &amp; Gene Ontology Enrichment</h2>
    <div class="table-wrap"><table>
        <thead><tr><th>Pathway / Biological Process</th><th>Database</th><th>Gene Overlap</th>
        <th>Fold Enrichment</th><th>FDR Adj. p-value</th><th>Overlapping Genes</th></tr></thead>
        <tbody>{path_rows}</tbody>
    </table></div>

    <h2>3. Apparent Training Performance — Exploratory Signature</h2>
    <p class="notice">The signature, gene scaling, and Youden cutoff are fitted on the same discovery
    cohort being evaluated. These are apparent training metrics, not held-out performance or estimates of clinical utility.
    Selection bias is not removed by bootstrapping these scores. No clinical confidence intervals are reported.
    A locked model and independent evaluation are required before assessing clinical use.</p>
    <p>Recorded validation status: <strong>{validation_status}</strong>.
    Recorded evaluation scope: <strong>{evaluation_scope}</strong>.
    Independent clinical validity: <strong>Not established by this report</strong>.</p>
    <div class="meta-grid">{training_cards}</div>
    <p>Weight source: {_text(provenance.get('weight_source'))};
    selection data: {_text(provenance.get('selection_data'))};
    scaling fit data: {_text(provenance.get('standardization_fit_data'))};
    cutoff fit data: {_text(provenance.get('threshold_fit_data'))};
    evaluation data: {_text(provenance.get('evaluation_data'))}.</p>

    <h2>4. Recorded Sample Quality Control</h2>
    <p>Samples: {_format(qc_meta.get('n_samples'), '.0f')};
    genes: {_format(qc_meta.get('n_genes'), '.0f')};
    flagged outliers: {_format(qc_meta.get('n_outliers'), '.0f')}.
    QC summaries do not certify absence of bias or data leakage.</p>
    <div class="footer">Generated from saved analysis artifacts by Cancer Biomarker Discovery Pipeline.
    Descriptive research report; not a clinical diagnostic validation.</div>
</main>
</body>
</html>'''
    report_path = os.path.join(out_dir, "biomarker_discovery_report.html")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(html_content)
    logger.info("Descriptive HTML report generated: %s", report_path)
    return report_path


if __name__ == "__main__":
    print(f"Report generated: {generate_html_report()}")
