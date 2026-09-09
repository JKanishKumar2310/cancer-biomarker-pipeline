"""
Dashboard Layout — Defines the structure and components of the Dash app.

5-tab layout: Overview, Volcano Plot, Heatmap, Biomarker Ranking, Pathway Enrichment.
"""
import os
import sys
import dash_bootstrap_components as dbc
from dash import html, dcc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Color Constants ──────────────────────────────────────────
PLOT_BG = "rgba(10, 10, 26, 0)"
PAPER_BG = "rgba(10, 10, 26, 0)"
GRID_COLOR = "rgba(255, 255, 255, 0.05)"
FONT_COLOR = "#e8e8f0"
CYAN = "#00d4ff"
PURPLE = "#a855f7"
PINK = "#ec4899"
GREEN = "#10b981"
ORANGE = "#f59e0b"
RED = "#ef4444"

PLOT_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=12),
    margin=dict(l=50, r=30, t=40, b=50),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.1)",
        font=dict(size=11),
    ),
)


def make_stat_card(value, label, color_class, anim_delay):
    """Create a single stat card."""
    return html.Div(
        className=f"stat-card {color_class} fade-in fade-in-delay-{anim_delay}",
        children=[
            html.Div(str(value), className="stat-value"),
            html.Div(label, className="stat-label"),
        ],
    )


def create_layout():
    """Build the full dashboard layout."""
    return html.Div(
        className="main-container",
        children=[
            # ── Header ───────────────────────────────────────
            html.Div(
                className="dashboard-header fade-in",
                children=[
                    html.H1("🧬 Cancer Biomarker Discovery Dashboard"),
                    html.P(
                        f"Autonomous Discovery Pipeline • Current Dataset: {config.GEO_ACCESSION} ({config.CANCER_TYPE})",
                        id="dashboard-subtitle",
                        className="subtitle",
                    ),
                ],
            ),

            # ── AI GEO Dataset Ingestion & Curation Bar ──────
            dbc.Card(
                className="mb-4 p-3 fade-in",
                style={
                    "backgroundColor": "rgba(20, 20, 45, 0.75)",
                    "border": "1px solid rgba(0, 212, 255, 0.3)",
                    "borderRadius": "12px",
                    "boxShadow": "0 8px 32px 0 rgba(0, 0, 0, 0.37)",
                },
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.Span("🤖 AI Dataset Ingestion & Curation:", style={"fontWeight": "600", "color": CYAN, "marginRight": "8px"}),
                                            html.Span("Download from NCBI, inspect metadata with AI, and auto-analyze:", style={"color": "#a0a0b0", "fontSize": "0.9rem"}),
                                        ],
                                        className="mb-2",
                                    ),
                                    dbc.InputGroup(
                                        [
                                            dbc.Input(
                                                id="geo-accession-input",
                                                placeholder="Enter GEO Accession (e.g. GSE19804, GSE8671, GSE15852)...",
                                                value=config.GEO_ACCESSION,
                                                style={"backgroundColor": "rgba(10, 10, 26, 0.8)", "color": FONT_COLOR, "border": "1px solid rgba(255,255,255,0.15)"},
                                            ),
                                            dbc.Button(
                                                "⚡ AI Curate & Run",
                                                id="btn-run-geo",
                                                color="primary",
                                                style={"background": f"linear-gradient(135deg, {CYAN} 0%, {PURPLE} 100%)", "border": "none", "fontWeight": "600"},
                                            ),
                                        ],
                                        size="sm",
                                    ),
                                ],
                                md=7,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Quick Load Validated Cohorts:", style={"fontSize": "0.85rem", "color": "#a0a0b0", "marginBottom": "6px"}),
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button("Colon (GSE8671)", id="btn-quick-gse8671", size="sm", outline=True, color="info"),
                                            dbc.Button("Lung (GSE19804)", id="btn-quick-gse19804", size="sm", outline=True, color="info"),
                                            dbc.Button("Breast (GSE15852)", id="btn-quick-gse15852", size="sm", outline=True, color="info"),
                                        ],
                                        size="sm",
                                    ),
                                ],
                                md=5,
                                className="d-flex flex-column justify-content-center",
                            ),
                        ],
                        align="center",
                    ),
                    dcc.Loading(
                        id="geo-loading",
                        type="circle",
                        color=CYAN,
                        children=html.Div(id="geo-status-output", className="mt-2", style={"fontSize": "0.88rem"}),
                    ),
                ],
            ),

            # ── Stat Cards (populated by callback) ──────────
            html.Div(id="stats-row", className="stats-row"),

            # ── Tabs ─────────────────────────────────────────
            dbc.Tabs(
                id="main-tabs",
                className="custom-tabs",
                active_tab="tab-overview",
                children=[
                    # Tab 1: Overview
                    dbc.Tab(
                        label="📊 Overview",
                        tab_id="tab-overview",
                        children=[
                            html.Div(
                                className="two-column",
                                children=[
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("PCA — Sample Clustering", className="plot-title"),
                                            html.Div(
                                                "Tumor vs Normal samples projected onto first two principal components",
                                                className="plot-subtitle",
                                            ),
                                            dcc.Graph(id="pca-plot", config={"displayModeBar": True}),
                                        ],
                                    ),
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("Expression Distribution", className="plot-title"),
                                            html.Div(
                                                "Overall gene expression distribution per sample group",
                                                className="plot-subtitle",
                                            ),
                                            dcc.Graph(id="distribution-plot", config={"displayModeBar": True}),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # Tab 2: Volcano Plot
                    dbc.Tab(
                        label="🌋 Volcano Plot",
                        tab_id="tab-volcano",
                        children=[
                            html.Div(
                                className="plot-container",
                                children=[
                                    html.Div("Interactive Volcano Plot", className="plot-title"),
                                    html.Div(
                                        "Click on a gene dot to see its expression profile below. "
                                        "Adjust thresholds with the sliders.",
                                        className="plot-subtitle",
                                    ),
                                    html.Div(
                                        className="two-column",
                                        style={"marginBottom": "16px"},
                                        children=[
                                            html.Div([
                                                html.Div("Log2 Fold-Change Threshold", className="control-label"),
                                                dcc.Slider(
                                                    id="fc-slider",
                                                    min=0, max=4, step=0.25, value=1.0,
                                                    marks={i: str(i) for i in range(5)},
                                                    tooltip={"placement": "bottom"},
                                                ),
                                            ]),
                                            html.Div([
                                                html.Div("-log10(adj. p-value) Threshold", className="control-label"),
                                                dcc.Slider(
                                                    id="pval-slider",
                                                    min=0, max=10, step=0.5,
                                                    value=-1 * __import__("math").log10(0.05),
                                                    marks={i: str(i) for i in range(11)},
                                                    tooltip={"placement": "bottom"},
                                                ),
                                            ]),
                                        ],
                                    ),
                                    dcc.Graph(id="volcano-plot", config={"displayModeBar": True}),
                                ],
                            ),
                            html.Div(
                                className="plot-container",
                                id="gene-detail-container",
                                style={"display": "none"},
                                children=[
                                    html.Div(id="gene-detail-title", className="plot-title"),
                                    dcc.Graph(id="gene-boxplot"),
                                ],
                            ),
                        ],
                    ),

                    # Tab 3: Heatmap
                    dbc.Tab(
                        label="🔥 Heatmap",
                        tab_id="tab-heatmap",
                        children=[
                            html.Div(
                                className="plot-container",
                                children=[
                                    html.Div("Top DEG Expression Heatmap", className="plot-title"),
                                    html.Div(
                                        "Clustered heatmap of top differentially expressed genes across all samples",
                                        className="plot-subtitle",
                                    ),
                                    dcc.Graph(
                                        id="heatmap-plot",
                                        config={"displayModeBar": True},
                                        style={"height": "600px"},
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # Tab 4: Biomarker Ranking
                    dbc.Tab(
                        label="🎯 Biomarkers",
                        tab_id="tab-biomarkers",
                        children=[
                            html.Div(
                                className="plot-container",
                                children=[
                                    html.Div("Consensus Biomarker Ranking", className="plot-title"),
                                    html.Div(
                                        [
                                            html.Li("Machine Learning Ranking using Random Forest."),
                                            "Only genes significant in BOTH DE and ML are shown."
                                        ],
                                        className="plot-subtitle",
                                    ),
                                    dcc.Graph(id="biomarker-bar-chart", config={"displayModeBar": True}),
                                ],
                            ),
                            html.Div(
                                className="two-column",
                                children=[
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("Top 10 ML Features", className="plot-title"),
                                            html.Div(
                                                "Feature importance from Random Forest",
                                                className="plot-subtitle",
                                            ),
                                            dcc.Graph(id="model-comparison-plot"),
                                        ],
                                    ),
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("DE vs ML Importance", className="plot-title"),
                                            html.Div(
                                                "Scatter: statistical significance vs ML importance score",
                                                className="plot-subtitle",
                                            ),
                                            dcc.Graph(id="de-vs-ml-scatter"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # Tab 5: Pathway Enrichment
                    dbc.Tab(
                        label="🧬 Pathways",
                        tab_id="tab-pathways",
                        children=[
                            html.Div(
                                className="two-column",
                                children=[
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("GO Biological Process Enrichment", className="plot-title"),
                                            dcc.Graph(id="go-enrichment-plot", config={"displayModeBar": True}),
                                        ],
                                    ),
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("KEGG Pathway Enrichment", className="plot-title"),
                                            dcc.Graph(id="kegg-enrichment-plot", config={"displayModeBar": True}),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # Tab 6: Clinical Survival & Drugs
                    dbc.Tab(
                        label="⏳ Clinical Survival & Drugs",
                        tab_id="tab-survival",
                        children=[
                            html.Div(
                                className="control-panel",
                                style={"marginBottom": "20px"},
                                children=[
                                    html.Div(
                                        className="control-group",
                                        children=[
                                            html.Label("Select Biomarker for Clinical Outcomes:"),
                                            dcc.Dropdown(
                                                id="survival-gene-dropdown",
                                                options=[
                                                    {"label": f"{g} (Top Prognostic Target)", "value": g}
                                                    for g in ["MELK", "TOP2A", "TACSTD2", "GATA3", "PTEN", "CDH1", "KRT19", "BIN1", "PCNA", "CDK4", "ERBB2", "ESR1"]
                                                ],
                                                value="MELK",
                                                clearable=False,
                                                className="dark-dropdown",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="control-group",
                                        children=[
                                            html.Label("Survival Endpoint:"),
                                            dcc.RadioItems(
                                                id="survival-endpoint-radio",
                                                options=[
                                                    {"label": " Overall Survival (OS)", "value": "overall"},
                                                    {"label": " Relapse-Free Survival (RFS)", "value": "relapse"},
                                                ],
                                                value="overall",
                                                inline=True,
                                                className="radio-group",
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="two-column",
                                children=[
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("Kaplan-Meier Survival Curves (GSE1456 Cohort, n=159)", className="plot-title"),
                                            html.Div(
                                                "Patient stratification into High vs Low biomarker expression with Log-Rank test",
                                                className="plot-subtitle",
                                            ),
                                            dcc.Graph(id="kaplan-meier-plot", config={"displayModeBar": True}),
                                        ],
                                    ),
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("Targeted Oncology Drug Actionability", className="plot-title"),
                                            html.Div(
                                                "Matched FDA-approved therapeutics and clinical mechanisms",
                                                className="plot-subtitle",
                                            ),
                                            html.Div(id="drug-actionability-content"),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="two-column",
                                style={"marginTop": "20px"},
                                children=[
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("Cross-Cohort Generalization (Zero-Shot on GSE42568, n=121)", className="plot-title"),
                                            html.Div(
                                                "External ROC Curve: Model trained on GSE15852 evaluated on independent European cohort",
                                                className="plot-subtitle",
                                            ),
                                            dcc.Graph(id="external-roc-plot", config={"displayModeBar": True}),
                                        ],
                                    ),
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("External Validation Rigor Metrics", className="plot-title"),
                                            html.Div(
                                                "Key indicators of cross-cohort clinical robustness",
                                                className="plot-subtitle",
                                            ),
                                            html.Div(id="external-metrics-content"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # ── Footer ──────────────────────────────────────
            html.Div(
                className="dashboard-footer",
                children=[
                    html.P("Cancer Biomarker Discovery Dashboard • Powered by Python, Plotly Dash, scikit-learn"),
                    html.P("Data source: NCBI GEO (GSE15852) • For academic research purposes only"),
                ],
            ),
        ],
    )
