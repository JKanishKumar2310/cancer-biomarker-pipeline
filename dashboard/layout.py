"""
Dashboard Layout — Defines the bespoke structure and components of the Dash app.
Precision Oncology & Biomarker Discovery Research Workbench.
"""
import os
import sys
import dash_bootstrap_components as dbc
from dash import html, dcc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Color Palette Constants (Scientific & Clinical Precision) ─
PLOT_BG = "rgba(0, 0, 0, 0)"
PAPER_BG = "rgba(0, 0, 0, 0)"
GRID_COLOR = "rgba(255, 255, 255, 0.04)"
FONT_COLOR = "#f8fafc"
FONT_MUTED = "#94a3b8"

CYAN = "#38bdf8"       # Azure / Focus
PURPLE = "#818cf8"     # Indigo / Target
PINK = "#f43f5e"       # Rose / Downregulated / Tumor
GREEN = "#10b981"      # Emerald / Upregulated / Normal
ORANGE = "#f59e0b"     # Amber / Somatic
RED = "#f43f5e"

PLOT_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color=FONT_MUTED, size=11),
    margin=dict(l=48, r=24, t=32, b=44),
    hoverlabel=dict(
        bgcolor="#0f172a",
        font_size=12,
        font_family="Plus Jakarta Sans, Inter, sans-serif",
        font_color="#f8fafc",
        bordercolor="rgba(255, 255, 255, 0.15)",
    ),
)


def make_stat_card(value, label, color_class, anim_delay, subtext=""):
    """Create a single refined scientific stat card."""
    return html.Div(
        className=f"stat-card {color_class}",
        children=[
            html.Div(
                className="stat-card-header",
                children=[
                    html.Span(label, className="stat-label"),
                    html.Span("●", className="stat-icon", style={"color": "currentColor"}),
                ],
            ),
            html.Div(str(value), className="stat-value"),
            html.Div(subtext if subtext else "Live Dataset Metric", className="stat-subtext"),
        ],
    )


def create_layout():
    """Build the full precision dashboard layout."""
    return html.Div(
        className="main-container",
        children=[
            # ── Top Bar / Header ─────────────────────────────
            html.Div(
                className="app-topbar",
                children=[
                    html.Div(
                        className="brand-section",
                        children=[
                            html.Div("🧬", className="brand-icon-box"),
                            html.Div([
                                html.Div([
                                    html.Span("ONCO•DISCOVERY", className="brand-title"),
                                    html.Span("RESEARCH SUITE", className="platform-tag"),
                                ], className="d-flex align-items-center gap-2"),
                                html.Div(
                                    f"Autonomous Biomarker Engine • Active: {config.GEO_ACCESSION} ({config.CANCER_TYPE})",
                                    id="dashboard-subtitle",
                                    className="brand-subtitle",
                                ),
                            ]),
                        ],
                    ),
                    html.Div(
                        className="topbar-meta",
                        children=[
                            html.Div([
                                html.Span(className="status-dot"),
                                html.Span("Pipeline Engine Live", style={"color": "#f8fafc"}),
                            ], className="live-status-pill"),
                            dbc.Badge("v2.4 Pro", color="dark", className="border text-secondary px-2 py-1", style={"fontSize": "0.72rem"}),
                        ],
                    ),
                ],
            ),

            dcc.Store(id="active-cohort-store", data={"accession": config.GEO_ACCESSION, "cancer_type": config.CANCER_TYPE}),

            # ── Command Bar / Cohort Ingestion ────────────────
            html.Div(
                className="command-card",
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div("Cohort Ingestion & NCBI GEO Query", className="command-label"),
                                    dbc.InputGroup(
                                        [
                                            dbc.Input(
                                                id="geo-accession-input",
                                                placeholder="Enter NCBI GEO Accession (e.g., GSE8671, GSE19804, GSE15852)...",
                                                value="",
                                                style={
                                                    "backgroundColor": "var(--bg-surface-1)",
                                                    "color": "var(--text-primary)",
                                                    "border": "1px solid var(--border-medium)",
                                                    "fontSize": "0.85rem",
                                                },
                                            ),
                                            dbc.Button(
                                                "Curate & Analyze",
                                                id="btn-run-geo",
                                                color="primary",
                                                style={
                                                    "background": "var(--accent-blue)",
                                                    "border": "none",
                                                    "fontWeight": "600",
                                                    "fontSize": "0.82rem",
                                                    "color": "#06090e",
                                                },
                                            ),
                                        ],
                                        size="sm",
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div("Fast-Load Validated Cohorts:", style={"fontSize": "0.78rem", "color": "var(--text-muted)", "marginBottom": "6px", "fontWeight": "500"}),
                                    html.Div(
                                        className="cohort-pills-wrap",
                                        children=[
                                            dbc.Button("Kidney (GSE53757)", id="btn-quick-gse53757", size="sm", className="cohort-pill-btn"),
                                            dbc.Button("Oral (GSE30784)", id="btn-quick-gse30784", size="sm", className="cohort-pill-btn"),
                                            dbc.Button("Colon (GSE8671)", id="btn-quick-gse8671", size="sm", className="cohort-pill-btn"),
                                            dbc.Button("Lung (GSE19804)", id="btn-quick-gse19804", size="sm", className="cohort-pill-btn"),
                                            dbc.Button("Breast (GSE15852)", id="btn-quick-gse15852", size="sm", className="cohort-pill-btn"),
                                            dbc.Button("Standby", id="btn-reset-cohort", size="sm", className="cohort-pill-btn"),
                                        ],
                                    ),
                                ],
                                md=6,
                                className="d-flex flex-column justify-content-center mt-2 mt-md-0",
                            ),
                        ],
                        align="center",
                    ),
                    dcc.Loading(
                        id="geo-loading",
                        type="dot",
                        color=CYAN,
                        children=html.Div(id="geo-status-output", className="mt-2", style={"fontSize": "0.82rem"}),
                    ),
                ],
            ),

            # ── Standby Welcome Banner ───────────────────────
            html.Div(id="welcome-banner"),

            # ── Stat Cards ───────────────────────────────────
            html.Div(id="stats-row", className="stats-row"),

            # ── Tabs Navigation ──────────────────────────────
            dbc.Tabs(
                id="main-tabs",
                className="custom-tabs",
                active_tab="tab-overview",
                children=[
                    # Tab 1: Overview
                    dbc.Tab(
                        label="Sample Landscape",
                        tab_id="tab-overview",
                        children=[
                            html.Div(
                                className="two-column",
                                children=[
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div(
                                                className="plot-header-bar",
                                                children=[
                                                    html.Div([
                                                        html.Div("Principal Component Analysis (PCA)", className="plot-title"),
                                                        html.Div("Projection of high-dimensional transcriptomes onto variance axes", className="plot-subtitle"),
                                                    ]),
                                                ],
                                            ),
                                            dcc.Graph(id="pca-plot", config={"displayModeBar": True, "displaylogo": False}),
                                        ],
                                    ),
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div(
                                                className="plot-header-bar",
                                                children=[
                                                    html.Div([
                                                        html.Div("Cohort Expression Distribution", className="plot-title"),
                                                        html.Div("Quantile density distribution across stratified sample groups", className="plot-subtitle"),
                                                    ]),
                                                ],
                                            ),
                                            dcc.Graph(id="distribution-plot", config={"displayModeBar": True, "displaylogo": False}),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # Tab 2: Volcano Plot
                    dbc.Tab(
                        label="Differential Expression",
                        tab_id="tab-volcano",
                        children=[
                            html.Div(
                                className="plot-container",
                                children=[
                                    html.Div(
                                        className="plot-header-bar",
                                        children=[
                                            html.Div([
                                                html.Div("Volcano Plot Matrix", className="plot-title"),
                                                html.Div("Click any individual gene locus to inspect its full sample expression distribution below.", className="plot-subtitle"),
                                            ]),
                                        ],
                                    ),
                                    html.Div(
                                        className="control-panel",
                                        children=[
                                            html.Div(
                                                className="two-column",
                                                children=[
                                                    html.Div([
                                                        html.Div("Log2 Fold-Change Threshold (|Log2FC|)", className="control-label"),
                                                        dcc.Slider(
                                                            id="fc-slider",
                                                            min=0, max=4, step=0.25, value=1.0,
                                                            marks={i: f"{i}.0" for i in range(5)},
                                                            tooltip={"placement": "bottom"},
                                                        ),
                                                    ]),
                                                    html.Div([
                                                        html.Div("-Log10 Adjusted P-Value Threshold (FDR)", className="control-label"),
                                                        dcc.Slider(
                                                            id="pval-slider",
                                                            min=0, max=10, step=0.5,
                                                            value=-1 * __import__("math").log10(0.05),
                                                            marks={i: f"{i}" for i in range(11)},
                                                            tooltip={"placement": "bottom"},
                                                        ),
                                                    ]),
                                                ],
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(id="volcano-plot", config={"displayModeBar": True, "displaylogo": False}),
                                ],
                            ),
                            html.Div(
                                className="plot-container",
                                id="gene-detail-container",
                                style={"display": "none"},
                                children=[
                                    html.Div(id="gene-detail-title", className="plot-title"),
                                    dcc.Graph(id="gene-boxplot", config={"displayModeBar": True, "displaylogo": False}),
                                ],
                            ),
                        ],
                    ),

                    # Tab 3: Heatmap
                    dbc.Tab(
                        label="Hierarchical Matrix",
                        tab_id="tab-heatmap",
                        children=[
                            html.Div(
                                className="plot-container",
                                children=[
                                    html.Div(
                                        className="plot-header-bar",
                                        children=[
                                            html.Div([
                                                html.Div("Unsupervised DEG Cluster Heatmap", className="plot-title"),
                                                html.Div("Z-score normalized expression matrix of top discriminating genomic features", className="plot-subtitle"),
                                            ]),
                                        ],
                                    ),
                                    dcc.Graph(
                                        id="heatmap-plot",
                                        config={"displayModeBar": True, "displaylogo": False},
                                        style={"height": "600px"},
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # Tab 4: Biomarker Ranking
                    dbc.Tab(
                        label="Machine Learning Consensus",
                        tab_id="tab-biomarkers",
                        children=[
                            html.Div(
                                className="plot-container",
                                children=[
                                    html.Div(
                                        className="plot-header-bar",
                                        children=[
                                            html.Div([
                                                html.Div("Consensus Biomarker Signature", className="plot-title"),
                                                html.Div("Validated genes confirmed through simultaneous statistical differential expression and Gini importance ranking.", className="plot-subtitle"),
                                            ]),
                                        ],
                                    ),
                                    dcc.Graph(id="biomarker-bar-chart", config={"displayModeBar": True, "displaylogo": False}),
                                ],
                            ),
                            html.Div(
                                className="two-column",
                                children=[
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("Random Forest Gini Importance", className="plot-title"),
                                            html.Div("Top ensemble feature weights across nested cross-validation iterations", className="plot-subtitle"),
                                            dcc.Graph(id="model-comparison-plot", config={"displayModeBar": True, "displaylogo": False}),
                                        ],
                                    ),
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("Statistical Power vs. Feature Weight", className="plot-title"),
                                            html.Div("Concordance between empirical hypothesis testing and machine learning classifiers", className="plot-subtitle"),
                                            dcc.Graph(id="de-vs-ml-scatter", config={"displayModeBar": True, "displaylogo": False}),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # Tab 5: Pathway Enrichment
                    dbc.Tab(
                        label="Biological Pathways",
                        tab_id="tab-pathways",
                        children=[
                            html.Div(
                                className="two-column",
                                children=[
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("Gene Ontology (GO) Biological Processes", className="plot-title"),
                                            html.Div("Statistically over-represented functional gene sets (Enrichr / Fisher exact test)", className="plot-subtitle"),
                                            dcc.Graph(id="go-enrichment-plot", config={"displayModeBar": True, "displaylogo": False}),
                                        ],
                                    ),
                                    html.Div(
                                        className="plot-container",
                                        children=[
                                            html.Div("KEGG Pathway Enrichment", className="plot-title"),
                                            html.Div("Overrepresented signaling cascades and metabolic circuits", className="plot-subtitle"),
                                            dcc.Graph(id="kegg-enrichment-plot", config={"displayModeBar": True, "displaylogo": False}),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # Tab 6: Clinical Survival & Therapeutics
                    dbc.Tab(
                        label="Clinical Survival & Drugs",
                        tab_id="tab-survival",
                        children=[
                            html.Div(
                                className="control-panel",
                                children=[
                                    html.Div(
                                        className="two-column",
                                        children=[
                                            html.Div([
                                                html.Div("Stratification Biomarker", className="control-label"),
                                                dcc.Dropdown(
                                                    id="survival-gene-dropdown",
                                                    options=[],
                                                    value=None,
                                                    placeholder="Select target biomarker for Kaplan-Meier analysis...",
                                                    clearable=False,
                                                    className="dark-dropdown",
                                                ),
                                            ]),
                                            html.Div([
                                                html.Div("Clinical Endpoint", className="control-label"),
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
                                            ]),
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
                                            html.Div("Kaplan-Meier Survival Analysis", id="km-plot-title", className="plot-title"),
                                            html.Div(
                                                "Patient stratification into High vs Low biomarker expression with Log-Rank test",
                                                id="km-plot-subtitle",
                                                className="plot-subtitle",
                                            ),
                                            dcc.Graph(id="kaplan-meier-plot", config={"displayModeBar": True, "displaylogo": False}),
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
                                            html.Div("Cross-Cohort Generalization (Zero-Shot ROC)", id="ext-roc-title", className="plot-title"),
                                            html.Div(
                                                "External ROC Curve: Model evaluated on independent cohort",
                                                id="ext-roc-subtitle",
                                                className="plot-subtitle",
                                            ),
                                            dcc.Graph(id="external-roc-plot", config={"displayModeBar": True, "displaylogo": False}),
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

                    # Tab 7: Multi-Omics & Somatic Alterations
                    dbc.Tab(
                        label="Multi-Omics Landscape",
                        tab_id="tab-multiomics",
                        children=[
                            html.Div(
                                className="plot-container",
                                children=[
                                    html.Div("Dual-Omics Integration: Transcriptomics vs. Somatic Mutations", className="plot-title"),
                                    html.Div(
                                        "Comparing RNA expression fold change against DNA somatic mutation frequencies from TCGA / COSMIC. "
                                        "Resolves the 'Jammed Gas Pedal' paradox where mutation-driven kinases (EGFR, KRAS, BRAF) show flat RNA levels.",
                                        className="plot-subtitle",
                                    ),
                                    dcc.Graph(id="multiomics-scatter-plot", config={"displayModeBar": True, "displaylogo": False}, style={"height": "520px"}),
                                ],
                            ),
                            html.Div(
                                className="plot-container",
                                style={"marginTop": "20px"},
                                children=[
                                    html.Div("Genomic Hotspots & Precision Actionability Catalog", className="plot-title"),
                                    html.Div(
                                        "Recurrent oncogenic hotspots, chromosomal fusions, clinical tiers, and matched targeted therapies.",
                                        className="plot-subtitle",
                                    ),
                                    html.Div(id="multiomics-table-content"),
                                ],
                            ),
                        ],
                    ),

                    # Tab 8: AI Oncology Intelligence Studio
                    dbc.Tab(
                        label="AI Clinical Copilot",
                        tab_id="tab-ai-copilot",
                        children=[
                            html.Div(
                                className="ai-copilot-container my-3",
                                children=[
                                    html.Div(
                                        className="copilot-header-card",
                                        children=[
                                            dbc.Row([
                                                dbc.Col([
                                                    html.Div([
                                                        html.Span("Clinical Oncology Copilot", className="plot-title m-0"),
                                                        dbc.Badge("Grounding Engine Active", color="success", className="ms-2 px-2 py-1", style={"fontSize": "0.72rem"}),
                                                    ], className="d-flex align-items-center"),
                                                    html.P(
                                                        id="ai-copilot-header-desc",
                                                        className="text-muted m-0 small mt-1",
                                                        children="Autonomous clinical reasoning copilot grounded in live pipeline biomarkers, pathways, survival metrics, and targeted therapeutics.",
                                                    ),
                                                ], md=8),
                                                dbc.Col([
                                                    dbc.Button("Clear History", id="btn-clear-chat", size="sm", outline=True, color="secondary", className="cohort-pill-btn"),
                                                ], md=4, className="d-flex justify-content-md-end align-items-center mt-2 mt-md-0"),
                                            ]),
                                            html.Hr(style={"borderColor": "var(--border-subtle)", "margin": "14px 0 12px"}),
                                            html.Div("Quick Investigation Queries:", style={"fontSize": "0.78rem", "color": "var(--text-muted)", "marginBottom": "8px", "fontWeight": "500"}),
                                            html.Div(
                                                className="d-flex flex-wrap gap-2",
                                                children=[
                                                    dbc.Button("Executive Summary", id="btn-chip-summary", size="sm", className="chip-btn"),
                                                    dbc.Button("Targeted Drug Opportunities", id="btn-chip-drugs", size="sm", className="chip-btn"),
                                                    dbc.Button("Top Biomarker Interpretation", id="btn-chip-biomarkers", size="sm", className="chip-btn"),
                                                    dbc.Button("Survival Prognosis", id="btn-chip-survival", size="sm", className="chip-btn"),
                                                    dbc.Button("Wet-Lab Protocol", id="btn-chip-validation", size="sm", className="chip-btn"),
                                                ],
                                            ),
                                        ],
                                    ),

                                    # Chat history container with loading indicator
                                    dcc.Loading(
                                        id="chat-loading",
                                        type="dot",
                                        color=CYAN,
                                        children=html.Div(
                                            id="chat-history-container",
                                            className="chat-history-box",
                                            children=[
                                                html.Div(
                                                    className="chat-row-assistant",
                                                    children=[
                                                        html.Div(
                                                            className="chat-bubble-assistant",
                                                            children=[
                                                                html.H4("AI Oncology Copilot Initialized"),
                                                                html.P(
                                                                    "I am connected directly to your active discovery cohort and results data. "
                                                                    "I have real-time access to the differential expression tables, Random Forest ranking, "
                                                                    "Kaplan-Meier hazard ratios, pathway enrichments, and therapeutic actionability catalogs."
                                                                ),
                                                                html.P(
                                                                    "Select a prompt chip above or query any biomarker gene or biological pathway below."
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ),

                                    # Input group
                                    dbc.InputGroup([
                                        dbc.Input(
                                            id="chat-input-text",
                                            placeholder="Ask anything about the biomarkers, pathways, survival metrics, or drug mechanisms...",
                                            type="text",
                                            style={
                                                "backgroundColor": "var(--bg-surface-1)",
                                                "color": "var(--text-primary)",
                                                "border": "1px solid var(--border-medium)",
                                                "fontSize": "0.88rem",
                                                "padding": "10px 16px",
                                            },
                                        ),
                                        dbc.Button(
                                            "Send Query",
                                            id="btn-send-chat",
                                            color="primary",
                                            style={
                                                "background": "var(--accent-blue)",
                                                "border": "none",
                                                "fontWeight": "600",
                                                "color": "#06090e",
                                                "padding": "0 22px",
                                                "fontSize": "0.85rem",
                                            },
                                        ),
                                    ], className="mb-2"),

                                    dcc.Store(id="chat-history-store", data=[]),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # ── Modern Footer ────────────────────────────────
            html.Div(
                className="dashboard-footer",
                children=[
                    html.Span("Cancer Biomarker Discovery Platform • Powered by Python, Plotly Dash, and scikit-learn"),
                    html.Span(f"NCBI GEO Cohort {config.GEO_ACCESSION} • For Academic & Clinical Research"),
                ],
            ),
        ],
    )
