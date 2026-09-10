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
                        "Autonomous Discovery Pipeline • Standby: Select or Search a Cohort to Begin",
                        id="dashboard-subtitle",
                        className="subtitle",
                    ),
                ],
            ),

            dcc.Store(id="active-cohort-store", data=None),

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
                                                placeholder="Enter GEO Accession (e.g. GSE8671, GSE19804, GSE15852)...",
                                                value="",
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
                                            dbc.Button("🔄 Standby", id="btn-reset-cohort", size="sm", outline=True, color="secondary"),
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

            # ── Standby Welcome Banner ───────────────────────
            html.Div(id="welcome-banner"),

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
                                                options=[],
                                                value=None,
                                                placeholder="Select a prognostic biomarker target...",
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
                                            html.Div("Kaplan-Meier Survival Curves", id="km-plot-title", className="plot-title"),
                                            html.Div(
                                                "Patient stratification into High vs Low biomarker expression with Log-Rank test",
                                                id="km-plot-subtitle",
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
                                            html.Div("Cross-Cohort Generalization (Zero-Shot ROC)", id="ext-roc-title", className="plot-title"),
                                            html.Div(
                                                "External ROC Curve: Model evaluated on independent cohort",
                                                id="ext-roc-subtitle",
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

                    # Tab 7: Multi-Omics & Mutations
                    dbc.Tab(
                        label="🧬 Multi-Omics & Mutations",
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
                                    dcc.Graph(id="multiomics-scatter-plot", config={"displayModeBar": True}, style={"height": "520px"}),
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

                    # Tab 8: AI Oncologist Copilot
                    dbc.Tab(
                        label="🤖 AI Oncologist Copilot",
                        tab_id="tab-ai-copilot",
                        children=[
                            html.Div(
                                className="ai-copilot-container my-3",
                                children=[
                                    dbc.Card(
                                        className="p-3 mb-3",
                                        style={
                                            "backgroundColor": "rgba(20, 20, 50, 0.75)",
                                            "border": "1px solid rgba(0, 212, 255, 0.3)",
                                            "borderRadius": "14px",
                                            "boxShadow": "0 8px 32px 0 rgba(0, 0, 0, 0.37)",
                                        },
                                        children=[
                                            dbc.Row([
                                                dbc.Col([
                                                    html.H4("🧬 AI Oncology Research Assistant", className="m-0", style={"color": CYAN, "fontWeight": "700"}),
                                                    html.P(
                                                        id="ai-copilot-header-desc",
                                                        className="text-muted m-0 small mt-1",
                                                        children="Autonomous clinical reasoning copilot grounded in live pipeline biomarkers, pathways, survival metrics, and targeted therapeutics.",
                                                    ),
                                                ], md=8),
                                                dbc.Col([
                                                    dbc.Badge("NVIDIA Nemotron 3.5 (Free)", color="success", className="me-2 p-2"),
                                                    dbc.Button("🧹 Clear Conversation", id="btn-clear-chat", size="sm", outline=True, color="secondary"),
                                                ], md=4, className="d-flex justify-content-md-end align-items-center mt-2 mt-md-0"),
                                            ]),
                                            html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "14px 0 10px"}),
                                            html.Div("Quick Investigation Queries:", style={"fontSize": "0.82rem", "color": "#94a3b8", "marginBottom": "8px"}),
                                            html.Div(
                                                className="d-flex flex-wrap gap-2",
                                                children=[
                                                    dbc.Button("💡 Summarize Findings", id="btn-chip-summary", size="sm", outline=True, className="chip-btn"),
                                                    dbc.Button("💊 Targeted Drug Opportunities", id="btn-chip-drugs", size="sm", outline=True, className="chip-btn"),
                                                    dbc.Button("🧬 Interpret Top Biomarkers", id="btn-chip-biomarkers", size="sm", outline=True, className="chip-btn"),
                                                    dbc.Button("⏳ Clinical Survival Prognosis", id="btn-chip-survival", size="sm", outline=True, className="chip-btn"),
                                                    dbc.Button("🧪 Wet-Lab Validation Protocol", id="btn-chip-validation", size="sm", outline=True, className="chip-btn"),
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
                                            style={
                                                "height": "480px",
                                                "overflowY": "auto",
                                                "padding": "20px",
                                                "backgroundColor": "rgba(10, 10, 26, 0.75)",
                                                "borderRadius": "14px",
                                                "border": "1px solid rgba(255, 255, 255, 0.08)",
                                                "marginBottom": "16px",
                                                "display": "flex",
                                                "flexDirection": "column",
                                                "gap": "14px",
                                            },
                                            children=[
                                                html.Div(
                                                    className="chat-row-assistant",
                                                    children=[
                                                        html.Div(
                                                            className="chat-bubble-assistant",
                                                            children=[
                                                                html.H4("👋 Welcome to the AI Oncology Copilot"),
                                                                html.P(
                                                                    "I am your AI research partner, connected directly to this discovery run. "
                                                                    "I have access to the differentially expressed genes, Random Forest consensus biomarkers, "
                                                                    "survival hazard ratios, enriched signaling pathways, and targeted therapeutics."
                                                                ),
                                                                html.P(
                                                                    "Click any of the quick inquiry buttons above or ask your own question below!"
                                                                ),
                                                                html.Div("• Ask about specific genes, drug repurposing, or wet-lab experimental designs.", className="small text-muted"),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ),

                                    # Input form
                                    dbc.InputGroup([
                                        dbc.Input(
                                            id="chat-input-text",
                                            placeholder="Ask anything about the biomarkers, pathways, survival metrics, or drug mechanisms...",
                                            type="text",
                                            style={
                                                "backgroundColor": "rgba(20, 20, 45, 0.85)",
                                                "color": FONT_COLOR,
                                                "border": "1px solid rgba(0, 212, 255, 0.3)",
                                                "fontSize": "0.95rem",
                                                "padding": "12px 16px",
                                            },
                                        ),
                                        dbc.Button(
                                            "Ask Copilot 🚀",
                                            id="btn-send-chat",
                                            color="primary",
                                            style={
                                                "background": f"linear-gradient(135deg, {CYAN} 0%, {PURPLE} 100%)",
                                                "border": "none",
                                                "fontWeight": "600",
                                                "padding": "0 24px",
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
