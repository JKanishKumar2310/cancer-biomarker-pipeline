"""
Dashboard Callbacks — All interactive logic for the Dash app.

Connects user interactions (clicks, sliders) to plot updates.
"""
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, html, no_update, ctx
import dash_bootstrap_components as dbc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from dashboard.layout import (
    PLOT_LAYOUT_DEFAULTS, CYAN, PURPLE, PINK, GREEN, ORANGE, RED,
    FONT_COLOR, GRID_COLOR, make_stat_card,
)


def load_results():
    """Load all pre-computed analysis results from CSV files."""
    results = {}
    files = {
        "de_results": "de_results.csv",
        "ml_ranking": "ml_ranking_top100.csv",
        "consensus": "consensus_biomarkers.csv",
        "enrichment": "pathway_enrichment.csv",
        "survival": "survival_validation.csv",
        "drugs": "drug_actionability.csv",
        "ext_metrics": "external_validation_metrics.csv",
        "ext_roc": "external_roc_curve.csv",
    }
    for key, filename in files.items():
        filepath = os.path.join(config.RESULTS_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if "gene.1" in df.columns:
                df = df.drop(columns=["gene.1"])
            if "gene" in df.columns:
                df.index = df["gene"]
            results[key] = df
        else:
            results[key] = pd.DataFrame()

    # Load expression data and labels
    expr_path = os.path.join(config.DATA_DIR, "expression_matrix.csv")
    labels_path = os.path.join(config.DATA_DIR, "sample_labels.csv")
    if os.path.exists(expr_path) and os.path.exists(labels_path):
        results["expression"] = pd.read_csv(expr_path, index_col=0)
        results["labels"] = pd.read_csv(labels_path, index_col=0).squeeze()
    else:
        results["expression"] = pd.DataFrame()
        results["labels"] = pd.Series(dtype=str)

    return results


# Load data once at import
RESULTS = load_results()


def register_callbacks(app):
    """Register all Dash callbacks with the app."""

    # ── Stats Row ────────────────────────────────────────────
    @app.callback(
        Output("stats-row", "children"),
        Input("main-tabs", "active_tab"),
    )
    def update_stats(_tab):
        de = RESULTS.get("de_results", pd.DataFrame())
        consensus = RESULTS.get("consensus", pd.DataFrame())
        expr = RESULTS.get("expression", pd.DataFrame())

        total_genes = len(de) if len(de) > 0 else expr.shape[0]
        n_up = len(de[de["regulation"] == "Upregulated"]) if len(de) > 0 else 0
        n_down = len(de[de["regulation"] == "Downregulated"]) if len(de) > 0 else 0
        n_consensus = len(consensus) if len(consensus) > 0 else 0
        n_samples = expr.shape[1] if len(expr) > 0 else 0

        return [
            make_stat_card(f"{total_genes:,}", "Total Genes", "cyan", 1),
            make_stat_card(n_up, "Upregulated", "green", 2),
            make_stat_card(n_down, "Downregulated", "pink", 3),
            make_stat_card(n_consensus, "Consensus Biomarkers", "purple", 4),
            make_stat_card(n_samples, "Samples", "orange", 5),
        ]

    # ── PCA Plot ─────────────────────────────────────────────
    @app.callback(
        Output("pca-plot", "figure"),
        Input("main-tabs", "active_tab"),
    )
    def update_pca(tab):
        if tab != "tab-overview":
            return no_update

        expr = RESULTS.get("expression", pd.DataFrame())
        labels = RESULTS.get("labels", pd.Series(dtype=str))

        if len(expr) == 0:
            return _empty_figure("No expression data available")

        X = expr.T.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
        pcs = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
            "Condition": labels.values,
            "Sample": labels.index,
        })

        color_map = {"Tumor": PINK, "Normal": GREEN}

        fig = px.scatter(
            pca_df, x="PC1", y="PC2", color="Condition",
            hover_data=["Sample"],
            color_discrete_map=color_map,
            title="",
        )
        fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color="rgba(255,255,255,0.3)")))
        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)",
            height=420,
        )
        return fig

    # ── Distribution Plot ────────────────────────────────────
    @app.callback(
        Output("distribution-plot", "figure"),
        Input("main-tabs", "active_tab"),
    )
    def update_distribution(tab):
        if tab != "tab-overview":
            return no_update

        expr = RESULTS.get("expression", pd.DataFrame())
        labels = RESULTS.get("labels", pd.Series(dtype=str))

        if len(expr) == 0:
            return _empty_figure("No data available")

        # Sample a subset of genes for speed
        sample_genes = expr.sample(min(500, len(expr)), random_state=config.RANDOM_SEED)

        tumor_vals = sample_genes[labels[labels == "Tumor"].index].values.flatten()
        normal_vals = sample_genes[labels[labels == "Normal"].index].values.flatten()

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=tumor_vals, name="Tumor", marker_color=PINK,
            opacity=0.7, nbinsx=50,
        ))
        fig.add_trace(go.Histogram(
            x=normal_vals, name="Normal", marker_color=GREEN,
            opacity=0.7, nbinsx=50,
        ))
        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            barmode="overlay",
            xaxis_title="Expression Value (log2)",
            yaxis_title="Frequency",
            height=420,
        )
        return fig

    # ── Volcano Plot ─────────────────────────────────────────
    @app.callback(
        Output("volcano-plot", "figure"),
        [Input("fc-slider", "value"), Input("pval-slider", "value")],
    )
    def update_volcano(fc_thresh, pval_thresh):
        de = RESULTS.get("de_results", pd.DataFrame())

        if len(de) == 0:
            return _empty_figure("Run analysis first")

        # Recategorize with new thresholds
        de = de.copy()
        sig_mask = de["neg_log10_pval"] > pval_thresh
        de["display_reg"] = "Not Significant"
        de.loc[sig_mask & (de["log2FC"] > fc_thresh), "display_reg"] = "Upregulated"
        de.loc[sig_mask & (de["log2FC"] < -fc_thresh), "display_reg"] = "Downregulated"

        color_map = {"Upregulated": GREEN, "Downregulated": RED, "Not Significant": "rgba(100,100,140,0.3)"}

        fig = px.scatter(
            de, x="log2FC", y="neg_log10_pval",
            color="display_reg",
            hover_data={"gene": True, "log2FC": ":.2f", "adj_pvalue": ":.2e", "display_reg": False, "neg_log10_pval": False},
            color_discrete_map=color_map,
            custom_data=["gene"],
        )

        # Add threshold lines
        fig.add_hline(y=pval_thresh, line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1)
        fig.add_vline(x=fc_thresh, line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1)
        fig.add_vline(x=-fc_thresh, line_dash="dash", line_color="rgba(255,255,255,0.2)", line_width=1)

        fig.update_traces(marker=dict(size=5, opacity=0.75))
        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            xaxis_title="log₂ Fold Change",
            yaxis_title="-log₁₀(adjusted p-value)",
            height=500,
            legend_title_text="Regulation",
        )
        return fig

    # ── Gene Detail (click on volcano) ───────────────────────
    @app.callback(
        [Output("gene-detail-container", "style"),
         Output("gene-detail-title", "children"),
         Output("gene-boxplot", "figure")],
        Input("volcano-plot", "clickData"),
    )
    def update_gene_detail(click_data):
        if click_data is None:
            return {"display": "none"}, "", go.Figure()

        gene = click_data["points"][0]["customdata"][0]
        expr = RESULTS.get("expression", pd.DataFrame())
        labels = RESULTS.get("labels", pd.Series(dtype=str))
        de = RESULTS.get("de_results", pd.DataFrame())

        if gene not in expr.index:
            return {"display": "none"}, "", go.Figure()

        # Gene info
        gene_info = ""
        if gene in de.index:
            row = de.loc[gene]
            gene_info = f" — log2FC: {row['log2FC']:.2f}, adj p: {row['adj_pvalue']:.2e}, {row['regulation']}"

        # Expression boxplot
        gene_expr = expr.loc[gene]
        box_df = pd.DataFrame({
            "Expression": gene_expr.values,
            "Condition": labels[gene_expr.index].values,
        })

        fig = go.Figure()
        for cond, color in [("Normal", GREEN), ("Tumor", PINK)]:
            subset = box_df[box_df["Condition"] == cond]
            fig.add_trace(go.Box(
                y=subset["Expression"], name=cond,
                marker_color=color, boxmean="sd",
                jitter=0.3, pointpos=-1.5,
            ))

        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            yaxis_title="Expression (log2)",
            height=350,
            showlegend=False,
        )

        return {"display": "block"}, f"Gene: {gene}{gene_info}", fig

    # ── Heatmap ──────────────────────────────────────────────
    @app.callback(
        Output("heatmap-plot", "figure"),
        Input("main-tabs", "active_tab"),
    )
    def update_heatmap(tab):
        if tab != "tab-heatmap":
            return no_update

        de = RESULTS.get("de_results", pd.DataFrame())
        expr = RESULTS.get("expression", pd.DataFrame())
        labels = RESULTS.get("labels", pd.Series(dtype=str))

        if len(de) == 0 or len(expr) == 0:
            return _empty_figure("No data available")

        # Get top DEGs
        sig = de[de["regulation"] != "Not Significant"].head(config.TOP_DE_GENES)
        top_genes = sig.index.tolist()

        # Filter genes present in expression matrix
        top_genes = [g for g in top_genes if g in expr.index]
        if len(top_genes) == 0:
            return _empty_figure("No matching genes found")

        heatmap_data = expr.loc[top_genes]

        # Sort samples by condition
        sorted_samples = labels.sort_values().index
        sorted_samples = [s for s in sorted_samples if s in heatmap_data.columns]
        heatmap_data = heatmap_data[sorted_samples]

        # Z-score normalize across samples for visualization
        heatmap_z = heatmap_data.sub(heatmap_data.mean(axis=1), axis=0).div(
            heatmap_data.std(axis=1) + 1e-10, axis=0
        )

        # Color bar for sample labels
        sample_colors = [PINK if labels[s] == "Tumor" else GREEN for s in sorted_samples]

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_z.values,
            x=sorted_samples,
            y=top_genes,
            colorscale=[
                [0, "#2563eb"],
                [0.25, "#1e40af"],
                [0.5, "#0a0a1a"],
                [0.75, "#b91c1c"],
                [1, "#ef4444"],
            ],
            colorbar=dict(title="Z-score", titleside="right"),
            hovertemplate="Gene: %{y}<br>Sample: %{x}<br>Z-score: %{z:.2f}<extra></extra>",
        ))

        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            height=600,
            xaxis=dict(
                tickfont=dict(size=7), tickangle=90,
                gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
            ),
            yaxis=dict(
                tickfont=dict(size=9),
                gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
            ),
        )
        return fig

    # ── Biomarker Bar Chart ──────────────────────────────────
    @app.callback(
        Output("biomarker-bar-chart", "figure"),
        Input("main-tabs", "active_tab"),
    )
    def update_biomarker_chart(tab):
        if tab != "tab-biomarkers":
            return no_update

        consensus = RESULTS.get("consensus", pd.DataFrame())
        if len(consensus) == 0:
            return _empty_figure("No consensus biomarkers found")

        top = consensus.head(20).sort_values("ensemble_score", ascending=True)

        colors = [GREEN if r == "Upregulated" else RED for r in top["regulation"]]

        fig = go.Figure(go.Bar(
            x=top["ensemble_score"],
            y=top["gene"],
            orientation="h",
            marker_color=colors,
            hovertemplate=(
                "Gene: %{y}<br>"
                "Ensemble Score: %{x:.4f}<br>"
                "<extra></extra>"
            ),
        ))

        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            xaxis_title="Ensemble ML Score",
            yaxis_title="",
            height=500,
        )
        return fig

    # ── Model Comparison ─────────────────────────────────────
    @app.callback(
        Output("model-comparison-plot", "figure"),
        Input("main-tabs", "active_tab"),
    )
    def update_model_comparison(tab):
        if tab != "tab-biomarkers":
            return no_update

        ml = RESULTS.get("ml_ranking", pd.DataFrame())
        if len(ml) == 0:
            return _empty_figure("No ML results")

        # Show top 10 genes
        top10 = ml.head(10)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top10["gene"], y=top10["RF_importance"],
            name="Random Forest", marker_color=GREEN, opacity=0.85,
        ))

        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            barmode="group",
            xaxis_title="Gene",
            yaxis_title="Normalized Importance",
            height=380,
        )
        return fig

    # ── DE vs ML Scatter ─────────────────────────────────────
    @app.callback(
        Output("de-vs-ml-scatter", "figure"),
        Input("main-tabs", "active_tab"),
    )
    def update_de_ml_scatter(tab):
        if tab != "tab-biomarkers":
            return no_update

        de = RESULTS.get("de_results", pd.DataFrame())
        ml = RESULTS.get("ml_ranking", pd.DataFrame())

        if len(de) == 0 or len(ml) == 0:
            return _empty_figure("No data")

        # Merge on gene
        common = de.index.intersection(ml.index)
        if len(common) == 0:
            return _empty_figure("No overlapping genes")

        merged = pd.DataFrame({
            "gene": common,
            "neg_log10_pval": de.loc[common, "neg_log10_pval"].values,
            "ensemble_score": ml.loc[common, "ensemble_score"].values,
            "regulation": de.loc[common, "regulation"].values,
        })

        color_map = {"Upregulated": GREEN, "Downregulated": RED, "Not Significant": "rgba(100,100,140,0.2)"}

        fig = px.scatter(
            merged, x="neg_log10_pval", y="ensemble_score",
            color="regulation", hover_data=["gene"],
            color_discrete_map=color_map,
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            xaxis_title="-log₁₀(adj. p-value) [Statistical]",
            yaxis_title="Ensemble ML Score",
            height=380,
        )
        return fig

    # ── Pathway Enrichment Plots ─────────────────────────────
    @app.callback(
        [Output("go-enrichment-plot", "figure"),
         Output("kegg-enrichment-plot", "figure")],
        Input("main-tabs", "active_tab"),
    )
    def update_pathway_plots(tab):
        if tab != "tab-pathways":
            return no_update, no_update

        enr = RESULTS.get("enrichment", pd.DataFrame())
        if len(enr) == 0:
            empty = _empty_figure("No enrichment data")
            return empty, empty

        def make_enrichment_fig(data, color, title_prefix):
            if len(data) == 0:
                return _empty_figure(f"No {title_prefix} results")

            data = data.head(config.ENRICHMENT_TOP_TERMS).sort_values("Adjusted P-value", ascending=True)
            data = data.iloc[::-1]  # Reverse for horizontal bar

            # Truncate long term names
            data["Term_short"] = data["Term"].str[:50]

            fig = go.Figure(go.Bar(
                x=-np.log10(data["Adjusted P-value"].clip(lower=1e-20)),
                y=data["Term_short"],
                orientation="h",
                marker=dict(
                    color=-np.log10(data["Adjusted P-value"].clip(lower=1e-20)),
                    colorscale=[[0, "rgba(100,100,140,0.3)"], [1, color]],
                ),
                hovertemplate=(
                    "Term: %{y}<br>"
                    "-log₁₀(adj. p): %{x:.2f}<br>"
                    "<extra></extra>"
                ),
            ))
            fig.update_layout(
                **PLOT_LAYOUT_DEFAULTS,
                xaxis_title="-log₁₀(adjusted p-value)",
                yaxis_title="",
                height=450,
                yaxis=dict(tickfont=dict(size=9), gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            )
            return fig

        # Split by gene set
        go_data = enr[enr["Gene_set"].str.contains("GO", case=False, na=False)]
        kegg_data = enr[enr["Gene_set"].str.contains("KEGG", case=False, na=False)]

        go_fig = make_enrichment_fig(go_data, CYAN, "GO")
        kegg_fig = make_enrichment_fig(kegg_data, PURPLE, "KEGG")

        # ── Tab 6: Clinical Survival & Drugs Callbacks ──────
        @app.callback(
            [
                Output("kaplan-meier-plot", "figure"),
                Output("drug-actionability-content", "children"),
            ],
            [
                Input("survival-gene-dropdown", "value"),
                Input("survival-endpoint-radio", "value"),
            ],
        )
        def update_survival_and_drugs(gene, endpoint):
            from src.survival_analysis import evaluate_biomarker_survival
            from src.drug_mapping import get_drug_details_for_gene

            if not gene:
                return _empty_figure("Select a biomarker"), html.Div("No gene selected")

            # 1. Kaplan-Meier Curve
            res = evaluate_biomarker_survival(gene, outcome=endpoint)
            if "error" in res:
                km_fig = _empty_figure(res["error"])
            else:
                km_fig = go.Figure()
                # High Expression Step Curve
                km_fig.add_trace(go.Scatter(
                    x=res["km_high"]["timeline"],
                    y=res["km_high"]["survival"],
                    mode="lines",
                    line=dict(color=PINK, width=2.5, shape="hv"),
                    name=f"High {gene} (n={res['n_high']})",
                    hovertemplate="Time: %{x:.1f} yrs<br>Survival: %{y:.1%}<extra></extra>",
                ))
                # Low Expression Step Curve
                km_fig.add_trace(go.Scatter(
                    x=res["km_low"]["timeline"],
                    y=res["km_low"]["survival"],
                    mode="lines",
                    line=dict(color=CYAN, width=2.5, shape="hv"),
                    name=f"Low {gene} (n={res['n_low']})",
                    hovertemplate="Time: %{x:.1f} yrs<br>Survival: %{y:.1%}<extra></extra>",
                ))

                endpoint_title = "Overall Survival (OS)" if endpoint == "overall" else "Relapse-Free Survival (RFS)"
                km_fig.update_layout(
                    **PLOT_LAYOUT_DEFAULTS,
                    title=dict(
                        text=f"<b>{gene}</b> — {endpoint_title}<br><sup>Log-Rank p = {res['p_value']:.2e} | Hazard Ratio (HR) = {res['hazard_ratio']:.2f} | Cutoff: {res['median_cutoff']:.2f}</sup>",
                        font=dict(size=13, color=FONT_COLOR),
                    ),
                    xaxis_title="Time (Years)",
                    yaxis_title="Probability of Survival",
                    yaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor=GRID_COLOR),
                    xaxis=dict(gridcolor=GRID_COLOR),
                    height=420,
                    legend=dict(x=0.02, y=0.05, bgcolor="rgba(10,10,26,0.6)"),
                )

            # 2. Targeted Drug Card
            drug_info = get_drug_details_for_gene(gene)
            if drug_info:
                drug_card = html.Div(
                    className="biomarker-card",
                    children=[
                        html.Div(
                            className="d-flex justify-content-between align-items-center mb-2",
                            children=[
                                html.H4(f"🎯 {gene}", className="m-0 text-info"),
                                dbc.Badge(drug_info["evidence_tier"].split(":")[0], color="success", className="p-2"),
                            ],
                        ),
                        html.P(html.B(drug_info["gene_name"]), className="text-light mb-2"),
                        html.Div(className="divider-line my-2"),
                        html.P([html.B("Approved Drugs: "), html.Span(drug_info["approved_drugs"], className="text-warning")]),
                        html.P([html.B("Drug Class: "), html.Span(drug_info["drug_class"])]),
                        html.P([html.B("Mechanism: "), html.Span(drug_info["mechanism"])]),
                        html.P([html.B("Clinical Indication: "), html.Span(drug_info["indication"], className="text-muted")]),
                        html.Div(className="divider-line my-2"),
                        html.P([html.B("Actionability: "), html.Span(drug_info["clinical_action"], className="text-info")]),
                    ],
                )
            else:
                drug_card = html.Div(
                    className="biomarker-card text-center p-4",
                    children=[
                        html.H5(f"ℹ️ {gene}", className="text-muted"),
                        html.P("No direct FDA-approved targeted drug currently mapped for this gene.", className="text-muted"),
                        html.P("Explore clinical trials or upstream kinase pathway inhibitors.", className="small text-info"),
                    ],
                )

            return km_fig, drug_card

        @app.callback(
            [
                Output("external-roc-plot", "figure"),
                Output("external-metrics-content", "children"),
            ],
            [Input("main-tabs", "active_tab")],
        )
        def update_external_validation(active_tab):
            ext_m = RESULTS.get("ext_metrics", pd.DataFrame())
            ext_r = RESULTS.get("ext_roc", pd.DataFrame())

            if len(ext_r) == 0 or len(ext_m) == 0:
                return _empty_figure("External validation results not loaded"), html.Div("Metrics unavailable")

            # ROC Curve
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(
                x=ext_r["fpr"],
                y=ext_r["tpr"],
                mode="lines",
                line=dict(color=GREEN, width=3),
                name=f"ROC (AUC = {ext_m['roc_auc'].iloc[0]:.4f})",
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            ))
            # Diagonal chance line
            roc_fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
                name="Chance (AUC = 0.50)",
            ))
            roc_fig.update_layout(
                **PLOT_LAYOUT_DEFAULTS,
                title=dict(
                    text=f"<b>Cross-Cohort ROC Curve</b> (AUC = {ext_m['roc_auc'].iloc[0]:.4f})<br><sup>Trained: GSE15852 (Malaysia) | Tested: GSE42568 (Europe)</sup>",
                    font=dict(size=12, color=FONT_COLOR),
                ),
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Sensitivity)",
                xaxis=dict(range=[-0.02, 1.02], gridcolor=GRID_COLOR),
                yaxis=dict(range=[-0.02, 1.02], gridcolor=GRID_COLOR),
                height=350,
                legend=dict(x=0.55, y=0.15, bgcolor="rgba(10,10,26,0.6)"),
            )

            # Metrics cards
            m = ext_m.iloc[0]
            metrics_content = html.Div(
                className="stats-row",
                style={"gridTemplateColumns": "repeat(2, 1fr)", "gap": "12px", "marginTop": "10px"},
                children=[
                    make_stat_card(f"{m['test_accuracy']*100:.1f}%", "External Test Accuracy", "accent-cyan", 0),
                    make_stat_card(f"{m['roc_auc']:.4f}", "Cross-Cohort ROC-AUC", "accent-green", 1),
                    make_stat_card(f"{m['sensitivity']*100:.1f}%", "Sensitivity (Tumor Recall)", "accent-pink", 2),
                    make_stat_card(f"{m['specificity']*100:.1f}%", "Specificity (Normal Rule-Out)", "accent-purple", 3),
                ],
            )

            return roc_fig, metrics_content

    # ── Quick Select Cohort Buttons ──────────────────────────
    @app.callback(
        Output("geo-accession-input", "value"),
        [Input("btn-quick-gse8671", "n_clicks"),
         Input("btn-quick-gse19804", "n_clicks"),
         Input("btn-quick-gse15852", "n_clicks")],
        prevent_initial_call=True,
    )
    def update_geo_input(c1, c2, c3):
        if not ctx.triggered_id:
            return no_update
        mapping = {
            "btn-quick-gse8671": "GSE8671",
            "btn-quick-gse19804": "GSE19804",
            "btn-quick-gse15852": "GSE15852",
        }
        return mapping.get(ctx.triggered_id, no_update)

    # ── AI Ingestion & Analysis Runner ───────────────────────
    @app.callback(
        [Output("geo-status-output", "children"),
         Output("dashboard-subtitle", "children")],
        Input("btn-run-geo", "n_clicks"),
        State("geo-accession-input", "value"),
        prevent_initial_call=True,
    )
    def run_ai_curation_and_analysis(n_clicks, accession):
        if not n_clicks or not accession:
            return no_update, no_update

        accession = accession.strip().upper()
        from src.ai_geo_curator import curate_dataset
        import run_analysis

        # Step 1: Curate
        curation = curate_dataset(accession)
        if not curation.get("success"):
            err_msg = curation.get("error", "Unknown curation error")
            return dbc.Alert(f"❌ AI Curation Failed: {err_msg}", color="danger", dismissable=True), no_update

        # Step 2: Configure
        config.GEO_ACCESSION = accession
        config.CANCER_TYPE = curation.get("cancer_type", f"Cancer ({accession})")

        # Clear discovery cache
        for f in ["expression_matrix.csv", "sample_labels.csv"]:
            p = os.path.join(config.DATA_DIR, f)
            if os.path.exists(p):
                os.remove(p)

        # Step 3: Run pipeline
        try:
            run_analysis.main()
            # Step 4: Reload RESULTS in memory
            RESULTS.clear()
            RESULTS.update(load_results())
        except Exception as e:
            return dbc.Alert(f"❌ Pipeline error after curation: {e}", color="danger", dismissable=True), no_update

        status_alert = dbc.Alert([
            html.B("✅ AI Curation & Analysis Complete! "),
            f"Dataset {accession} ({curation['cancer_type']}) • ",
            f"Platform: {curation['platform']} • ",
            f"Classified: {curation['n_tumor']} Tumor vs {curation['n_normal']} Normal samples. ",
            html.Span(f"({curation['rationale']})", style={"fontStyle": "italic", "color": "#cbd5e1"}),
            html.Div("Dashboard re-indexed. Switch tabs to explore new findings!", className="mt-1", style={"fontWeight": "600"}),
        ], color="success", dismissable=True)

        new_subtitle = f"Autonomous Discovery Pipeline • Current Dataset: {accession} ({config.CANCER_TYPE})"
        return status_alert, new_subtitle


def _empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=FONT_COLOR),
    )
    fig.update_layout(
        paper_bgcolor="rgba(10,10,26,0)",
        plot_bgcolor="rgba(10,10,26,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=300,
    )
    return fig
