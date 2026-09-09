"""
Dashboard App — Entry point for the Plotly Dash web application.
"""
import dash
import dash_bootstrap_components as dbc

from dashboard.layout import create_layout
from dashboard.callbacks import register_callbacks


def create_app():
    """Create and configure the Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        assets_folder="assets",
        title="Cancer Biomarker Discovery Dashboard",
        update_title="Loading...",
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
            {"name": "description", "content": "AI-powered cancer biomarker discovery dashboard using differential expression and machine learning."},
        ],
    )

    app.layout = create_layout()
    register_callbacks(app)

    return app


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config

    app = create_app()
    app.run(
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        debug=config.DASHBOARD_DEBUG,
    )
