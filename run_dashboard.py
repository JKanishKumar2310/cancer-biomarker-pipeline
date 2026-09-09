"""
Run Dashboard — Launch the interactive Biomarker Discovery Dashboard.

Prerequisites: Run `python run_analysis.py` first to generate results.

Usage:
    python run_dashboard.py
"""
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.utils import logger


def main():
    # Check if results exist
    required_files = ["de_results.csv", "ml_ranking_top100.csv", "consensus_biomarkers.csv"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(config.RESULTS_DIR, f))]

    if missing:
        logger.error("Analysis results not found! Run the pipeline first:")
        logger.error("  python run_analysis.py")
        logger.error(f"Missing files: {missing}")
        sys.exit(1)

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  CANCER BIOMARKER DISCOVERY DASHBOARD                    ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")
    logger.info(f"  Starting dashboard server...")
    logger.info(f"  Open your browser at: http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
    logger.info(f"  Press Ctrl+C to stop")
    logger.info("")

    from dashboard.app import create_app

    app = create_app()
    app.run(
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        debug=config.DASHBOARD_DEBUG,
    )


if __name__ == "__main__":
    main()
