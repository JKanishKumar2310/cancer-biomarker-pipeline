"""
Shared utility functions for the biomarker discovery pipeline.
"""
import os
import logging
import pandas as pd

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("biomarker")


