"""Offline smoke checks for callback registration and survival/drug rendering.

These checks exercise the clinical callback, not every registered callback and
not authentic clinical validation. Synthetic fixtures are never saved as results.
"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath("."))
from dashboard.app import create_app
from src import survival_analysis
from src.drug_mapping import get_drug_details_for_gene


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def clinical_fixture():
    rng = np.random.default_rng(42)
    samples = [f"SYNTHETIC_{i}" for i in range(60)]
    expression = pd.DataFrame(rng.normal(size=(1, 60)), index=["EGFR"], columns=samples)
    clinical = pd.DataFrame({
        "SURV_DEATH": rng.uniform(.1, 10, 60),
        "DEATH": rng.integers(0, 2, 60),
        "SURV_RELAPSE": rng.uniform(.1, 8, 60),
        "RELAPSE": rng.integers(0, 2, 60),
    }, index=samples)
    return expression, clinical


def clinical_callback(app):
    callbacks = [entry["callback"] for target, entry in app.callback_map.items()
                 if "kaplan-meier-plot.figure" in target]
    assert len(callbacks) == 1
    return callbacks[0].__wrapped__


def test_callback_registration(app):
    assert len(app.callback_map) >= 18
    assert all(callable(entry["callback"]) for entry in app.callback_map.values())
    assert app.server.test_client().get("/_dash-layout").status_code == 200


@pytest.mark.parametrize("endpoint", ["overall", "relapse"])
def test_clinical_callback_success_with_explicit_fixture(app, clinical_fixture, endpoint):
    with patch.object(survival_analysis, "load_survival_cohort", return_value=clinical_fixture):
        result = survival_analysis.evaluate_biomarker_survival("EGFR", outcome=endpoint)
        assert "error" not in result, result
        figure, card = clinical_callback(app)("EGFR", endpoint, None)
    assert len(figure.data) == 2
    assert card is not None


def test_missing_survival_cohort_keeps_drug_callback_usable(app):
    with patch.object(survival_analysis, "load_survival_cohort", side_effect=RuntimeError("Authentic cohort unavailable")):
        figure, card = clinical_callback(app)("EGFR", "overall", None)
    assert not figure.data
    assert "unavailable" in str(figure.to_plotly_json()).lower()
    assert "Approved Drugs" in str(card)


def test_absent_endpoints_are_not_imputed(app, clinical_fixture):
    expression, clinical = clinical_fixture
    clinical = clinical.astype("float64")
    clinical.loc[:, ["SURV_DEATH", "DEATH"]] = np.nan
    with patch.object(survival_analysis, "load_survival_cohort", return_value=(expression, clinical)):
        result = survival_analysis.evaluate_biomarker_survival("EGFR")
        assert "Insufficient matched clinical samples" in result["error"]
        figure, card = clinical_callback(app)("EGFR", "overall", None)
    assert not figure.data
    assert "Approved Drugs" in str(card)


def test_drug_mapping():
    drug_info = get_drug_details_for_gene("EGFR")
    assert drug_info is not None
    assert drug_info["approved_drugs"]
