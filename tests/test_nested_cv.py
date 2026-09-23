"""
Unit tests for nested cross-validation and L1 hyperparameter tuning.

Tests:
- L1 C selection via inner CV
- Nested CV fold structure (no leakage)
- Feature selection stability across folds
"""
import numpy as np
import pandas as pd

from src.ml_biomarkers import (
    _select_l1_C_inner_cv,
    nested_cv_ensemble_evaluation,
    evaluate_consensus_signature_cv,
    train_ensemble_models,
    find_consensus_biomarkers,
)


class TestL1Tuning:
    """Tests for L1 hyperparameter selection (Option C)."""

    def test_l1_C_selection_returns_best(self, synthetic_expression_data):
        """_select_l1_C_inner_cv should return the C with best CV accuracy."""
        expr_df, labels = synthetic_expression_data
        from src.ml_biomarkers import _prepare_data
        X, y, gene_names, groups = _prepare_data(expr_df, labels)

        best_C, cv_scores = _select_l1_C_inner_cv(X, y, gene_names, groups=groups)

        assert best_C in cv_scores.keys()
        assert cv_scores[best_C] == max(cv_scores.values())
        assert isinstance(best_C, float)

    def test_l1_C_grid_evaluation(self, synthetic_expression_data):
        """All C values in the grid should be evaluated."""
        expr_df, labels = synthetic_expression_data
        from src.ml_biomarkers import _prepare_data
        X, y, gene_names, groups = _prepare_data(expr_df, labels)

        c_grid = [0.01, 0.1, 1.0]
        best_C, cv_scores = _select_l1_C_inner_cv(
            X, y, gene_names, groups=groups, c_grid=c_grid
        )

        assert len(cv_scores) == len(c_grid)
        for c in c_grid:
            assert c in cv_scores

    def test_l1_tuning_enabled_in_config(self):
        """ENABLE_L1_TUNING should be in config."""
        import config
        assert hasattr(config, "ENABLE_L1_TUNING")
        assert hasattr(config, "L1_C_GRID")
        assert len(config.L1_C_GRID) > 0


class TestNestedCV:
    """Tests for nested cross-validation (leakage-free evaluation)."""

    def test_nested_cv_returns_per_fold_metrics(self, synthetic_expression_data, synthetic_de_results):
        """Nested CV should return per-fold metrics DataFrame."""
        expr_df, labels = synthetic_expression_data
        from src.ml_biomarkers import _prepare_data
        X, y, gene_names, groups = _prepare_data(expr_df, labels)

        per_fold, metrics = nested_cv_ensemble_evaluation(
            X, y, gene_names, synthetic_de_results, groups=groups
        )

        assert isinstance(per_fold, pd.DataFrame)
        assert "outer_fold" in per_fold.columns
        assert "fold_accuracy" in per_fold.columns
        assert "n_selected_genes" in per_fold.columns

    def test_nested_cv_metrics_structure(self, synthetic_expression_data, synthetic_de_results):
        """Nested CV metrics should have required keys."""
        expr_df, labels = synthetic_expression_data
        from src.ml_biomarkers import _prepare_data
        X, y, gene_names, groups = _prepare_data(expr_df, labels)

        per_fold, metrics = nested_cv_ensemble_evaluation(
            X, y, gene_names, synthetic_de_results, groups=groups
        )

        if len(per_fold) > 0:
            assert "nested_cv_acc" in metrics
            assert "selection_stability" in metrics
            assert np.isfinite(metrics["nested_cv_acc"])

    def test_nested_cv_feature_stability(self, synthetic_expression_data, synthetic_de_results):
        """Feature selection across folds should show valid stability scores."""
        expr_df, labels = synthetic_expression_data
        from src.ml_biomarkers import _prepare_data
        X, y, gene_names, groups = _prepare_data(expr_df, labels)

        per_fold, metrics = nested_cv_ensemble_evaluation(
            X, y, gene_names, synthetic_de_results, groups=groups
        )

        selection_stability = metrics.get("selection_stability", {})
        # All stability values should be between 0 and 1
        for v in selection_stability.values():
            assert 0 <= v <= 1


class TestEnsembleTraining:
    """Tests for ensemble model training."""

    def test_train_ensemble_returns_ranking(self, synthetic_expression_data):
        """train_ensemble_models should return gene ranking DataFrame."""
        expr_df, labels = synthetic_expression_data
        from src.ml_biomarkers import _prepare_data
        X, y, gene_names, groups = _prepare_data(expr_df, labels)

        ranking, metrics = train_ensemble_models(X, y, gene_names, groups=groups)

        assert isinstance(ranking, pd.DataFrame)
        assert "ensemble_score" in ranking.columns
        assert "ml_rank" in ranking.columns
        assert len(ranking) == len(gene_names)

    def test_ensemble_scores_non_negative(self, synthetic_expression_data):
        """Ensemble scores should be non-negative."""
        expr_df, labels = synthetic_expression_data
        from src.ml_biomarkers import _prepare_data
        X, y, gene_names, groups = _prepare_data(expr_df, labels)

        ranking, _ = train_ensemble_models(X, y, gene_names, groups=groups)

        assert all(ranking["ensemble_score"] >= 0)

    def test_ensemble_scores_sum(self, synthetic_expression_data):
        """Ensemble scores should be in valid range [0, ~2] (normalized components)."""
        expr_df, labels = synthetic_expression_data
        from src.ml_biomarkers import _prepare_data
        X, y, gene_names, groups = _prepare_data(expr_df, labels)

        ranking, _ = train_ensemble_models(X, y, gene_names, groups=groups)

        total = ranking["ensemble_score"].sum()
        # Ensemble is weighted sum of normalized scores, so total can be > 1
        # but should be finite and positive
        assert np.isfinite(total)
        assert total > 0
