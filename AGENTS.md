# AGENTS.md — Project Configuration Guide

## Project Overview
Cancer Biomarker Discovery & Translational Pipeline — a multi-step computational pipeline for discovering, validating, and annotating cancer biomarkers from GEO microarray and RNA-seq datasets.

## Standard Development Commands

| Task              | Command                                        |
|-------------------|------------------------------------------------|
| Install deps      | `python -m pip install -r requirements.txt`    |
| Run pipeline      | `python run_analysis.py`                       |
| Run smoke test    | `python run_analysis.py --test-mode`           |
| Run tests         | `python -m pytest tests/ -q`                   |
| Run tests w/ cov  | `python -m pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=90` |
| Lint (if available) | `ruff check src/ tests/`                   |
| Typecheck (if available) | `mypy src/`                             |
| Build dashboard   | `python run_dashboard.py`                      |

## Configuration Flags
See `config.py` for the full list. Key feature flags:
- `DATA_TYPE` — "microarray" or "rnaseq"
- `DISCOVERY_COHORTS` — list of GEO accessions for multi-cohort analysis
- `ENABLE_RNASEQ` — enable RNA-seq count processing (voom/limra)
- `ENABLE_META_ANALYSIS` — enable multi-cohort fixed-effects meta-analysis
- `ENABLE_BATCH_CORRECTION` — enable ComBat batch correction
- `ENABLE_L1_TUNING` — enable L1 penalty hyperparameter tuning (inner CV)
- `RNASEQ_NORMALIZATION` — "tmm" (edgeR) or "median_of_ratios" (DESeq2)

## Testing
Target: **≥90% coverage** for statistical modules in `src/`.
Test files are in `tests/` with shared fixtures in `tests/conftest.py`.

## Dependency Pinning
- Direct dependencies: `requirements.in`
- Compiled with: `pip-compile --resolver=backtracking requirements.in -o requirements.txt`
- All versions are pinned (no floating dependencies)