"""
Run Analysis — Full computational pipeline orchestrator.

Executes all analysis steps in sequence:
1. Load data (GEO or synthetic)
2. Preprocess (normalize, filter)
3. Differential expression analysis
4. ML-based biomarker ranking
5. Pathway enrichment analysis
6. Save all results

Usage:
    python run_analysis.py
"""
import os
import sys
import time
import math
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.utils import logger
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.differential_expression import run_differential_expression, get_top_degs
from src.ml_biomarkers import run_ml_biomarker_ranking
from src.pathway_analysis import run_enrichment
from src.ai_annotator import annotate_biomarkers, generate_summary_report, fetch_cohorts_for_cancer
from src.survival_analysis import run_survival_pipeline
from src.external_validation import run_external_validation
from src.drug_mapping import map_biomarkers_to_drugs
from src.genomic_annotator import integrate_multi_omics


def main(force_synthetic: bool = False):
    """Run the complete biomarker discovery pipeline."""
    start_time = time.time()

    # Test-mode output isolation: synthetic smoke-test runs must never
    # write into the authentic results directory.
    if force_synthetic:
        test_results_dir = os.path.join(config.BASE_DIR, "results", "synthetic_test")
        os.makedirs(test_results_dir, exist_ok=True)
        config.RESULTS_DIR = test_results_dir
        logger.warning(
            f"TEST MODE: redirecting all outputs to isolated directory -> {test_results_dir}"
        )

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  CANCER BIOMARKER DISCOVERY & TRANSLATIONAL PIPELINE     ║")
    logger.info("║  Multi-Cohort Discovery, Validation & Survival Analysis  ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")

    # ── Step 1: Load Discovery Data ──────────────────────────
    logger.info(f"STEP 1/8: Loading discovery cohort data ({config.GEO_ACCESSION})...")
    expr_df, labels = load_data(force_synthetic=force_synthetic)
    logger.info("")

    # ── Step 2: Preprocess ───────────────────────────────────
    logger.info("STEP 2/8: Preprocessing & quality filtering...")
    expr_clean, labels_clean = preprocess(expr_df, labels)
    logger.info("")

    # ── Step 3: Differential Expression ──────────────────────
    logger.info("STEP 3/8: Differential expression analysis...")
    de_results = run_differential_expression(expr_clean, labels_clean)
    top_degs = get_top_degs(de_results)
    logger.info("")

    # ── Step 4: ML Biomarker Ranking ─────────────────────────
    logger.info("STEP 4/8: ML-based biomarker ranking (Random Forest)...")
    ml_ranking, consensus = run_ml_biomarker_ranking(
        expr_clean, labels_clean, de_results
    )
    logger.info("")

    # ── Step 5: Pathway Enrichment ───────────────────────────
    logger.info("STEP 5/8: Pathway enrichment analysis (GO + KEGG)...")
    if len(consensus) > 0:
        enrichment_genes = consensus["gene"].tolist()
    else:
        enrichment_genes = top_degs["gene"].tolist()[:30]

    enrichment = run_enrichment(enrichment_genes)
    logger.info("")

    # ── Step 6 & 7: AI selects survival + validation cohorts ────────────────
    cohorts = fetch_cohorts_for_cancer(config.CANCER_TYPE)
    surv_info = cohorts["survival"]
    val_info  = cohorts["validation"]

    # Store on config so survival_analysis.py and external_validation.py can read them
    config.SURVIVAL_COHORT   = surv_info
    config.VALIDATION_COHORT = val_info

    # ── Step 6: Clinical Survival Analysis ───────────────────
    logger.info(f"STEP 6/8: Clinical survival analysis ({surv_info['label']})...")
    survival_df = run_survival_pipeline(force_synthetic=force_synthetic)
    logger.info("")

    # ── Step 7: Cross-Cohort External Validation ─────────────
    logger.info(f"STEP 7/8: Cross-cohort external validation ({val_info['label']})...")
    ext_val = run_external_validation(force_synthetic=force_synthetic)
    logger.info("")

    # ── Step 8: Drug Mapping, Multi-Omics & AI Annotation ─────
    logger.info("STEP 8/8: Targeted drug mapping, multi-omics & biological summary...")
    drugs_df = map_biomarkers_to_drugs(consensus)
    try:
        integrate_multi_omics(
            de_file=os.path.join(config.RESULTS_DIR, "de_results.csv"),
            cancer_type=config.CANCER_TYPE,
            output_file=os.path.join(config.RESULTS_DIR, "multi_omics_integration.csv"),
        )
    except Exception as e:
        logger.warning(f"Multi-omics integration skipped: {e}")
    annotated = annotate_biomarkers(consensus)
    summary = generate_summary_report(de_results, consensus, annotated)
    logger.info("")

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  TRANSLATIONAL PIPELINE COMPLETE                         ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"")
    logger.info(f"  ⏱  Time elapsed: {elapsed:.1f} seconds")
    logger.info(f"  📊 Total genes analyzed: {len(de_results):,}")
    logger.info(f"  ⬆️  Upregulated DEGs: {len(de_results[de_results['regulation'] == 'Upregulated'])}")
    logger.info(f"  ⬇️  Downregulated DEGs: {len(de_results[de_results['regulation'] == 'Downregulated'])}")
    logger.info(f"  🎯 Consensus biomarkers: {len(consensus)}")
    logger.info(f"  🧬 Enriched pathways: {len(enrichment)}")
    if np.isnan(ext_val.get("accuracy", np.nan)):
        logger.info(f"  🌐 External Cohort Validation: N/A ({ext_val.get('status', 'Zero Overlap')})")
    else:
        logger.info(f"  🌐 External Cohort Test Accuracy: {ext_val['accuracy']*100:.1f}% (ROC-AUC: {ext_val['roc_auc']:.4f})")
    logger.info(f"")
    logger.info(f"  Results saved to: {config.RESULTS_DIR}")
    logger.info(f"")
    logger.info(f"  Launch interactive dashboard with:")
    logger.info(f"    python run_dashboard.py")
    logger.info(f"")

    if len(consensus) > 0:
        logger.info("  Top Consensus Biomarkers:")
        for i, (_, row) in enumerate(consensus.head(8).iterrows()):
            marker = "⬆" if row.get("regulation") == "Upregulated" else "⬇"
            logger.info(
                f"    {i+1:2d}. {row['gene']:12s}  {marker}  "
                f"FC={row.get('log2FC', 0):+.2f}  "
                f"ML={row.get('ensemble_score', 0):.4f}"
            )


def check_accession(accession: str) -> int:
    """
    Dry-run triage of a single GEO accession without running the full pipeline.

    Downloads only the series matrix metadata and reports assay compatibility,
    platform, sample counts and detected group labels. Use this to vet candidate
    cancer datasets quickly before committing to an 8-stage run.

    Returns a process exit code (0 = usable, 1 = incompatible/unusable).
    """
    from src.ai_geo_curator import curate_dataset, fetch_geo_metadata
    accession = accession.strip().upper()
    logger.info("╔" + "═" * 58 + "╗")
    logger.info(f"║  DATASET COMPATIBILITY CHECK: {accession:<29}║")
    logger.info("╚" + "═" * 58 + "╝")

    meta = fetch_geo_metadata(accession)
    if not meta.get("success"):
        logger.error(f"✗ {accession} is NOT usable: {meta.get('error')}")
        return 1
    logger.info(f"  Title:          {meta.get('series_title', '(unknown)')}")
    logger.info(f"  Platform:       {meta.get('platform')}")
    logger.info(f"  Series types:   {meta.get('series_types')}")
    logger.info(f"  Samples found:  {len(meta.get('samples', {}))}")

    curation = curate_dataset(accession)
    if not curation.get("success"):
        logger.error(f"✗ {accession} could not be curated: {curation.get('error')}")
        return 1
    logger.info("")
    logger.info(f"  Cancer type:    {curation.get('cancer_type')}")
    logger.info(f"  Group counts:   {curation.get('counts')}")
    logger.info(f"  Rationale:      {curation.get('rationale')}")
    logger.info("")
    logger.info(f"✓ {accession} is USABLE by the pipeline.")
    logger.info(f"  To run the full pipeline on it, set GEO_ACCESSION='{accession}' in config.py")
    logger.info(f"  (and adjust CANCER_TYPE / COMPARISON_MODE if needed), then run: python run_analysis.py")
    return 0
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cancer Biomarker Discovery Pipeline")
    parser.add_argument(
        "--test-mode", "--synthetic",
        action="store_true",
        dest="test_mode",
        help="Run end-to-end smoke test on synthetic data without downloading large GEO datasets (for CI)",
    )
    parser.add_argument(
        "--check",
        metavar="ACCESSION",
        dest="check",
        default=None,
        help="Dry-run compatibility check of a GEO accession (e.g. --check GSE19804) without running the pipeline",
    )
    args = parser.parse_args()

    if args.check:
        sys.exit(check_accession(args.check))
    main(force_synthetic=args.test_mode)
