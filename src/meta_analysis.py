"""
Meta-Analysis Module — Fixed-effects inverse-variance weighted meta-analysis
across multiple GEO discovery cohorts.

Implements the standard fixed-effects meta-analysis model:
    β_meta = Σ(w_i * β_i) / Σ(w_i)
    w_i = 1 / SE_i²
    SE_meta = sqrt(1 / Σ(w_i))

This combines effect sizes (log2FC) from per-cohort DE analyses into a single
pooled estimate with reduced standard error, increasing statistical power.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger


def run_meta_analysis(
    per_cohort_de: list[pd.DataFrame],
    cohort_labels: list[str],
) -> pd.DataFrame:
    """
    Perform fixed-effects inverse-variance weighted meta-analysis.
    
    Parameters
    ----------
    per_cohort_de : list of pd.DataFrame
        List of DE result DataFrames, one per cohort. Each must have:
        'gene', 'log2FC', 'pvalue', 'adj_pvalue', 'mean_tumor', 'mean_normal'
    cohort_labels : list of str
        Labels for each cohort (e.g., accession names)
    
    Returns
    -------
    meta_results : pd.DataFrame
        Combined meta-analysis results with columns:
        gene, log2FC_meta, se_meta, z_stat, pvalue_meta, adj_pvalue_meta,
        direction_consistent, cohort_count
    """
    logger.info("=" * 60)
    logger.info("FIXED-EFFECTS META-ANALYSIS (Inverse-Variance Weighting)")
    logger.info("=" * 60)
    
    # Collect all genes across cohorts
    all_genes = set()
    for de_df in per_cohort_de:
        all_genes.update(de_df["gene"].tolist() if "gene" in de_df.columns else de_df.index.tolist())
    
    all_genes = sorted(list(all_genes))
    logger.info(f"Total genes for meta-analysis: {len(all_genes)}")
    logger.info(f"Cohorts: {cohort_labels}")
    
    results = []
    
    for gene in all_genes:
        betas = []  # log2FC per cohort
        ses = []    # Standard errors per cohort
        directions = []  # Up/down per cohort
        present_cohorts = []
        
        for i, de_df in enumerate(per_cohort_de):
            gene_data = _find_gene_in_df(de_df, gene)
            if gene_data is not None:
                fc = gene_data["log2FC"]
                p = gene_data["pvalue"]
                adj_p = gene_data.get("adj_pvalue", p)
                
                # Convert p-value to t-statistic, then to SE
                # t = fc / SE → SE = fc / t
                # |t| = sqrt(t²), p = 2 * (1 - CDF(|t|)) → |t| = ppf(1 - p/2)
                try:
                    if p > 0 and p <= 1:
                        t_stat = stats.t.ppf(1 - p / 2, df=len(de_df) - 2)
                        if abs(t_stat) > 1e-10 and abs(fc) > 0:
                            se = abs(fc / t_stat)
                        else:
                            # Fallback: use fold change magnitude as proxy
                            se = 1.0
                    else:
                        se = 1.0
                except Exception:
                    se = 1.0
                
                betas.append(fc)
                ses.append(max(se, 1e-10))  # Avoid division by zero
                directions.append(1 if fc > 0 else -1)
                present_cohorts.append(cohort_labels[i])
        
        if len(betas) < 2:
            # Only one cohort has this gene - no meta-analysis possible
            continue
        
        # Fixed-effects inverse-variance weighting
        weights = np.array([1.0 / (se ** 2) for se in ses])
        betas = np.array(betas)
        
        # Check for direction consistency
        direction_consistent = all(d == directions[0] for d in directions)
        
        # Pooled effect
        beta_meta = np.sum(weights * betas) / np.sum(weights)
        se_meta = np.sqrt(1.0 / np.sum(weights))
        
        # Z-statistic and p-value
        z_stat = beta_meta / se_meta
        p_meta = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        results.append({
            "gene": gene,
            "log2FC_meta": beta_meta,
            "se_meta": se_meta,
            "z_stat": z_stat,
            "pvalue_meta": p_meta,
            "direction_consistent": direction_consistent,
            "cohort_count": len(betas),
            "cohorts": ";".join(present_cohorts),
        })
    
    meta_df = pd.DataFrame(results)
    
    if len(meta_df) == 0:
        logger.warning("No genes found in multiple cohorts for meta-analysis")
        return pd.DataFrame(columns=["gene", "log2FC_meta", "se_meta", "z_stat", "pvalue_meta", "adj_pvalue_meta", "direction_consistent", "cohort_count"])
    
    # FDR correction on meta p-values
    meta_df["adj_pvalue_meta"] = multipletests(
        meta_df["pvalue_meta"].values,
        alpha=config.PVALUE_THRESHOLD,
        method="fdr_bh"
    )[1]
    
    # Sort by meta p-value
    meta_df = meta_df.sort_values("pvalue_meta").reset_index(drop=True)
    
    # Log summary
    n_sig = (meta_df["adj_pvalue_meta"] < config.PVALUE_THRESHOLD).sum()
    n_consistent = meta_df["direction_consistent"].sum()
    logger.info(f"Meta-analysis results: {len(meta_df)} genes, {n_sig} significant (FDR < {config.PVALUE_THRESHOLD})")
    logger.info(f"Directionally consistent across cohorts: {n_consistent}/{len(meta_df)}")
    
    # Log top hits
    top = meta_df.head(10)
    for _, row in top.iterrows():
        logger.info(
            f"  {row['gene']}: log2FC={row['log2FC_meta']:+.2f}, "
            f"p={row['pvalue_meta']:.2e}, FDR={row['adj_pvalue_meta']:.2e}, "
            f"consistent={row['direction_consistent']}, cohorts={row['cohort_count']}"
        )
    
    # Save
    out_path = os.path.join(config.RESULTS_DIR, "meta_analysis_results.csv")
    meta_df.to_csv(out_path, index=False)
    logger.info(f"Saved meta-analysis results → {out_path}")
    
    logger.info("Meta-analysis complete ✓")
    return meta_df


def _find_gene_in_df(de_df: pd.DataFrame, gene: str) -> pd.Series | None:
    """Find a gene in a DE results DataFrame, checking both 'gene' column and index."""
    if gene in de_df.index:
        return de_df.loc[gene]
    if "gene" in de_df.columns and gene in de_df["gene"].values:
        return de_df[de_df["gene"] == gene].iloc[0]
    return None


if __name__ == "__main__":
    # Test with synthetic multi-cohort data
    np.random.seed(42)
    
    test_genes = ["EGFR", "KRAS", "TP53", "MYC", "TOP2A", "CDH1", "CXCR4"]
    
    de1 = pd.DataFrame({
        "gene": test_genes,
        "log2FC": [2.1, 1.5, -1.2, 1.8, 2.0, -1.0, 1.3],
        "pvalue": [1e-5, 1e-4, 1e-3, 1e-6, 1e-7, 1e-2, 1e-3],
        "adj_pvalue": [1e-4, 1e-3, 1e-2, 1e-5, 1e-6, 1e-1, 1e-2],
        "mean_tumor": [10, 9, 5, 8, 7, 4, 6],
        "mean_normal": [5, 7, 6, 4, 3, 5, 4],
    })
    
    de2 = pd.DataFrame({
        "gene": test_genes,
        "log2FC": [1.8, np.nan, -0.9, 2.1, 2.2, -0.8, np.nan],
        "pvalue": [1e-4, 1e-3, 1e-2, 1e-5, 1e-6, 1e-2, 1e-3],
        "adj_pvalue": [1e-3, 1e-2, 1e-1, 1e-4, 1e-5, 1e-1, 1e-2],
        "mean_tumor": [9, 8, 6, 9, 8, 5, 7],
        "mean_normal": [6, 6, 7, 5, 4, 6, 5],
    })
    
    results = run_meta_analysis([de1, de2], ["GSE8671", "GSE20916"])
    print("\nMeta-analysis results:")
    print(results[["gene", "log2FC_meta", "pvalue_meta", "adj_pvalue_meta", "direction_consistent"]].to_string())