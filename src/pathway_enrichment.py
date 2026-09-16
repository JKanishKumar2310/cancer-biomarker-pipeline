"""
Functional Pathway & Gene Ontology (GO) Enrichment Analysis Module.

Performs Over-Representation Analysis (ORA) via Fisher's exact / Hypergeometric tests
on significantly differentiated genes and consensus biomarkers.
Maps biomarkers to KEGG Signaling Pathways, GO Biological Processes, and Cancer Hallmarks.
"""
import os
import sys
import json
import urllib.request
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger


# Curated Hallmark & Canonical Oncogenic Pathways for Fast, Offline ORA
CANONICAL_PATHWAYS = {
    # Cell Cycle & Proliferation
    "Cell Cycle & Mitotic G1/S Transition (KEGG:hsa04110)": [
        "CDK1", "CDK2", "CDK4", "CDK6", "CCND1", "CCNE1", "CCNA2", "CCNB1", "TOP2A", "MKI67",
        "PCNA", "E2F1", "E2F2", "E2F3", "RB1", "CDKN1A", "CDKN2A", "CDKN2B", "BUB1", "AURKA",
        "AURKB", "PLK1", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "CDC20", "CDC6", "CHEK1"
    ],
    # p53 Signaling & DNA Damage Response
    "p53 Signaling Pathway (KEGG:hsa04115)": [
        "TP53", "MDM2", "MDM4", "CDKN1A", "GADD45A", "BAX", "BAK1", "PUMA", "NOXA", "FAS",
        "CASP3", "CASP8", "CASP9", "ATM", "ATR", "CHEK1", "CHEK2", "RRM2", "ZMAT3", "SERPINE1"
    ],
    # PI3K-Akt-mTOR Signaling
    "PI3K-Akt Signaling Pathway (KEGG:hsa04151)": [
        "PIK3CA", "PIK3CB", "PIK3CD", "PIK3R1", "PTEN", "AKT1", "AKT2", "AKT3", "MTOR", "RPS6KB1",
        "EIF4EBP1", "GSK3B", "FOXO1", "FOXO3", "BAD", "BCL2", "VEGFA", "EGFR", "ERBB2", "IGF1R"
    ],
    # MAPK / RAS / RAF Signaling
    "MAPK Signaling Pathway (KEGG:hsa04010)": [
        "EGFR", "ERBB2", "KRAS", "HRAS", "NRAS", "BRAF", "RAF1", "MAP2K1", "MAP2K2", "MAPK1",
        "MAPK3", "MYC", "FOS", "JUN", "ETS1", "DUSP1", "DUSP6", "SPRY2", "SPRY4", "NF1"
    ],
    # Epithelial to Mesenchymal Transition & Metastasis
    "Epithelial-Mesenchymal Transition (GO:0001837)": [
        "CDH1", "CDH2", "VIM", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "FN1", "MMP2",
        "MMP9", "MMP14", "TGFB1", "TGFBR1", "SMAD2", "SMAD3", "COL1A1", "COL1A2", "COL3A1", "COL6A3",
        "CEMIP", "LGR5", "CLDN1", "EPCAM"
    ],
    # Hypoxia & Angiogenesis
    "Hypoxia & Angiogenesis Signaling (GO:0001525)": [
        "HIF1A", "EPAS1", "VEGFA", "VEGFB", "VEGFC", "KDR", "FLT1", "ANGPT1", "ANGPT2", "TEK",
        "SLC2A1", "LDHA", "ENO1", "PDK1", "CA9", "SERPINE1", "CXCR4", "NOS3"
    ],
    # Wnt / Beta-Catenin Signaling
    "Wnt Signaling Pathway (KEGG:hsa04310)": [
        "WNT1", "WNT3A", "WNT5A", "CTNNB1", "APC", "AXIN1", "AXIN2", "GSK3B", "TCF7", "LEF1",
        "MYC", "CCND1", "LGR5", "RNF43", "ZNRF3", "SFRP1", "SFRP2", "DKK1", "FZD1", "DVL1"
    ],
    # Glycolysis & Metabolic Reprogramming
    "Glycolytic Reprogramming & Warburg Effect (GO:0006096)": [
        "SLC2A1", "SLC2A3", "HK1", "HK2", "GPI", "PFKP", "ALDOA", "GAPDH", "PGK1", "PGAM1",
        "ENO1", "PKM", "LDHA", "PDK1", "TIGAR", "FBP1", "G6PD"
    ],
    # Apoptosis & Programmed Cell Death
    "Apoptosis Signaling (KEGG:hsa04210)": [
        "BCL2", "BCL2L1", "MCL1", "BAX", "BAK1", "BOK", "BID", "BIM", "PUMA", "NOXA",
        "CASP3", "CASP7", "CASP8", "CASP9", "APAF1", "DIABLO", "CYCS", "XIAP", "FADD", "TNFRSF10A"
    ],
    # Immune Checkpoint & T-Cell Exhaustion
    "Immune Checkpoint & T-Cell Infiltration (GO:0002250)": [
        "CD274", "PDCD1LG2", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "CD8A", "CD8B", "CD4",
        "IFNG", "GZMA", "GZMB", "PRF1", "CXCL9", "CXCL10", "CXCL11", "FOXP3", "TGFB1", "IDO1"
    ],
    # Extracellular Matrix Organization
    "Extracellular Matrix Organization (GO:0030198)": [
        "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL5A1", "COL6A1", "COL6A2", "COL6A3", "FN1", "LAMA1",
        "LAMB1", "LAMC1", "MMP1", "MMP2", "MMP7", "MMP9", "MMP11", "MMP14", "TIMP1", "TIMP2", "LOX", "LOXL2"
    ],
    # Thyroid Hormone / Endocrine Signaling
    "Thyroid & Endocrine Carcinogenesis (KEGG:hsa05216)": [
        "BRAF", "RET", "NTRK1", "NTRK3", "RAS", "HRAS", "KRAS", "NRAS", "PAX8", "PPARG",
        "TG", "TPO", "SLC5A5", "TSHR", "TTF1", "NKX2-1", "MMRN1", "TFF3", "TPPP", "CDKN2B"
    ],
    # Prostate & Androgen Receptor Signaling
    "Androgen Receptor & Prostate Pathogenesis (KEGG:hsa05215)": [
        "AR", "KLK3", "KLK2", "TMPRSS2", "PCA3", "FOXA1", "ERG", "ETV1", "ETV4", "PTEN",
        "TP53", "RB1", "MYC", "EZH2", "NCOR1", "NCOR2", "SPOP", "RAD51B", "SLAMF7"
    ]
}


def run_pathway_enrichment(
    genes: list[str],
    background_gene_count: int = 16500,
    pathway_db: dict = None
) -> pd.DataFrame:
    """
    Execute Over-Representation Analysis (ORA) via Fisher's Exact test.

    Parameters
    ----------
    genes : list of str
        List of target gene symbols (e.g. significant DEGs or consensus biomarkers).
    background_gene_count : int
        Estimated total background transcriptome size (default ~16,500 after variance filtering).
    pathway_db : dict, optional
        Dictionary mapping pathway names to lists of gene symbols. Defaults to CANONICAL_PATHWAYS.

    Returns
    -------
    pd.DataFrame with enrichment results sorted by adjusted p-value.
    """
    if pathway_db is None:
        pathway_db = CANONICAL_PATHWAYS

    query_genes = set(g.strip().upper() for g in genes if isinstance(g, str) and g.strip())
    k = len(query_genes)  # Number of input genes
    N = background_gene_count

    results = []

    for pathway_name, p_genes in pathway_db.items():
        path_set = set(g.strip().upper() for g in p_genes)
        M = len(path_set)  # Pathway size

        overlap = query_genes.intersection(path_set)
        x = len(overlap)   # Overlap count

        if x == 0:
            continue

        # 2x2 contingency table for Fisher's Exact Test:
        #                 In Pathway    Not In Pathway
        # In Query List       x             k - x
        # Not In Query      M - x         N - M - (k - x)
        table = [
            [x, k - x],
            [M - x, max(0, N - M - (k - x))]
        ]

        try:
            odds_ratio, p_val = fisher_exact(table, alternative="greater")
        except Exception:
            p_val = 1.0

        # Expected overlap by chance
        expected = (k * M) / N
        fold_enrichment = (x / expected) if expected > 0 else 0.0

        # Extract Database Category
        db_type = "KEGG" if "KEGG" in pathway_name else ("GO" if "GO" in pathway_name else "Hallmark")

        results.append({
            "pathway": pathway_name,
            "database": db_type,
            "overlap_count": x,
            "pathway_size": M,
            "fold_enrichment": round(fold_enrichment, 2),
            "p_value": float(p_val),
            "genes": ", ".join(sorted(overlap)),
        })

    if not results:
        return pd.DataFrame(columns=[
            "pathway", "database", "overlap_count", "pathway_size", "fold_enrichment",
            "p_value", "adjusted_p_value", "neg_log10_fdr", "genes"
        ])

    df = pd.DataFrame(results)

    # Benjamini-Hochberg FDR correction
    df = df.sort_values("p_value").reset_index(drop=True)
    m = len(df)
    adj_p = []
    for rank, p in enumerate(df["p_value"], start=1):
        adj_p.append(min(1.0, p * (m / rank)))

    # Ensure monotonicity of BH adjusted p-values
    for i in range(m - 2, -1, -1):
        adj_p[i] = min(adj_p[i], adj_p[i + 1])

    df["adjusted_p_value"] = [float(f"{p:.2e}") if p < 0.001 else round(p, 4) for p in adj_p]
    df["neg_log10_fdr"] = [round(-np.log10(max(p, 1e-300)), 2) for p in adj_p]

    return df


def analyze_and_export_pathways(
    de_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    out_dir: str = None
) -> pd.DataFrame:
    """
    Runs pathway enrichment for both the full significant DEG pool and consensus biomarkers,
    and saves the results to CSV.
    """
    if out_dir is None:
        out_dir = config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    sig_degs = de_df[de_df["significant"]]["gene"].dropna().tolist() if "significant" in de_df.columns else de_df["gene"].tolist()
    consensus_genes = consensus_df["gene"].dropna().tolist() if "gene" in consensus_df.columns else []

    combined_query = list(set(sig_degs + consensus_genes))
    enrichment_df = run_pathway_enrichment(combined_query)

    csv_path = os.path.join(out_dir, "pathway_enrichment.csv")
    enrichment_df.to_csv(csv_path, index=False)
    logger.info(f"Pathway enrichment complete: {len(enrichment_df)} pathways evaluated (Saved to: {csv_path})")

    return enrichment_df


if __name__ == "__main__":
    # Test with sample oncogenic genes
    sample_genes = ["CDK1", "TOP2A", "MKI67", "PCNA", "CCND1", "MYC", "EGFR", "VEGFA", "TP53", "CDH1", "VIM"]
    res = run_pathway_enrichment(sample_genes)
    print("Pathway Enrichment Test Results:")
    print(res[["pathway", "overlap_count", "fold_enrichment", "p_value", "adjusted_p_value", "genes"]].to_string(index=False))
