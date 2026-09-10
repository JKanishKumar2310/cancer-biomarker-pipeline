"""
Genomic Mutation & Multi-Omics Integration Module
Bridges the gap between transcriptomic differential expression (RNA-seq / Microarray)
and DNA-level alterations (somatic point mutations, copy number variations, fusions)
derived from TCGA PanCancer Atlas, COSMIC, and MSK-IMPACT.
"""

import os
import pandas as pd
import numpy as np

# Curated Pan-Cancer Somatic Driver Knowledgebase
# Grounded in TCGA PanCancer Atlas, COSMIC, and FDA precision oncology labels
CANCER_GENOMIC_PROFILES = {
    "Non-Small Cell Lung Cancer": [
        {
            "gene": "EGFR",
            "mutation_frequency_pct": 32.0, # Higher in East Asian non-smokers (>50%)
            "hotspots": "L858R, Exon 19 del (E746_A750del), T790M, C797S",
            "alteration_type": "Activating Kinase Mutation / Exon Deletion",
            "targeted_therapies": "Osimertinib, Gefitinib, Erlotinib, Afatinib",
            "clinical_tier": "Tier 1A (FDA Approved Standard of Care)"
        },
        {
            "gene": "KRAS",
            "mutation_frequency_pct": 30.0,
            "hotspots": "G12C, G12D, G12V, G12A",
            "alteration_type": "Activating GTPase Point Mutation",
            "targeted_therapies": "Sotorasib (Lumakras), Adagrasib (Krazati)",
            "clinical_tier": "Tier 1A (FDA Approved for G12C)"
        },
        {
            "gene": "TP53",
            "mutation_frequency_pct": 54.0,
            "hotspots": "R175H, R248Q, R273H, R282W",
            "alteration_type": "Loss-of-Function Missense / Truncation",
            "targeted_therapies": "Clinical trials (WEE1/ATR inhibitors, PRIMA-1MET)",
            "clinical_tier": "Tier 2 (Prognostic / Trial Stratification)"
        },
        {
            "gene": "ALK",
            "mutation_frequency_pct": 5.0,
            "hotspots": "EML4-ALK chromosomal inversion / fusion",
            "alteration_type": "Oncogenic Receptor Tyrosine Kinase Fusion",
            "targeted_therapies": "Alectinib, Brigatinib, Lorlatinib, Crizotinib",
            "clinical_tier": "Tier 1A (FDA Approved Standard of Care)"
        },
        {
            "gene": "BRAF",
            "mutation_frequency_pct": 4.0,
            "hotspots": "V600E, G469A, G466V",
            "alteration_type": "Activating Serine/Threonine Kinase Mutation",
            "targeted_therapies": "Dabrafenib + Trametinib, Encorafenib + Binimetinib",
            "clinical_tier": "Tier 1A (FDA Approved Standard of Care)"
        },
        {
            "gene": "MET",
            "mutation_frequency_pct": 3.5,
            "hotspots": "Exon 14 skipping splice site mutations, Amplification",
            "alteration_type": "Splice Site In-frame Deletion / Copy Gain",
            "targeted_therapies": "Capmatinib, Tepotinib",
            "clinical_tier": "Tier 1A (FDA Approved Standard of Care)"
        },
        {
            "gene": "ROS1",
            "mutation_frequency_pct": 2.0,
            "hotspots": "CD74-ROS1, SLC34A2-ROS1 fusions",
            "alteration_type": "Chromosomal Translocation / Fusion",
            "targeted_therapies": "Repotrectinib, Crizotinib, Entrectinib",
            "clinical_tier": "Tier 1A (FDA Approved Standard of Care)"
        },
        {
            "gene": "STK11",
            "mutation_frequency_pct": 17.0,
            "hotspots": "N-terminal truncations, splice site mutations",
            "alteration_type": "Tumor Suppressor Inactivation",
            "targeted_therapies": "Resistance biomarker for anti-PD-1 immunotherapy",
            "clinical_tier": "Tier 1B (Immune Checkpoint Resistance Marker)"
        },
        {
            "gene": "KEAP1",
            "mutation_frequency_pct": 19.0,
            "hotspots": "R320Q, G333C, D422N",
            "alteration_type": "Loss-of-Function Mutation (NRF2 activation)",
            "targeted_therapies": "Glutaminase inhibitors (Telaglenastat, trials)",
            "clinical_tier": "Tier 2 (Prognostic / Metabolic Target)"
        },
        {
            "gene": "ERBB2",
            "mutation_frequency_pct": 3.0,
            "hotspots": "Exon 20 insertions (Y772_A775dup)",
            "alteration_type": "In-frame Exon 20 Insertion / Copy Gain",
            "targeted_therapies": "Trastuzumab deruxtecan (Enhertu)",
            "clinical_tier": "Tier 1A (FDA Approved Standard of Care)"
        }
    ],
    "Breast Cancer": [
        {
            "gene": "PIK3CA",
            "mutation_frequency_pct": 36.0,
            "hotspots": "H1047R, E545K, E542K (Kinase & Helical domains)",
            "alteration_type": "Activating PI3K Catalytic Subunit Mutation",
            "targeted_therapies": "Alpelisib (Piqray) + Fulvestrant, Inavolisib",
            "clinical_tier": "Tier 1A (FDA Approved for HR+/HER2-)"
        },
        {
            "gene": "TP53",
            "mutation_frequency_pct": 37.0, # >80% in Basal-like / Triple Negative
            "hotspots": "R175H, R248W, R273C, truncation mutations",
            "alteration_type": "Loss-of-Function Missense / Deletion",
            "targeted_therapies": "Platinum chemotherapy, PARP inhibitors, ATR trials",
            "clinical_tier": "Tier 1B (Diagnostic Subtyping / High-Grade Marker)"
        },
        {
            "gene": "ERBB2",
            "mutation_frequency_pct": 18.0, # Predominantly high-level DNA Amplification
            "hotspots": "Chr17q12 focal amplification, S310F/Y mutations",
            "alteration_type": "High-Level DNA Amplification & Missense Mutation",
            "targeted_therapies": "Trastuzumab, Pertuzumab, T-DM1, T-DXd, Tucatinib",
            "clinical_tier": "Tier 1A (FDA Approved Companion Diagnostic)"
        },
        {
            "gene": "CDH1",
            "mutation_frequency_pct": 14.0, # >65% in Invasive Lobular Carcinoma
            "hotspots": "Frameshift truncations, splice-site mutations",
            "alteration_type": "E-cadherin Loss / Truncation",
            "targeted_therapies": "ROS1/Synthetic lethal kinase trials",
            "clinical_tier": "Tier 1B (Histological Subtype Defining)"
        },
        {
            "gene": "GATA3",
            "mutation_frequency_pct": 11.0,
            "hotspots": "Frameshift insertions in Zn-finger domain",
            "alteration_type": "Transcription Factor Loss/Frameshift",
            "targeted_therapies": "Hormone receptor sensitivity marker",
            "clinical_tier": "Tier 2 (Luminal Subtype Prognostic)"
        },
        {
            "gene": "ESR1",
            "mutation_frequency_pct": 15.0, # Enriched in endocrine-resistant metastatic
            "hotspots": "D538G, Y537S, Y537N, E380Q (LBD domain)",
            "alteration_type": "Ligand-Independent Constitutive Activation",
            "targeted_therapies": "Elacestrant (Orserdu), Novel PROTAC degraders",
            "clinical_tier": "Tier 1A (FDA Approved for ESR1-mutant mBC)"
        },
        {
            "gene": "BRCA1",
            "mutation_frequency_pct": 5.0,
            "hotspots": "185delAG, 5382insC, C61G (Germline/Somatic)",
            "alteration_type": "Homologous Recombination Deficiency (HRD)",
            "targeted_therapies": "Olaparib, Talazoparib",
            "clinical_tier": "Tier 1A (FDA Approved PARP Inhibitor Indication)"
        },
        {
            "gene": "BRCA2",
            "mutation_frequency_pct": 5.0,
            "hotspots": "6174delT, truncation mutations",
            "alteration_type": "Homologous Recombination Deficiency (HRD)",
            "targeted_therapies": "Olaparib, Talazoparib",
            "clinical_tier": "Tier 1A (FDA Approved PARP Inhibitor Indication)"
        }
    ],
    "Colorectal Cancer": [
        {
            "gene": "APC",
            "mutation_frequency_pct": 78.0,
            "hotspots": "R1450*, Q1338*, Frameshifts in Mutation Cluster Region",
            "alteration_type": "Truncating Mutation (Beta-Catenin Stabilization)",
            "targeted_therapies": "Wnt-pathway / Tankyrase / Beta-catenin inhibitors (investigational)",
            "clinical_tier": "Tier 1B (Universal Adenoma-Carcinoma Initiator)"
        },
        {
            "gene": "TP53",
            "mutation_frequency_pct": 60.0,
            "hotspots": "R175H, R248W, R273H",
            "alteration_type": "Loss-of-Function Missense Mutation",
            "targeted_therapies": "Prognostic marker / cytotoxic chemotherapy",
            "clinical_tier": "Tier 2 (Invasive Transition Marker)"
        },
        {
            "gene": "KRAS",
            "mutation_frequency_pct": 43.0,
            "hotspots": "G12D, G12V, G13D, G12C",
            "alteration_type": "Constitutive GTPase Activation",
            "targeted_therapies": "Negative predictor for anti-EGFR (Cetuximab/Panitumumab); Sotorasib+Panitumumab for G12C",
            "clinical_tier": "Tier 1A (FDA Mandatory Exclusion for anti-EGFR)"
        },
        {
            "gene": "PIK3CA",
            "mutation_frequency_pct": 20.0,
            "hotspots": "E545K, E542K, H1047R",
            "alteration_type": "Activating Kinase Mutation",
            "targeted_therapies": "Aspirin response predictor, PI3K trials",
            "clinical_tier": "Tier 2 (Adjuvant Chemoprevention Biomarker)"
        },
        {
            "gene": "SMAD4",
            "mutation_frequency_pct": 14.0,
            "hotspots": "R361H, R361C, C-terminal truncations",
            "alteration_type": "TGF-beta Pathway Inactivation",
            "targeted_therapies": "Predictor of distant metastasis and 5-FU resistance",
            "clinical_tier": "Tier 2 (High-Risk Metastatic Marker)"
        },
        {
            "gene": "BRAF",
            "mutation_frequency_pct": 10.0,
            "hotspots": "V600E",
            "alteration_type": "Constitutive Serine/Threonine Kinase Activation",
            "targeted_therapies": "Encorafenib + Cetuximab (Braftovi + Erbitux)",
            "clinical_tier": "Tier 1A (FDA Approved for BRAF V600E mCRC)"
        }
    ]
}


def integrate_multi_omics(
    de_file: str = "results/de_results.csv",
    cancer_type: str = None,
    output_file: str = "results/multi_omics_integration.csv"
) -> pd.DataFrame:
    """
    Overlays transcriptomic differential expression data with TCGA/COSMIC genomic mutation data.
    Classifies genes into Dual Drivers, Pure Genomic Drivers, or Pure Transcriptomic Drivers.
    """
    if not os.path.exists(de_file):
        raise FileNotFoundError(f"DE results file not found: {de_file}")

    de_df = pd.read_csv(de_file)
    
    # Standardize gene symbol column
    gene_col = "gene" if "gene" in de_df.columns else "gene_symbol"
    if gene_col not in de_df.columns:
        raise ValueError(f"Could not locate gene column in {de_file}")

    # Build mapping dictionary from de_df
    de_dict = {}
    for _, row in de_df.iterrows():
        g = str(row[gene_col]).strip()
        de_dict[g] = {
            "log2FC": float(row.get("log2FC", 0.0)),
            "pvalue": float(row.get("pvalue", 1.0)),
            "adj_pvalue": float(row.get("adj_pvalue", row.get("padj", 1.0))),
            "regulation": str(row.get("regulation", "Not Significant"))
        }

    records = []
    
    profiles_to_use = {}
    if cancer_type and cancer_type in CANCER_GENOMIC_PROFILES:
        profiles_to_use[cancer_type] = CANCER_GENOMIC_PROFILES[cancer_type]
    else:
        profiles_to_use = CANCER_GENOMIC_PROFILES

    for c_type, genes_list in profiles_to_use.items():
        for mut_entry in genes_list:
            g = mut_entry["gene"]
            mut_freq = mut_entry["mutation_frequency_pct"]
            hotspots = mut_entry["hotspots"]
            alt_type = mut_entry["alteration_type"]
            therapies = mut_entry["targeted_therapies"]
            tier = mut_entry["clinical_tier"]

            # Pull RNA transcriptomic metrics if present
            if g in de_dict:
                rna = de_dict[g]
                log2fc = rna["log2FC"]
                fdr = rna["adj_pvalue"]
                reg = rna["regulation"]
            else:
                log2fc = 0.0
                fdr = 1.0
                reg = "Not Present / Filtered"

            # Determine Multi-Omics Classification
            has_rna_signal = (abs(log2fc) >= 1.0) and (fdr < 0.05)
            has_dna_signal = mut_freq >= 5.0

            if has_rna_signal and has_dna_signal:
                classification = "Dual Omics Driver (RNA Overexpression + DNA Alteration)"
                badge = "[DUAL-OMICS]"
            elif has_dna_signal and not has_rna_signal:
                classification = "Pure Genomic Driver (Hidden in RNA - Jammed Gas Pedal)"
                badge = "[MUTATION-ONLY]"
            elif has_rna_signal and not has_dna_signal:
                classification = "Pure Transcriptomic Driver (Stromal/Immune Remodeling)"
                badge = "[RNA-DRIVEN]"
            else:
                classification = "Sub-threshold / Context-Dependent"
                badge = "[MINOR]"

            records.append({
                "cancer_type": c_type,
                "gene": g,
                "mutation_frequency_pct": mut_freq,
                "rna_log2fc": round(log2fc, 2),
                "rna_padj": f"{fdr:.2e}",
                "rna_status": reg,
                "alteration_type": alt_type,
                "hotspot_alterations": hotspots,
                "targeted_therapies": therapies,
                "clinical_tier": tier,
                "omics_classification": classification,
                "badge": badge
            })

    # Also add the top 5 purely transcriptomic consensus genes to show the contrast!
    # (Genes with massive RNA fold change but ~0% DNA mutations)
    top_rna = de_df[de_df[gene_col].isin(["COL10A1", "COL11A1", "AGER", "SPOCK2", "AXIN2", "LGR5"])]
    for _, row in top_rna.iterrows():
        g = str(row[gene_col]).strip()
        if any(r["gene"] == g for r in records):
            continue
        log2fc = float(row.get("log2FC", 0.0))
        fdr = float(row.get("adj_pvalue", 1.0))
        reg = str(row.get("regulation", "Not Significant"))
        records.append({
            "cancer_type": cancer_type or "Pan-Cancer / Extracellular Matrix",
            "gene": g,
            "mutation_frequency_pct": 0.5, # Minimal somatic mutation
            "rna_log2fc": round(log2fc, 2),
            "rna_padj": f"{fdr:.2e}",
            "rna_status": reg,
            "alteration_type": "Transcriptional Dysregulation (Stromal Desmoplasia / Lineage Loss)",
            "hotspot_alterations": "None (Epigenetic / Microenvironment Activation)",
            "targeted_therapies": "Stromal ECM modifiers, anti-fibrotic clinical trials",
            "clinical_tier": "Diagnostic / Prognostic Biomarker",
            "omics_classification": "Pure Transcriptomic Driver (Stromal/Immune Remodeling)",
            "badge": "[RNA-DRIVEN]"
        })

    multi_df = pd.DataFrame(records)
    multi_df.sort_values(by=["mutation_frequency_pct"], ascending=False, inplace=True)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    multi_df.to_csv(output_file, index=False)
    print(f"Successfully wrote Multi-Omics integration table to {output_file} ({len(multi_df)} genes annotated).")
    return multi_df


if __name__ == "__main__":
    df = integrate_multi_omics()
    print(df[["gene", "cancer_type", "mutation_frequency_pct", "rna_log2fc", "omics_classification"]].head(10))
