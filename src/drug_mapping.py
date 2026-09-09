"""
Targeted Drug Sensitivity & Actionability Mapping.

Maps identified cancer biomarkers to FDA-approved oncology drugs,
investigational agents, mechanisms of action, and clinical indications.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger

# Clinically verified oncology drug database
# Source: FDA Oncology Approvals, NCCN Guidelines, DrugBank, OncoKB
DRUG_DATABASE = {
    "EGFR": {
        "gene_name": "Epidermal Growth Factor Receptor (ErbB-1 / HER1)",
        "approved_drugs": ["Osimertinib (Tagrisso)", "Gefitinib (Iressa)", "Erlotinib (Tarceva)", "Afatinib (Gilotrif)"],
        "drug_class": "1st/2nd/3rd-Generation EGFR Tyrosine Kinase Inhibitors (TKIs)",
        "mechanism": "Covalent/ATP-competitive inhibition of EGFR kinase domain suppressing downstream MAPK and PI3K signaling",
        "indication": "Metastatic Non-Small Cell Lung Cancer (NSCLC) with sensitizing EGFR exon 19 del or L858R mutations",
        "evidence_tier": "Tier 1: FDA-Approved Standard of Care",
        "clinical_action": "Primary actionable oncogenic driver; dictates first-line Osimertinib therapy"
    },
    "KRAS": {
        "gene_name": "KRAS Proto-Oncogene GTPase",
        "approved_drugs": ["Sotorasib (Lumakras)", "Adagrasib (Krazati)"],
        "drug_class": "Small Molecule Covalent KRAS G12C Inhibitors",
        "mechanism": "Irreversible binding to the switch-II pocket of GDP-bound KRAS G12C locking it in an inactive state",
        "indication": "Locally advanced or metastatic NSCLC with KRAS G12C mutation after systemic therapy",
        "evidence_tier": "Tier 1: FDA-Approved Targeted Therapy",
        "clinical_action": "Confirms KRAS G12C actionable mutation for selective covalent inhibition"
    },
    "ALK": {
        "gene_name": "Anaplastic Lymphoma Receptor Tyrosine Kinase",
        "approved_drugs": ["Alectinib (Alecensa)", "Brigatinib (Alunbrig)", "Lorlatinib (Lorbrena)", "Crizotinib (Xalkori)"],
        "drug_class": "2nd/3rd-Generation ALK Kinase Inhibitors",
        "mechanism": "Potent inhibition of ALK tyrosine kinase activity and blood-brain barrier penetration",
        "indication": "Metastatic NSCLC harboring ALK rearrangements (EML4-ALK fusions)",
        "evidence_tier": "Tier 1: FDA-Approved Standard of Care",
        "clinical_action": "Rearrangement dictates immediate first-line Alectinib/Lorlatinib targeted therapy"
    },
    "MET": {
        "gene_name": "MET Proto-Oncogene Receptor Tyrosine Kinase (c-Met)",
        "approved_drugs": ["Capmatinib (Tabrecta)", "Tepotinib (Tepmetko)"],
        "drug_class": "Selective MET Tyrosine Kinase Inhibitors",
        "mechanism": "ATP-competitive inhibition of MET autophosphorylation preventing oncogenic downstream signaling",
        "indication": "Metastatic NSCLC with MET exon 14 skipping mutations",
        "evidence_tier": "Tier 1: FDA-Approved Targeted Therapy",
        "clinical_action": "Overexpression or exon 14 skipping guides Capmatinib/Tepotinib treatment"
    },
    "BRAF": {
        "gene_name": "B-Raf Proto-Oncogene Serine/Threonine Kinase",
        "approved_drugs": ["Dabrafenib (Tafinlar) + Trametinib (Mekinist)"],
        "drug_class": "BRAF Inhibitor + MEK Inhibitor Combination",
        "mechanism": "Dual inhibition of BRAF kinase and downstream MEK1/2 preventing ERK reactivation",
        "indication": "Metastatic NSCLC with BRAF V600E mutation",
        "evidence_tier": "Tier 1: FDA-Approved Standard of Care",
        "clinical_action": "Detects BRAF V600E driver requiring dual pathway targeted combination"
    },
    "RET": {
        "gene_name": "Ret Proto-Oncogene Receptor Tyrosine Kinase",
        "approved_drugs": ["Selpercatinib (Retevmo)", "Pralsetinib (Gavreto)"],
        "drug_class": "Highly Selective RET Tyrosine Kinase Inhibitors",
        "mechanism": "Potent ATP-competitive inhibition of RET wild-type and oncogenic fusion isoforms",
        "indication": "Metastatic NSCLC with RET fusions (e.g. KIF5B-RET, CCDC6-RET)",
        "evidence_tier": "Tier 1: FDA-Approved Targeted Therapy",
        "clinical_action": "Rearrangement indicates high sensitivity to selective RET kinase inhibition"
    },
    "CD274": {
        "gene_name": "Programmed Cell Death 1 Ligand 1 (PD-L1 / B7-H1)",
        "approved_drugs": ["Pembrolizumab (Keytruda)", "Atezolizumab (Tecentriq)", "Cemiplimab (Libtayo)"],
        "drug_class": "Immune Checkpoint Monoclonal Antibodies (anti-PD-1 / anti-PD-L1)",
        "mechanism": "Blocks PD-1/PD-L1 interaction to reverse T-cell exhaustion and restore antitumor cytotoxicity",
        "indication": "First-line monotherapy or chemo-immunotherapy for advanced/metastatic NSCLC",
        "evidence_tier": "Tier 1: FDA-Approved Standard of Care",
        "clinical_action": "Tumor Proportion Score (TPS >= 50%) establishes first-line immunotherapy eligibility"
    },
    "TACSTD2": {
        "gene_name": "TROP2 (Trophoblast cell-surface antigen 2)",
        "approved_drugs": ["Sacituzumab govitecan (Trodelvy)", "Dato-DXd (Datopotamab deruxtecan - Phase III)"],
        "drug_class": "Antibody-Drug Conjugate (ADC)",
        "mechanism": "Anti-Trop-2 humanized mAb conjugated to SN-38 or topoisomerase I inhibitor DXd",
        "indication": "Advanced NSCLC (Phase III TROPION-Lung01) and metastatic carcinomas",
        "evidence_tier": "Tier 2: Advanced Clinical Trials & Standard in Carcinomas",
        "clinical_action": "High expression predicts response to novel Trop-2 targeted antibody-drug conjugates"
    },
    "CDK4": {
        "gene_name": "Cyclin-Dependent Kinase 4",
        "approved_drugs": ["Palbociclib (Ibrance)", "Ribociclib (Kisqali)", "Abemaciclib (Verzenio)"],
        "drug_class": "Small Molecule Selective CDK4/6 Inhibitors",
        "mechanism": "Selective ATP-competitive inhibition of CDK4 and CDK6 preventing Rb phosphorylation and G1/S arrest",
        "indication": "Carcinomas and solid tumors with intact Rb / CDKN2A loss",
        "evidence_tier": "Tier 1: FDA-Approved Standard of Care",
        "clinical_action": "Elevated CDK4/Cyclin D1 signaling indicates sensitivity to CDK4/6 blockade"
    },
    "ERBB2": {
        "gene_name": "HER2 / Receptor Tyrosine-Protein Kinase erbB-2",
        "approved_drugs": ["Trastuzumab deruxtecan (T-DXd / Enhertu)", "Trastuzumab (Herceptin)"],
        "drug_class": "Antibody-Drug Conjugate & Monoclonal Antibody",
        "mechanism": "Targeting extracellular domain of HER2 delivering topoisomerase I inhibitor payload",
        "indication": "Unresectable or metastatic NSCLC with activating HER2 (ERBB2) mutations (FDA Approved)",
        "evidence_tier": "Tier 1: FDA-Approved Standard of Care",
        "clinical_action": "Activating mutations define immediate indication for T-DXd in lung cancer"
    },
    "TOP2A": {
        "gene_name": "DNA Topoisomerase II Alpha",
        "approved_drugs": ["Etoposide (Toposar / VePesid)", "Doxorubicin"],
        "drug_class": "Topoisomerase II Inhibitors",
        "mechanism": "Stabilizes topoisomerase II-DNA cleavage complexes causing lethal double-strand breaks",
        "indication": "First-line chemotherapy in lung carcinomas (Etoposide + Platinum)",
        "evidence_tier": "Tier 1: Clinical Guideline Established",
        "clinical_action": "High expression correlates with heightened sensitivity to Etoposide-based regimens"
    },
    "PTEN": {
        "gene_name": "Phosphatase and Tensin Homolog",
        "approved_drugs": ["Capivasertib (Truqap)", "Alpelisib (Piqray)"],
        "drug_class": "Pan-AKT Inhibitor / PI3K Alpha Inhibitor",
        "mechanism": "Inhibition of AKT1/2/3 phosphorylation downstream of PTEN loss",
        "indication": "Advanced solid tumors and carcinomas with PTEN deficiency",
        "evidence_tier": "Tier 1: Targeted Therapy",
        "clinical_action": "Loss of PTEN activates PI3K/AKT pathway; biomarker for Capivasertib/AKT inhibition"
    },
    "EPCAM": {
        "gene_name": "Epithelial Cell Adhesion Molecule",
        "approved_drugs": ["Catumaxomab (Removab)", "Adecatumumab (investigational)"],
        "drug_class": "Bispecific Monoclonal Antibody (anti-EpCAM x anti-CD3)",
        "mechanism": "Redirects T-cells to lyse EpCAM-positive carcinoma cells",
        "indication": "Malignant pleural effusions and ascites in epithelial carcinomas",
        "evidence_tier": "Tier 2: Investigational & Targeted Clinical Trials",
        "clinical_action": "Universal epithelial carcinoma target for bispecific antibodies and CAR-T trials"
    },
    "VEGFA": {
        "gene_name": "Vascular Endothelial Growth Factor A",
        "approved_drugs": ["Bevacizumab (Avastin)", "Ramucirumab (Cyramza)"],
        "drug_class": "Anti-VEGF / Anti-VEGFR2 Monoclonal Antibodies",
        "mechanism": "Neutralizes VEGF-A / blocks VEGFR2 preventing endothelial proliferation and angiogenesis",
        "indication": "Non-squamous advanced NSCLC in combination with platinum chemotherapy",
        "evidence_tier": "Tier 1: FDA-Approved Standard of Care",
        "clinical_action": "High expression indicates angiogenic tumor vasculature and benefit from Bevacizumab"
    },
    "KRT19": {
        "gene_name": "Keratin 19 / Cytokeratin 19 (CYFRA 21-1)",
        "approved_drugs": ["CYFRA 21-1 ECLIA / ELISA In Vitro Diagnostic (IVD) Assay"],
        "drug_class": "Diagnostic Serum Biomarker Companion Assay",
        "mechanism": "Serum measurement of soluble cytokeratin 19 fragments released by dying carcinoma cells",
        "indication": "Monitoring treatment response, residual disease, and recurrence in NSCLC",
        "evidence_tier": "Tier 1: International Clinical Diagnostic Benchmark",
        "clinical_action": "Established blood biomarker for monitoring therapeutic response in lung cancer"
    }
}


def map_biomarkers_to_drugs(biomarkers_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Map biomarker gene list to actionable oncology drugs.

    Parameters
    ----------
    biomarkers_df : pd.DataFrame, optional
        DataFrame of consensus biomarkers with 'gene' column.

    Returns
    -------
    actionability_df : pd.DataFrame
        Table of biomarkers with matched drugs, classes, indications, and tiers.
    """
    logger.info("Mapping biomarkers to targeted oncology drugs...")

    if biomarkers_df is not None and "gene" in biomarkers_df.columns:
        query_genes = list(dict.fromkeys(biomarkers_df["gene"].tolist() + config.KNOWN_MARKERS + list(DRUG_DATABASE.keys())))
    else:
        query_genes = list(DRUG_DATABASE.keys())

    records = []
    for gene in query_genes:
        if gene in DRUG_DATABASE:
            entry = DRUG_DATABASE[gene]
            records.append({
                "gene": gene,
                "gene_name": entry["gene_name"],
                "approved_drugs": "; ".join(entry["approved_drugs"]),
                "drug_class": entry["drug_class"],
                "mechanism": entry["mechanism"],
                "indication": entry["indication"],
                "evidence_tier": entry["evidence_tier"],
                "clinical_action": entry["clinical_action"],
            })

    df = pd.DataFrame(records)
    if len(df) > 0:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(config.RESULTS_DIR, "drug_actionability.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved drug actionability mappings for {len(df)} biomarkers -> {out_path}")
    return df


def get_drug_details_for_gene(gene: str) -> dict | None:
    """Return drug mapping dictionary for a single gene."""
    return DRUG_DATABASE.get(gene, None)


if __name__ == "__main__":
    df = map_biomarkers_to_drugs()
    print("\nDrug Actionability Preview:")
    for _, row in df.head(6).iterrows():
        print(f"\n[{row['gene']}] {row['gene_name']}")
        print(f"  Drugs: {row['approved_drugs']}")
        print(f"  Tier:  {row['evidence_tier']}")
