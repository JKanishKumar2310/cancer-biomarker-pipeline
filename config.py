"""
Central configuration for the Cancer Biomarker Discovery Pipeline.
All tunable parameters are defined here for reproducibility.
"""
import os
import sys
import warnings

# Ensure UTF-8 console output on Windows to prevent UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Suppress scikit-learn / urllib future deprecation notices for clean logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load environment variables from .env if present
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

# ============================================================
# Dataset
# ============================================================
GEO_ACCESSION = "GSE30784"  # 167 Oral Squamous Cell Carcinoma (OSCC) + 45 Normal oral mucosa (GPL570)
CANCER_TYPE = "Oral Squamous Cell Carcinoma (OSCC)"

# Multi-cohort discovery (fixed-effects meta-analysis)
# Set DISCOVERY_COHORTS to a list of GEO accessions to enable meta-analysis
# Example: DISCOVERY_COHORTS = ["GSE8671", "GSE20916"]
DISCOVERY_COHORTS = []

# Data type: "microarray" (default, Affymetrix/Illumina arrays) or "rnaseq" (raw counts)
DATA_TYPE = "microarray"

# ============================================================
# Comparison Design
# ============================================================
# The pipeline contrasts two sample groups. The default is the classic
# tumor-vs-normal oncology comparison, but many modern cancer datasets only
# provide a subtype / treatment-response / stage contrast. Set COMPARISON_MODE
# and adjust the two group labels + keyword tokens to handle those cohorts.
#
#   COMPARISON_MODE = "tumor_vs_normal"        # classic case vs healthy/adjacent
#   COMPARISON_MODE = "subtype_vs_subtype"     # e.g. ER+ vs ER-, stage I vs IV
#   COMPARISON_MODE = "responder_vs_non_responder"  # treatment response
#
# GROUP_A is the "case/positive" group (equivalent to Tumor in the DE code),
# GROUP_B is the "control/negative" group (equivalent to Normal).
COMPARISON_MODE = "tumor_vs_normal"
GROUP_A_LABEL = "Tumor"
GROUP_B_LABEL = "Normal"

# Keyword tokens used by the ontology matcher to assign samples to GROUP_A / GROUP_B.
# Matching is case-insensitive and substring-based against sample title + characteristics.
GROUP_A_TOKENS = [
    "tumor", "tumour", "cancer", "malignant", "carcinoma", "adenoma", "polyp", "neoplasm",
    "case", "primary", "disease", "glioma", "melanoma", "sarcoma", "astrocytoma",
    "glioblastoma", "oligodendroglioma", "ptc", "atc", "leukemia", "lymphoma", "myeloma",
]
GROUP_B_TOKENS = [
    "normal", "healthy", "adjacent", "control", "ctrl", "non-tumor", "nontumor", "non-tumour", "nontumour", "mucosa",
    "paired normal", "benign", "donor", "non-malignant", "nonmalignant", "epilepsy", "non-cancerous", "noncancerous",
]

# ============================================================
# Reproducibility
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Preprocessing
# ============================================================
LOW_VARIANCE_PERCENTILE = 25  # Remove bottom 25% low-variance genes
LOG2_TRANSFORM = True

# ============================================================
# Differential Expression
# ============================================================
FC_THRESHOLD = 1.0        # |log2 fold-change| threshold
PVALUE_THRESHOLD = 0.05   # Adjusted p-value (FDR) threshold
TOP_DE_GENES = 50         # Number of top DE genes for heatmap

# ============================================================
# ML Biomarker Ranking
# ============================================================
N_SPLITS = 5              # Stratified K-Fold splits
TOP_ML_GENES = 50         # Number of top ML-ranked genes

ML_MODELS = {
    "RandomForest": {
        "n_estimators": 100,
        "max_depth": None,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    "GradientBoosting": {
        "n_estimators": 50,
        "learning_rate": 0.1,
        "max_depth": 3,
        "max_features": "sqrt",
    },
    "L1_LogisticRegression": {
        "C": 0.1,
        "l1_ratio": 1,  # pure L1 (penalty= kwarg is deprecated in sklearn >= 1.8)
        "solver": "liblinear",
    },
}

L1_C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

# ============================================================
# Pathway Analysis
# ============================================================
ENRICHMENT_GENE_SETS = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
ENRICHMENT_TOP_TERMS = 15
ENRICHMENT_ORGANISM = "human"

# ============================================================
# Dashboard
# ============================================================
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = True

# ============================================================
# Feature Flags (default OFF for backward compatibility)
# ============================================================
ENABLE_RNASEQ = False           # Enable RNA-seq count processing (voom/limma)
ENABLE_META_ANALYSIS = False    # Enable multi-cohort fixed-effects meta-analysis
ENABLE_BATCH_CORRECTION = False # Enable ComBat batch correction for multi-cohort
ENABLE_L1_TUNING = True         # Enable L1 penalty hyperparameter tuning (inner CV)

# RNA-seq specific (used when DATA_TYPE = "rnaseq" or ENABLE_RNASEQ = True)
RNASEQ_NORMALIZATION = "tmm"    # "tmm" (edgeR) or "median_of_ratios" (DESeq2)
RNASEQ_MIN_COUNTS = 10          # Minimum counts per gene for filtering
RNASEQ_MIN_SAMPLES = 3          # Minimum samples with counts > RNASEQ_MIN_COUNTS

# ============================================================
# Known cancer marker genes (Colorectal Adenoma & Carcinoma Hallmarks)
# ============================================================
KNOWN_MARKERS = [
    # Wnt/Beta-Catenin Pathway (Early Adenoma Initiation & Stemness)
    "APC",      # Adenomatous polyposis coli (classic CRC gatekeeper suppressor)
    "CTNNB1",   # Beta-catenin (transcription co-activator of Wnt target genes)
    "AXIN2",    # Direct Wnt/Beta-catenin feedback target (hallmark in adenomas)
    "LGR5",     # Intestinal stem cell marker and R-spondin receptor
    "MYC",      # c-Myc oncogene (Wnt-driven transcription factor)
    "CCND1",    # Cyclin D1 (G1/S transition driver activated by Wnt)

    # Intestinal & Epithelial Lineage Biomarkers
    "CDX2",     # Caudal type homeobox 2 (colorectal lineage transcription factor)
    "CEACAM5",  # Carcinoembryonic antigen (CEA - standard clinical CRC tumor marker)
    "EPCAM",    # Epithelial cell adhesion molecule (carcinoma marker)
    "KRT20",    # Cytokeratin 20 (colorectal adenocarcinoma marker)
    "CDH1",     # E-cadherin (epithelial adherens junction / EMT regulator)

    # Actionable Oncogenic Kinases & Drivers
    "KRAS",     # KRAS GTPase (codon 12/13 mutation biomarker for anti-EGFR resistance)
    "BRAF",     # B-Raf kinase (V600E mutation target of Encorafenib)
    "EGFR",     # Epidermal growth factor receptor (target of Cetuximab & Panitumumab)
    "PIK3CA",   # PI3K catalytic subunit (actionable oncogenic kinase)
    "VEGFA",    # VEGF-A (target of Bevacizumab angiogenesis blockade)

    # Proliferation & Mitotic Spindle Machinery
    "MKI67",    # Ki-67 proliferation marker
    "TOP2A",    # DNA topoisomerase II alpha (target of Irinotecan/Etoposide)
    "CDK1",     # Cyclin-dependent kinase 1 (mitosis)
    "PCNA",     # Proliferating cell nuclear antigen

    # DNA Mismatch Repair (MMR / Microsatellite Instability)
    "MLH1",     # MutL homolog 1 (Lynch syndrome / MSI-H hallmark)
    "MSH2",     # MutS homolog 2 (DNA mismatch repair)

    # Tumor Suppressors & TGF-Beta Axis
    "TP53",     # p53 tumor suppressor (adenoma-to-carcinoma progression)
    "SMAD4",    # TGF-beta signal transducer (lost in invasive progression)
    "PTEN",     # PI3K pathway inhibitor
]
