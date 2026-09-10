"""
Central configuration for the Cancer Biomarker Discovery Pipeline.
All tunable parameters are defined here for reproducibility.
"""
import os

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
# ============================================================
# Dataset
# ============================================================
GEO_ACCESSION = "GSE8671"  # 32 colorectal adenoma polyps + 32 paired normal mucosa (GPL570)
CANCER_TYPE = "Colorectal Adenoma & Carcinoma (CRC)"

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
        "n_estimators": 200,
        "max_depth": None,
        "class_weight": "balanced",
    },
    "GradientBoosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
    },
    "L1_LogisticRegression": {
        "C": 0.1,
        "penalty": "l1",
        "solver": "liblinear",
    },
}

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
