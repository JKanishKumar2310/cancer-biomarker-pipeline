"""
Data Loader — Downloads and parses GEO dataset for cancer biomarker analysis.

Supports multiple GEO accessions configured in config.py:
  - GSE15852: Breast cancer (43 tumor + 43 normal, GPL96)
  - GSE19804: Lung adenocarcinoma (60 tumor + 60 normal, GPL570)
"""
import os
import gzip
import urllib.request
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger


# ── Platform annotation map ─────────────────────────────────
# Maps GEO accession → (platform GPL ID, annotation URL)
_PLATFORM_MAP = {
    "GSE15852": ("GPL96", "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz"),
    "GSE19804": ("GPL570", "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL570nnn/GPL570/annot/GPL570.annot.gz"),
    "GSE8671": ("GPL570", "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL570nnn/GPL570/annot/GPL570.annot.gz"),
}

# ── Series matrix download URLs ─────────────────────────────
_MATRIX_URLS = {
    "GSE15852": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE15nnn/GSE15852/matrix/GSE15852_series_matrix.txt.gz",
    "GSE19804": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE19nnn/GSE19804/matrix/GSE19804_series_matrix.txt.gz",
    "GSE8671": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE8nnn/GSE8671/matrix/GSE8671_series_matrix.txt.gz",
}


def _download_if_missing(path: str, url: str, label: str):
    """Download a file from a URL if it doesn't already exist locally."""
    if not os.path.exists(path):
        logger.info(f"Downloading {label} from NCBI...")
        urllib.request.urlretrieve(url, path)
        logger.info(f"  Saved → {os.path.basename(path)}")


def _load_from_series_matrix() -> tuple[pd.DataFrame, pd.Series]:
    """
    Parse series matrix for the configured GEO accession and map probes
    to gene symbols using the appropriate GPL annotation.
    """
    accession = config.GEO_ACCESSION
    gpl_id, annot_url = _PLATFORM_MAP.get(accession, ("GPL570", _PLATFORM_MAP["GSE19804"][1]))

    matrix_file = os.path.join(config.DATA_DIR, f"{accession}_series_matrix.txt.gz")
    annot_file = os.path.join(config.DATA_DIR, f"{gpl_id}.annot.gz")

    # Download if needed
    if accession in _MATRIX_URLS:
        _download_if_missing(matrix_file, _MATRIX_URLS[accession], f"{accession} series matrix")
    _download_if_missing(annot_file, annot_url, f"{gpl_id} annotation table")

    # ── Parse metadata ───────────────────────────────────────
    logger.info(f"Parsing sample metadata from {accession} series matrix...")
    meta_lines = []
    skiprows = 0
    with gzip.open(matrix_file, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("!series_matrix_table_begin"):
                skiprows = i + 1
                break
            meta_lines.append(line)

    sample_ids, sample_titles, characteristics = [], [], []
    for line in meta_lines:
        if line.startswith("!Sample_geo_accession"):
            sample_ids = [x.strip(' "\t\r\n') for x in line.split("\t")[1:]]
        elif line.startswith("!Sample_title"):
            sample_titles = [x.strip(' "\t\r\n') for x in line.split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            characteristics.append([x.strip(' "\t\r\n') for x in line.split("\t")[1:]])

    # Classify samples as Tumor/Normal (title has the cleanest signal)
    labels = {}
    for i, s_id in enumerate(sample_ids):
        title = sample_titles[i].lower() if i < len(sample_titles) else ""
        chars = " ".join([ch[i].lower() for ch in characteristics if i < len(ch) and not ch[i].lower().startswith("cel filename")])

        if any(w in title for w in ["normal", "healthy", "adjacent"]):
            labels[s_id] = "Normal"
        elif any(w in title for w in ["tumor", "tumour", "cancer", "malignant", "carcinoma", "adenoma", "polyp"]):
            labels[s_id] = "Tumor"
        elif any(w in chars for w in ["tissue: normal", "normal", "healthy", "non-tumor"]):
            labels[s_id] = "Normal"
        elif any(w in chars for w in ["tissue: adenoma", "tissue: tumor", "tumor", "cancer", "adenoma"]):
            labels[s_id] = "Tumor"
        else:
            labels[s_id] = "Unknown"

    label_series = pd.Series(labels, name="condition")
    n_unknown = (label_series == "Unknown").sum()
    if n_unknown > 0:
        logger.warning(f"{n_unknown} samples have unknown labels — will be dropped")
    label_series = label_series[label_series != "Unknown"]

    counts = label_series.value_counts()
    logger.info(f"Parsed labels: {dict(counts)}")

    # ── Read expression table ────────────────────────────────
    logger.info(f"Reading {accession} expression values...")
    df = pd.read_csv(
        matrix_file, compression="gzip", skiprows=skiprows, sep="\t", index_col=0, comment="!"
    )
    df = df[~df.index.astype(str).str.startswith("!")]
    df = df.apply(pd.to_numeric, errors="coerce")

    # ── Map probes → gene symbols ────────────────────────────
    logger.info(f"Mapping probe IDs to gene symbols via {gpl_id}...")
    annot_skip = 0
    with gzip.open(annot_file, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("!platform_table_begin"):
                annot_skip = i + 1
                break

    annot_df = pd.read_csv(
        annot_file, compression="gzip", skiprows=annot_skip, sep="\t",
        usecols=["ID", "Gene symbol"], low_memory=False
    )
    annot_df = annot_df.dropna(subset=["Gene symbol"])
    annot_df = annot_df[~annot_df["Gene symbol"].str.strip().isin(["", "---"])]
    annot_df["Gene symbol"] = annot_df["Gene symbol"].apply(lambda x: str(x).split("///")[0].strip())

    probe_to_gene = dict(zip(annot_df["ID"], annot_df["Gene symbol"]))
    df["gene"] = df.index.map(probe_to_gene)
    df = df.dropna(subset=["gene"]).set_index("gene")
    df_genes = df.groupby(df.index).mean()

    # Match common samples
    common = df_genes.columns.intersection(label_series.index)
    df_genes = df_genes[common]
    label_series = label_series[common]

    # Log2-transform raw intensities if needed
    if df_genes.values.max() > 50:
        df_genes = np.log2(df_genes + 1)

    logger.info(f"Parsed real GEO dataset: {df_genes.shape[0]} unique genes × {df_genes.shape[1]} samples")
    return df_genes, label_series


def _create_synthetic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Create a realistic synthetic expression dataset when GEO is unavailable.
    """
    logger.warning("Creating synthetic dataset for demonstration (GEO unavailable)")

    np.random.seed(config.RANDOM_SEED)

    n_tumor, n_normal = 60, 60
    n_samples = n_tumor + n_normal
    n_genes = 5000

    known = config.KNOWN_MARKERS
    other_genes = [f"GENE_{i}" for i in range(n_genes - len(known))]
    all_genes = known + other_genes

    baseline = np.random.normal(loc=7.0, scale=2.0, size=(len(all_genes), n_samples))
    baseline = np.clip(baseline, 1, 16)

    # Add differential expression for known markers
    de_effects = {"EGFR": 2.5, "KRAS": 1.5, "MKI67": 2.2, "TOP2A": 2.0, "NKX2-1": 3.0, "NAPSA": 2.8}
    for gene, effect in de_effects.items():
        if gene in all_genes:
            idx = all_genes.index(gene)
            noise = np.random.normal(0, 0.5, n_tumor)
            baseline[idx, :n_tumor] += effect + noise

    # Add ~200 random DE genes
    random_de_indices = np.random.choice(range(len(known), len(all_genes)), size=200, replace=False)
    for idx in random_de_indices:
        effect = np.random.choice([-1, 1]) * np.random.uniform(1.0, 3.0)
        noise = np.random.normal(0, 0.6, n_tumor)
        baseline[idx, :n_tumor] += effect + noise

    sample_names = [f"Tumor_{i+1}" for i in range(n_tumor)] + [f"Normal_{i+1}" for i in range(n_normal)]
    expr_df = pd.DataFrame(baseline, index=all_genes, columns=sample_names)
    labels = pd.Series(["Tumor"] * n_tumor + ["Normal"] * n_normal, index=sample_names, name="condition")
    return expr_df, labels


def load_data(force_synthetic: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load expression dataset configured in config.GEO_ACCESSION.

    Tries to load from cache first, then downloads from GEO.
    Falls back to synthetic data if GEO is unavailable.

    Returns
    -------
    expression_df : pd.DataFrame
        Genes (rows) × Samples (columns), log2-scale expression values.
    labels : pd.Series
        Sample labels ('Tumor' or 'Normal'), indexed by sample name.
    """
    cache_expr = os.path.join(config.DATA_DIR, "expression_matrix.csv")
    cache_labels = os.path.join(config.DATA_DIR, "sample_labels.csv")

    # ── Try cache first ──────────────────────────────────────
    if os.path.exists(cache_expr) and os.path.exists(cache_labels) and not force_synthetic:
        logger.info("Loading cached data...")
        expr_df = pd.read_csv(cache_expr, index_col=0)
        labels = pd.read_csv(cache_labels, index_col=0).squeeze()
        logger.info(f"Loaded: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")
        return expr_df, labels

    # ── Try real GEO series matrix ───────────────────────────
    if not force_synthetic:
        try:
            expr_df, labels = _load_from_series_matrix()
            expr_df.to_csv(cache_expr)
            labels.to_frame().to_csv(cache_labels)
            logger.info("Real GEO data cached for future runs")
            return expr_df, labels
        except Exception as e:
            logger.warning(f"Failed to load real GEO series matrix: {e}")
            logger.info("Falling back to synthetic data...")

    # ── Fallback: Synthetic data ─────────────────────────────
    expr_df, labels = _create_synthetic_dataset()
    expr_df.to_csv(cache_expr)
    labels.to_frame().to_csv(cache_labels)
    logger.info(f"Synthetic data: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")
    return expr_df, labels


if __name__ == "__main__":
    expr, labels = load_data()
    print(f"\nExpression matrix: {expr.shape}")
    print(f"Labels:\n{labels.value_counts()}")
    print(f"\nFirst 5 genes: {expr.index[:5].tolist()}")
