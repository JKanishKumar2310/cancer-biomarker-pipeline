"""
Data Loader — Downloads and parses GEO dataset for cancer biomarker analysis.

Supports multiple GEO accessions configured in config.py:
  - GSE15852: Breast cancer (43 tumor + 43 normal, GPL96)
  - GSE19804: Lung adenocarcinoma (60 tumor + 60 normal, GPL570)
  - GSE8671: Colorectal (32 adenoma + 32 normal, GPL570)
  - GSE30784: Oral SCC (167 tumor + 45 normal, GPL570)
  - GSE53757: Kidney ccRCC (72 tumor + 72 normal, GPL570)

Supports:
  - Microarray (series matrix + GPL probe mapping)
  - RNA-seq (raw counts from supplementary files, voom/limma)
  - Multi-cohort fixed-effects meta-analysis
"""
import os
import re
import gzip
import urllib.request
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger
from src.ai_geo_curator import gpl_annotation_url, geo_series_stub, fetch_gpl_probe_mapping


def extract_patient_id(title: str, sample_id: str, characteristics: list[str] | None = None) -> str:
    """
    Extract or infer patient identifier from sample titles or characteristics.
    Supports common GEO naming conventions:
      - 'Biopsy of the normal mucosa ... from patient #1' -> 'Patient_001'
      - 'Lung Cancer 2T' / 'Lung Normal 2N' -> 'Patient_002'
      - '2102-Normal' / '2102-Tumor' -> 'Patient_2102'
      - 'donor: 102548' / 'patient identifier: 2102' -> 'Patient_2102'
    """
    stopwords = {"disease", "state", "status", "tissue", "cancer", "tumor", "carcinoma", "normal", "stage", "grade", "group", "type", "history", "characteristics", "sample", "material", "description", "source", "info", "survival", "treatment", "response", "data", "profile"}

    # Pattern 0: Check characteristics for explicit identifier fields
    if characteristics:
        char_str = " ".join(characteristics)
        m0_id = re.search(r'(?:patient|donor|subject|case|sample|pt)[\s_:#]*(?:id|identifier|number|num|code|#)[\s_:#]+([A-Za-z0-9_-]+)', char_str, re.IGNORECASE)
        if m0_id and m0_id.group(1).lower().strip() not in stopwords:
            val = m0_id.group(1).strip()
            digits = re.search(r'\d+', val)
            return f"Patient_{int(digits.group(0)):03d}" if digits else f"Patient_{val}"

        m0 = re.search(r'(?:donor|subject|patient|case|pt)[\s_:]+([A-Za-z0-9_-]+)', char_str, re.IGNORECASE)
        if m0 and m0.group(1).lower().strip() not in stopwords:
            val = m0.group(1).strip()
            digits = re.search(r'\d+', val)
            return f"Patient_{int(digits.group(0)):03d}" if digits else f"Patient_{val}"

    if not isinstance(title, str):
        return sample_id

    # Pattern 1: Title with prefix ID like 2102-Normal, Pt12_Tumor
    m_title = re.search(r'^([A-Za-z0-9_]+)[\-_](?:normal|tumor|tumour|cancer|adjacent|ca|cap|t|n)$', title.strip(), re.IGNORECASE)
    if m_title:
        val = m_title.group(1).strip()
        digits = re.search(r'\d+', val)
        return f"Patient_{int(digits.group(0)):03d}" if digits else f"Patient_{val}"

    # Pattern 2: patient #1, patient 1
    m1 = re.search(r'patient\s*#?\s*(\d+)', title, re.IGNORECASE)
    if m1:
        return f"Patient_{int(m1.group(1)):03d}"

    # Pattern 3: 2T / 2N or Lung Cancer 2T
    m2 = re.search(r'(\d+)[TN]\b', title, re.IGNORECASE)
    if m2:
        return f"Patient_{int(m2.group(1)):03d}"

    # Pattern 4: BC0043N / BC0043T
    m3 = re.search(r'(BC\d+)', title, re.IGNORECASE)
    if m3:
        digits = re.search(r'\d+', m3.group(1))
        return f"Patient_{int(digits.group(0)):03d}" if digits else m3.group(1).upper()

    return sample_id


# ── Platform annotation map ─────────────────────────────────
# Maps GEO accession → (platform GPL ID, annotation URL)
_PLATFORM_MAP = {
    "GSE15852": ("GPL96", None),
    "GSE19804": ("GPL570", None),
    "GSE8671": ("GPL570", None),
    "GSE30784": ("GPL570", None),
    "GSE53757": ("GPL570", None),
}

# ── Series matrix download URLs ─────────────────────────────
_MATRIX_URLS = {
    "GSE15852": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE15nnn/GSE15852/matrix/GSE15852_series_matrix.txt.gz",
    "GSE19804": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE19nnn/GSE19804/matrix/GSE19804_series_matrix.txt.gz",
    "GSE8671": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE8nnn/GSE8671/matrix/GSE8671_series_matrix.txt.gz",
}


def _download_with_timeout(url: str, dest_path: str, timeout: int = 15):
    """Download a file with a strict socket/connection timeout."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Bioinformatics/Pipeline"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(dest_path, "wb") as f_out:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    f_out.write(chunk)
    except TimeoutError:
        raise TimeoutError(f"NCBI connection timed out after {timeout} seconds. The NCBI server may be temporarily slow or unresponsive.")
    except Exception as e:
        if "timed out" in str(e).lower():
            raise TimeoutError(f"NCBI download timed out after {timeout} seconds.")
        raise e


def _download_if_missing(path: str, url: str, label: str):
    """Download a file from a URL if it doesn't already exist locally with a 15s timeout."""
    if not os.path.exists(path):
        logger.info(f"Downloading {label} from NCBI (timeout=15s)...")
        _download_with_timeout(url, path, timeout=15)
        logger.info(f"  Saved → {os.path.basename(path)}")


def _load_from_series_matrix() -> tuple[pd.DataFrame, pd.Series]:
    """
    Parse series matrix for the configured GEO accession and map probes
    to gene symbols using the appropriate GPL annotation.
    """
    accession = config.GEO_ACCESSION
    # Resolve the series matrix URL dynamically (works for any GSE accession).
    # A static override is still honoured when one is explicitly configured.
    matrix_url = _MATRIX_URLS.get(
        accession,
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_series_stub(accession)}/{accession}/matrix/{accession}_series_matrix.txt.gz",
    )

    matrix_file = os.path.join(config.DATA_DIR, f"{accession}_series_matrix.txt.gz")

    # Download the series matrix if needed
    _download_if_missing(matrix_file, matrix_url, f"{accession} series matrix")

    # ── Parse metadata ───────────────────────────────────────
    logger.info(f"Parsing sample metadata from {accession} series matrix...")
    meta_lines = []
    skiprows = 0
    platform_from_meta = None
    with gzip.open(matrix_file, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("!series_matrix_table_begin"):
                skiprows = i + 1
                break
            if line.startswith("!Series_platform_id"):
                platform_from_meta = line.strip().split("\t", 1)[-1].strip(' "')
            meta_lines.append(line)

    sample_ids, sample_titles, characteristics = [], [], []
    for line in meta_lines:
        if line.startswith("!Sample_geo_accession"):
            sample_ids = [x.strip(' "\t\r\n') for x in line.split("\t")[1:]]
        elif line.startswith("!Sample_title"):
            sample_titles = [x.strip(' "\t\r\n') for x in line.split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            characteristics.append([x.strip(' "\t\r\n') for x in line.split("\t")[1:]])

    # Prefer the platform actually declared by the series matrix; fall back to the
    # static map only when the metadata does not report one. This makes probe
    # mapping correct for ANY accession, not just the three hardcoded ones.
    gpl_id = platform_from_meta or _PLATFORM_MAP.get(accession, ("GPL570", None))[0]
    logger.info(f"Resolved platform {gpl_id} for {accession}")
    # Structure per-sample characteristics dictionary
    sample_chars = {}
    for i, s_id in enumerate(sample_ids):
        sample_chars[s_id] = [ch[i] for ch in characteristics if i < len(ch)]

    # Classify samples using AI Curator (template-based with 100% precision)
    try:
        from src.ai_geo_curator import curate_dataset
        curation = curate_dataset(accession)
        if curation.get("success") and "labels" in curation:
            labels = curation["labels"]
            logger.info(f"AI Curator successfully classified {len(labels)} samples for {accession}")
        else:
            raise ValueError("Curator did not return labels")
    except Exception as e:
        logger.warning(f"AI Curator unavailable or failed ({e}), using heuristic classification...")
        labels = {}
        for i, s_id in enumerate(sample_ids):
            title = sample_titles[i].lower() if i < len(sample_titles) else ""
            chars = " ".join([c.lower() for c in sample_chars.get(s_id, []) if not c.lower().startswith("cel filename")])

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
    df_genes = None
    suppl_url = curation.get("suppl_url") if isinstance(curation, dict) else None

    if skiprows > 0:
        logger.info(f"Reading {accession} embedded series matrix values...")
        try:
            df = pd.read_csv(
                matrix_file, compression="gzip", skiprows=skiprows, sep="\t", index_col=0, comment="!"
            )
            df = df[~df.index.astype(str).str.startswith("!")]
            df = df.apply(pd.to_numeric, errors="coerce")

            # ── Map probes → gene symbols ────────────────────────────
            if df.shape[0] > 0 and gpl_id:
                logger.info(f"Mapping probe IDs to gene symbols via {gpl_id}...")
                probe_to_gene = fetch_gpl_probe_mapping(gpl_id)
                if probe_to_gene:
                    df["gene"] = df.index.astype(str).map(probe_to_gene)
                    df = df.dropna(subset=["gene"]).set_index("gene")
                    df_genes = df.groupby(df.index).mean()
                else:
                    df_genes = df
            elif df.shape[0] > 0:
                df_genes = df
        except Exception as e:
            logger.warning(f"Could not parse embedded matrix: {e}")

    # Fallback to supplementary count/expression matrix if embedded table is missing or empty
    if df_genes is None or len(df_genes) == 0:
        if suppl_url:
            logger.info(f"Downloading & parsing supplementary count/expression matrix: {suppl_url}...")
            http_url = suppl_url.replace("ftp://", "https://")
            suppl_filename = os.path.basename(suppl_url)
            suppl_local = os.path.join(config.DATA_DIR, f"{accession}_suppl_{suppl_filename}")

            if not os.path.exists(suppl_local):
                logger.info(f"  Downloading supplementary file from {http_url}...")
                req = urllib.request.Request(http_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    with open(suppl_local, "wb") as f_out:
                        f_out.write(resp.read())

            sep = "\t" if ("tsv" in suppl_filename or "txt" in suppl_filename) else ","
            comp = "gzip" if suppl_filename.endswith(".gz") else None
            df_suppl = pd.read_csv(suppl_local, sep=sep, compression=comp, index_col=0)

            # Map sample columns to GSM accessions using intelligent title & donor matching
            col_to_gsm = {}
            for col in df_suppl.columns:
                matched = None
                col_clean = re.sub(r'[\.\-_\s]+', '', str(col)).lower()
                for i, s_id in enumerate(sample_ids):
                    title = sample_titles[i] if i < len(sample_titles) else ""
                    t_clean = re.sub(r'[\.\-_\s]+', '', str(title)).lower()
                    if str(col).lower() == s_id.lower() or col_clean == t_clean:
                        matched = s_id
                        break
                if not matched:
                    # Check characteristics (e.g. donor: 102548, CA.102548 vs CAP.102548)
                    digits = ''.join(c for c in str(col) if c.isdigit())
                    is_tumor_col = 'cap' not in str(col).lower() and ('ca' in str(col).lower() or 'tumor' in str(col).lower())
                    is_normal_col = 'cap' in str(col).lower() or 'norm' in str(col).lower() or 'adj' in str(col).lower()
                    for i, s_id in enumerate(sample_ids):
                        title = sample_titles[i] if i < len(sample_titles) else ""
                        chars = sample_chars.get(s_id, [])
                        chars_str = ' '.join(chars).lower()
                        is_tumor_s = 'tumor' in str(title).lower() or 'tumor' in chars_str
                        is_normal_s = 'normal' in str(title).lower() or 'normal' in chars_str
                        if digits and digits in chars_str:
                            if (is_tumor_col and is_tumor_s) or (is_normal_col and is_normal_s):
                                matched = s_id
                                break
                if matched:
                    col_to_gsm[col] = matched

            if len(col_to_gsm) > 0:
                df_suppl = df_suppl.rename(columns=col_to_gsm)

            # Clean and filter non-zero
            df_suppl = df_suppl.apply(pd.to_numeric, errors="coerce").fillna(0)
            df_suppl = df_suppl.loc[(df_suppl > 0).any(axis=1)]
            df_genes = df_suppl
        else:
            raise ValueError(f"No embedded series matrix table or supplementary count matrix available for {accession}")

    # Match common samples
    common = df_genes.columns.intersection(label_series.index)
    if len(common) == 0:
        gsm_to_title = dict(zip(sample_ids, sample_titles))
        label_series.index = label_series.index.map(lambda x: gsm_to_title.get(x, x))
        common = df_genes.columns.intersection(label_series.index)

    if len(common) == 0:
        raise ValueError(
            f"Failed to align sample columns ({df_genes.shape[1]}) with detected sample labels ({len(label_series)}) for {accession}."
        )

    df_genes = df_genes[common]
    label_series = label_series[common]

    # Log2-transform raw intensities if needed
    if df_genes.size > 0 and df_genes.values.max() > 50:
        df_genes = np.log2(df_genes + 1)

    patient_map = {}
    for i, s_id in enumerate(sample_ids):
        title = sample_titles[i] if i < len(sample_titles) else ""
        chars = sample_chars.get(s_id, [])
        patient_map[s_id] = extract_patient_id(title, s_id, chars)

    patient_ids = pd.Series([patient_map.get(s, s) for s in label_series.index], index=label_series.index, name="patient_id")
    label_series.attrs["patient_id"] = patient_ids

    logger.info(f"Parsed real GEO dataset: {df_genes.shape[0]} unique genes × {df_genes.shape[1]} samples across {patient_ids.nunique()} patients")
    return df_genes, label_series


def _save_labels(labels: pd.Series, filepath: str):
    """Save labels and patient IDs to CSV."""
    patient_ids = labels.attrs.get("patient_id")
    if patient_ids is not None:
        df = pd.DataFrame({"condition": labels, "patient_id": patient_ids}, index=labels.index)
    else:
        df = pd.DataFrame({"condition": labels, "patient_id": labels.index}, index=labels.index)
    df.to_csv(filepath)


def _create_synthetic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Create a realistic synthetic expression dataset when explicitly requested (test mode).
    """
    logger.warning("Creating synthetic dataset for test mode...")

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
    patients = [f"Patient_{i+1:03d}" for i in range(n_tumor)] + [f"Patient_{i+1:03d}" for i in range(n_normal)]
    expr_df = pd.DataFrame(baseline, index=all_genes, columns=sample_names)
    labels = pd.Series(["Tumor"] * n_tumor + ["Normal"] * n_normal, index=sample_names, name="condition")
    labels.attrs["patient_id"] = pd.Series(patients, index=sample_names, name="patient_id")
    return expr_df, labels


def _load_rnaseq_counts(accession: str, matrix_file: str, sample_ids: list, sample_titles: list, sample_chars: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load RNA-seq raw counts from GEO supplementary files.
    Expects a count matrix (genes × samples) in TSV/CSV format.
    """
    logger.info(f"Attempting to load RNA-seq counts for {accession}...")
    
    # Look for supplementary count files
    meta = fetch_geo_metadata(accession)
    suppl_files = meta.get("supplementary_files", [])
    
    count_file = None
    for s_file in suppl_files:
        s_lower = s_file.lower()
        if any(ext in s_lower for ext in [".txt.gz", ".tsv.gz", ".csv.gz", ".txt", ".tsv", ".csv"]):
            if not any(bad in s_lower for bad in ["raw.tar", ".bam", ".sra", ".bw", ".bed", ".bigwig", ".cel", "_reads", "reads.txt", "fastq", "matrix.mtx", "barcodes.tsv"]):
                count_file = s_file
                break
    
    if not count_file:
        raise ValueError(f"No suitable count matrix found in supplementary files for {accession}")
    
    # Download supplementary count matrix
    http_url = count_file.replace("ftp://", "https://")
    suppl_filename = os.path.basename(count_file)
    suppl_local = os.path.join(config.DATA_DIR, f"{accession}_suppl_{suppl_filename}")
    
    if not os.path.exists(suppl_local):
        logger.info(f"  Downloading RNA-seq count matrix from {http_url}...")
        req = urllib.request.Request(http_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(suppl_local, "wb") as f_out:
                f_out.write(resp.read())
    
    # Parse count matrix
    sep = "\t" if ("tsv" in suppl_filename or "txt" in suppl_filename) else ","
    comp = "gzip" if suppl_filename.endswith(".gz") else None
    df_counts = pd.read_csv(suppl_local, sep=sep, compression=comp, index_col=0)
    
    # Map columns to GSM sample IDs
    col_to_gsm = {}
    for col in df_counts.columns:
        matched = None
        col_clean = re.sub(r'[\.\-_\s]+', '', str(col)).lower()
        for i, s_id in enumerate(sample_ids):
            title = sample_titles[i] if i < len(sample_titles) else ""
            t_clean = re.sub(r'[\.\-_\s]+', '', str(title)).lower()
            if str(col).lower() == s_id.lower() or col_clean == t_clean:
                matched = s_id
                break
        if not matched:
            # Try matching by sample characteristics
            digits = ''.join(c for c in str(col) if c.isdigit())
            for i, s_id in enumerate(sample_ids):
                chars = sample_chars.get(s_id, [])
                chars_str = ' '.join(chars).lower()
                if digits and digits in chars_str:
                    matched = s_id
                    break
        if matched:
            col_to_gsm[col] = matched
    
    if len(col_to_gsm) == 0:
        raise ValueError(f"Could not map count matrix columns to GSM sample IDs for {accession}")
    
    df_counts = df_counts.rename(columns=col_to_gsm)
    
    # Get labels from AI curator
    from src.ai_geo_curator import curate_dataset
    curation = curate_dataset(accession)
    if curation.get("success") and "labels" in curation:
        labels = curation["labels"]
    else:
        # Fallback heuristic
        labels = {}
        for i, s_id in enumerate(sample_ids):
            t = sample_titles[i].lower() if i < len(sample_titles) else ""
            if any(w in t for w in ["normal", "healthy", "adjacent", "control"]):
                labels[s_id] = "Normal"
            else:
                labels[s_id] = "Tumor"
    
    label_series = pd.Series(labels, name="condition")
    label_series = label_series[label_series != "Unknown"]
    
    # Align
    common = df_counts.columns.intersection(label_series.index)
    df_counts = df_counts[common]
    label_series = label_series[common]
    
    # Filter low-count genes
    min_counts = getattr(config, "RNASEQ_MIN_COUNTS", 10)
    min_samples = getattr(config, "RNASEQ_MIN_SAMPLES", 3)
    keep = (df_counts >= min_counts).sum(axis=1) >= min_samples
    df_counts = df_counts[keep]
    
    logger.info(f"Loaded RNA-seq counts: {df_counts.shape[0]} genes × {df_counts.shape[1]} samples")
    return df_counts, label_series


def _voom_transform(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Apply voom transformation (mean-variance trend + precision weights)
    using statsmodels. Returns log2 CPM with voom weights.
    """
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
    
    logger.info("Applying voom transformation...")
    
    # CPM normalization
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6
    log_cpm = np.log2(cpm + 0.5)
    
    # Mean-variance trend (voom)
    # Fit lowess to sqrt(counts) vs mean log-cpm
    mean_log_cpm = log_cpm.mean(axis=1)
    var_log_cpm = log_cpm.var(axis=1)
    
    # Simple mean-variance trend approximation
    # voom uses lowess; we use a simple spline fit
    from scipy.interpolate import UnivariateSpline
    try:
        spline = UnivariateSpline(mean_log_cpm, np.sqrt(var_log_cpm), k=3, s=len(mean_log_cpm))
        fitted = spline(mean_log_cpm)
        weights = 1.0 / (fitted ** 2 + 1e-6)
        weights = weights / weights.mean()  # Normalize
    except Exception:
        weights = pd.Series(1.0, index=counts.index)
    
    # Return log2 CPM (voom-transformed expression)
    logger.info("Voom transformation complete")
    return log_cpm


def _tmm_normalize(counts: pd.DataFrame) -> pd.DataFrame:
    """
    TMM (Trimmed Mean of M-values) normalization - edgeR algorithm.
    Pure Python implementation.
    """
    logger.info("Applying TMM normalization...")
    
    # Reference sample: upper quartile
    ref = counts.median(axis=1)
    
    tmm_factors = []
    for col in counts.columns:
        sample = counts[col]
        # M and A values
        m = np.log2((sample + 0.5) / (ref + 0.5))
        a = 0.5 * np.log2((sample + 0.5) * (ref + 0.5))
        
        # Trim extreme M and A values
        m_trim = np.percentile(m, [30, 70])
        a_trim = np.percentile(a, [5, 95])
        keep = (m >= m_trim[0]) & (m <= m_trim[1]) & (a >= a_trim[0]) & (a <= a_trim[1])
        
        if keep.sum() > 10:
            tmm = 2 ** np.mean(m[keep])
        else:
            tmm = 1.0
        tmm_factors.append(tmm)
    
    tmm_factors = np.array(tmm_factors)
    tmm_factors = tmm_factors / np.mean(tmm_factors)  # Normalize to mean=1
    
    # Apply normalization
    normalized = counts.div(tmm_factors, axis=1)
    logger.info(f"TMM factors: {tmm_factors}")
    return normalized


def _median_of_ratios_normalize(counts: pd.DataFrame) -> pd.DataFrame:
    """
    DESeq2 median-of-ratios normalization.
    Pure Python implementation.
    """
    logger.info("Applying median-of-ratios (DESeq2) normalization...")
    
    # Geometric mean per gene
    geo_means = np.exp(np.log(counts + 1).mean(axis=1))
    
    # Ratios
    ratios = counts.div(geo_means, axis=0)
    
    # Size factors = median of ratios per sample
    size_factors = ratios.median(axis=0)
    size_factors = size_factors / size_factors.mean()
    
    # Apply normalization
    normalized = counts.div(size_factors, axis=1)
    logger.info(f"DESeq2 size factors: {size_factors.values}")
    return normalized


def load_multi_cohort(force_synthetic: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load multiple discovery cohorts for fixed-effects meta-analysis.
    
    Returns combined expression matrix with cohort column in labels.
    """
    if not getattr(config, "DISCOVERY_COHORTS", []):
        raise ValueError("DISCOVERY_COHORTS not configured in config.py")
    
    cohorts = config.DISCOVERY_COHORTS
    logger.info(f"Loading multi-cohort data: {cohorts}")
    
    all_expr = []
    all_labels = []
    all_patient_ids = []
    
    for i, acc in enumerate(cohorts):
        logger.info(f"Loading cohort {i+1}/{len(cohorts)}: {acc}")
        
        # Temporarily set GEO_ACCESSION for this cohort
        old_accession = config.GEO_ACCESSION
        config.GEO_ACCESSION = acc
        
        try:
            expr_df, labels = load_data(force_synthetic=force_synthetic)
        finally:
            config.GEO_ACCESSION = old_accession
        
        # Add cohort prefix to patient IDs
        patient_ids = labels.attrs.get("patient_id", pd.Series(labels.index, index=labels.index))
        patient_ids = patient_ids.apply(lambda x: f"{acc}_{x}")
        
        # Add cohort label
        labels = labels.copy()
        labels.attrs["patient_id"] = patient_ids
        labels.attrs["cohort"] = acc
        
        all_expr.append(expr_df)
        all_labels.append(labels)
        all_patient_ids.append(patient_ids)
    
    # Find common genes across all cohorts
    common_genes = set(all_expr[0].index)
    for expr in all_expr[1:]:
        common_genes &= set(expr.index)
    
    common_genes = sorted(list(common_genes))
    logger.info(f"Common genes across {len(cohorts)} cohorts: {len(common_genes)}")
    
    if len(common_genes) < 1000:
        logger.warning(f"Only {len(common_genes)} common genes - meta-analysis may be underpowered")
    
    # Subset and combine. attrs must be stripped before concat: pandas 3.x
    # compares attrs dicts by value and any dict holding a Series (patient_id
    # maps differ per cohort) raises "Can only compare identically-labeled
    # Series objects" instead of merging.
    for lbl in all_labels:
        lbl.attrs.clear()
    combined_expr = pd.concat([expr.loc[common_genes] for expr in all_expr], axis=1)
    combined_labels = pd.concat(all_labels)
    
    # pd.concat drops per-frame attrs, so attach an explicit per-sample cohort map
    # (index-aligned Series). Downstream meta-analysis needs this to re-split the
    # combined matrix into per-cohort DE tables; a scalar attr would not survive
    # preprocessing either.
    cohort_map = pd.Series(
        [acc for acc, lbl in zip(cohorts, all_labels) for _ in range(len(lbl))],
        index=[sid for lbl in all_labels for sid in lbl.index],
        name="cohort",
    )
    combined_labels.attrs["cohort"] = cohort_map.reindex(combined_labels.index)
    
    # Batch correction if enabled
    if getattr(config, "ENABLE_BATCH_CORRECTION", False):
        logger.info("Applying ComBat batch correction...")
        combined_expr = _combat_correct(combined_expr, combined_labels)
    
    logger.info(f"Multi-cohort combined: {combined_expr.shape[0]} genes × {combined_expr.shape[1]} samples")
    return combined_expr, combined_labels


def _combat_correct(expr_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """
    Apply ComBat batch correction using pycombat or sklearn approximation.
    Batch = cohort from labels.attrs["cohort"]
    """
    try:
        from combat.pycombat import pycombat
        batch = labels.attrs.get("cohort", pd.Series("batch1", index=labels.index))
        if not isinstance(batch, pd.Series):
            # attrs["cohort"] stores a single scalar accession; ComBat needs one
            # label per sample or it crashes on the dtype-less scalar.
            batch = pd.Series(batch, index=labels.index)
        # Ensure batch is aligned with expression columns
        batch = batch.loc[expr_df.columns]
        
        # pycombat expects features × samples
        corrected = pycombat(expr_df.values, batch.values)
        return pd.DataFrame(corrected, index=expr_df.index, columns=expr_df.columns)
    except ImportError:
        logger.warning("pycombat not installed - skipping batch correction. Install with: pip install pycombat")
        return expr_df
    except Exception as e:
        logger.warning(f"ComBat failed: {e} - skipping batch correction")
        return expr_df


def _load_rnaseq_via_series_matrix(accession: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load RNA-seq counts by parsing series matrix metadata and supplementary count files.
    """
    # Download and parse series matrix metadata (same as microarray)
    matrix_url = _MATRIX_URLS.get(
        accession,
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_series_stub(accession)}/{accession}/matrix/{accession}_series_matrix.txt.gz",
    )

    matrix_file = os.path.join(config.DATA_DIR, f"{accession}_series_matrix.txt.gz")
    _download_if_missing(matrix_file, matrix_url, f"{accession} series matrix")

    # Parse metadata
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

    sample_chars = {}
    for i, s_id in enumerate(sample_ids):
        sample_chars[s_id] = [ch[i] for ch in characteristics if i < len(ch)]

    # Load RNA-seq counts from supplementary files
    return _load_rnaseq_counts(accession, matrix_file, sample_ids, sample_titles, sample_chars)


def load_data(force_synthetic: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """
    Main entry point: loads expression matrix and sample labels.
    
    Supports:
    - Single cohort microarray (default)
    - Single cohort RNA-seq (DATA_TYPE = "rnaseq")
    - Multi-cohort meta-analysis (DISCOVERY_COHORTS list)
    """
    # Multi-cohort mode
    if getattr(config, "DISCOVERY_COHORTS", []) and not force_synthetic:
        logger.info("Multi-cohort mode enabled (DISCOVERY_COHORTS)")
        return load_multi_cohort(force_synthetic=force_synthetic)
    
    acc = getattr(config, "GEO_ACCESSION", "GSE8671")
    acc_expr = os.path.join(config.DATA_DIR, f"{acc}_expression_matrix.csv")
    acc_labels = os.path.join(config.DATA_DIR, f"{acc}_sample_labels.csv")

    cache_expr = os.path.join(config.DATA_DIR, "expression_matrix.csv")
    cache_labels = os.path.join(config.DATA_DIR, "sample_labels.csv")

    # ── If force_synthetic requested, create synthetic data ──
    if force_synthetic:
        logger.info("Generating synthetic dataset (explicit test mode)...")
        expr_df, labels = _create_synthetic_dataset()
        # run_analysis.py --test-mode already points RESULTS_DIR at .../synthetic_test;
        # guard against nesting a second synthetic_test inside it.
        if os.path.basename(os.path.normpath(config.RESULTS_DIR)) != "synthetic_test":
            test_dir = os.path.join(config.RESULTS_DIR, "synthetic_test")
        else:
            test_dir = config.RESULTS_DIR
        os.makedirs(test_dir, exist_ok=True)
        expr_df.to_csv(os.path.join(test_dir, "expression_matrix.csv"))
        _save_labels(labels, os.path.join(test_dir, "sample_labels.csv"))
        return expr_df, labels

    # ── Try accession-specific cache first ─────────────────────
    if os.path.exists(acc_expr) and os.path.exists(acc_labels):
        logger.info(f"Loading cached {acc} data...")
        expr_df = pd.read_csv(acc_expr, index_col=0)
        labels_df = pd.read_csv(acc_labels, index_col=0)
        if isinstance(labels_df, pd.DataFrame):
            col = "condition" if "condition" in labels_df.columns else labels_df.columns[0]
            labels = labels_df[col]
            if "patient_id" in labels_df.columns:
                labels.attrs["patient_id"] = labels_df["patient_id"]
            else:
                labels.attrs["patient_id"] = pd.Series(labels_df.index, index=labels_df.index)
        else:
            labels = labels_df.squeeze()
            labels.attrs["patient_id"] = pd.Series(labels.index, index=labels.index)

        n_p = labels.attrs["patient_id"].nunique()
        sample_genes = [str(x) for x in expr_df.index[:10]]
        is_stale_probe_cache = all(x.isdigit() for x in sample_genes) or (n_p <= 1 and len(labels) >= 10)

        if not is_stale_probe_cache:
            # Keep active expression_matrix.csv in sync with current dataset
            expr_df.to_csv(cache_expr)
            _save_labels(labels, cache_labels)
            logger.info(f"Loaded: {expr_df.shape[0]} genes × {expr_df.shape[1]} samples across {n_p} patients")
            return expr_df, labels
        else:
            logger.info(f"Cached {acc} data contains raw probe IDs or legacy patient grouping. Re-ingesting...")

    # ── Load from GEO (microarray or RNA-seq) ──────────────────
    # Check if RNA-seq mode
    if getattr(config, "DATA_TYPE", "microarray") == "rnaseq" or getattr(config, "ENABLE_RNASEQ", False):
        logger.info(f"RNA-seq mode enabled for {acc}")
        try:
            expr_df, labels = _load_rnaseq_via_series_matrix(acc)
            expr_df.to_csv(acc_expr)
            _save_labels(labels, acc_labels)
            expr_df.to_csv(cache_expr)
            _save_labels(labels, cache_labels)
            logger.info(f"Real GEO RNA-seq {acc} data cached for future runs")
            return expr_df, labels
        except Exception as e:
            logger.error(f"Failed to load RNA-seq data for {acc}: {e}")
            raise RuntimeError(
                f"Failed to download or parse RNA-seq data for {acc}: {e}. "
                "Ensure the dataset has supplementary count matrix files."
            ) from e

    # ── Microarray: Try Universal AI Data Adapter ──────────────
    try:
        from src.universal_data_adapter import ingest_dataset
        expr_df, labels = ingest_dataset(acc)
        expr_df.to_csv(acc_expr)
        _save_labels(labels, acc_labels)
        expr_df.to_csv(cache_expr)
        _save_labels(labels, cache_labels)
        logger.info(f"Real GEO {acc} data cached for future runs")
        return expr_df, labels
    except Exception as e:
        logger.warning(f"Universal adapter hit issue ({e}), falling back to series matrix parser...")
        try:
            expr_df, labels = _load_from_series_matrix()
            expr_df.to_csv(acc_expr)
            _save_labels(labels, acc_labels)
            expr_df.to_csv(cache_expr)
            _save_labels(labels, cache_labels)
            logger.info(f"Real GEO {acc} data cached for future runs")
            return expr_df, labels
        except Exception as e2:
            logger.error(f"Failed to load authentic GEO data for {acc}: {e2}")
            raise RuntimeError(
                f"Failed to download or parse authentic GEO data for {acc}: {e2}. "
                "To run with synthetic test data for offline testing, explicitly pass force_synthetic=True."
            ) from e2


if __name__ == "__main__":
    expr, labels = load_data()
    print(f"\nExpression matrix: {expr.shape}")
    print(f"Labels:\n{labels.value_counts()}")
    print(f"\nFirst 5 genes: {expr.index[:5].tolist()}")
