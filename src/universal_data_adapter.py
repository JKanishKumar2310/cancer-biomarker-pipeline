"""
Universal AI Data Adapter — Ingests and standardizes arbitrary GEO and expression datasets.

Features:
1. Matrix Location Ingestion: Auto-detects embedded series matrix tables or supplementary
   files (.csv.gz, .tsv.gz, .txt.gz, .csv, .tsv, .txt, .xlsx).
2. Delimiter Sniffing: Auto-detects comma, tab, semicolon, or whitespace separators.
3. AI Schema & Sample Column Matcher: Uses OpenRouter LLM (or robust generalized fuzzy
   matcher) to map arbitrary matrix column headers to GSM accessions, conditions, and patient IDs.
4. Universal Gene Identifier Translation: Automatically handles Probe IDs (GPL mapping),
   Ensembl IDs (ENSG...), and Gene Symbols.
5. Dynamic Value Scale Normalization: Auto-detects raw counts (converts to log2 CPM),
   continuous TPM/FPKM (converts to log2(x+1)), or existing log2 intensities.
6. Intra-Patient Pairing Detection: Infers paired clinical samples for StratifiedGroupKFold
   and paired t-tests.
"""
import os
import sys
import gzip
import json
import re
import csv
import urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger
from src.ai_geo_curator import fetch_geo_metadata, _download_with_timeout


# ---------------------------------------------------------------------------
# Delimiter & Matrix Loading
# ---------------------------------------------------------------------------

def sniff_delimiter(filepath: str, is_gzip: bool = False) -> str:
    """Sniff the delimiter of a text file (comma, tab, semicolon, whitespace)."""
    try:
        if is_gzip:
            with gzip.open(filepath, "rt", encoding="utf-8", errors="ignore") as f:
                sample_lines = [f.readline() for _ in range(5)]
        else:
            with open(filepath, "rt", encoding="utf-8", errors="ignore") as f:
                sample_lines = [f.readline() for _ in range(5)]

        sample_text = "".join(sample_lines)
        # Check tab vs comma counts
        tab_count = sample_text.count("\t")
        comma_count = sample_text.count(",")
        semi_count = sample_text.count(";")

        if tab_count > comma_count and tab_count > semi_count:
            return "\t"
        if comma_count > tab_count and comma_count > semi_count:
            return ","
        if semi_count > tab_count and semi_count > comma_count:
            return ";"

        # Fallback to Sniffer
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text)
        return dialect.delimiter
    except Exception:
        return "\t" if ("tsv" in filepath.lower() or "txt" in filepath.lower()) else ","


def load_raw_supplementary_matrix(file_path: str) -> pd.DataFrame:
    """Load a supplementary expression matrix supporting csv, tsv, txt, xlsx, and gzip."""
    fn_lower = file_path.lower()
    if fn_lower.endswith(".xlsx") or fn_lower.endswith(".xls"):
        df = pd.read_excel(file_path, index_col=0)
    else:
        is_gz = fn_lower.endswith(".gz")
        sep = sniff_delimiter(file_path, is_gzip=is_gz)
        comp = "gzip" if is_gz else None
        df = pd.read_csv(file_path, sep=sep, compression=comp, index_col=0, low_memory=False)

    # If first column was not indexed properly or index contains duplicate empty values
    if df.index.name is None and df.columns[0].lower() in ["gene", "symbol", "id", "id_ref", "gene_symbol", "probe"]:
        df = df.set_index(df.columns[0])

    # Convert values to numeric, replace NaN with 0
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    # Remove all-zero rows
    df = df.loc[(df > 0).any(axis=1)]
    return df


# ---------------------------------------------------------------------------
# AI-Powered Column & Sample Mapping
# ---------------------------------------------------------------------------

def match_columns_with_ai(
    columns: list[str],
    samples: dict[str, dict],
    accession: str,
    series_title: str
) -> dict[str, dict] | None:
    """
    Query OpenRouter LLM to intelligently map arbitrary matrix column headers
    to GEO GSM accessions, sample condition (Tumor vs Normal), and Patient IDs.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    # Format compact sample metadata summary for prompt
    sample_summary = {}
    for s_id, s_data in list(samples.items())[:80]:
        t = s_data.get("title", "")
        chars = " | ".join(s_data.get("characteristics", [])[:4])
        sample_summary[s_id] = f"Title: {t} // {chars}"

    prompt = (
        f"You are an expert bioinformatician. Study GEO accession {accession}: '{series_title}'.\n"
        f"We downloaded an expression matrix with these column headers:\n"
        f"{json.dumps(columns[:80], indent=1)}\n\n"
        f"Here are the GEO sample metadata records (GSM accessions, titles, characteristics):\n"
        f"{json.dumps(sample_summary, indent=1)}\n\n"
        "Return a JSON object with key 'column_mappings', where each matrix column is mapped to:\n"
        "- 'sample_id': The corresponding GSM accession (e.g. 'GSM5574685')\n"
        "- 'condition': 'Tumor' or 'Normal' or 'Exclude'\n"
        "- 'patient_id': Inferred patient/subject identifier (e.g. 'Patient_01' or donor ID)\n\n"
        "Example response structure:\n"
        '{"column_mappings": {"CA.102548": {"sample_id": "GSM5574685", "condition": "Tumor", "patient_id": "Patient_102548"}}}'
    )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }

    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed.get("column_mappings")
    except Exception as e:
        logger.warning(f"AI column matching query failed or timed out: {e}. Using generalized fuzzy matcher.")
        return None


def match_columns_fuzzy(
    columns: list[str],
    samples: dict[str, dict]
) -> dict[str, dict]:
    """
    Generalized fuzzy matcher linking arbitrary column headers to GSM accessions,
    conditions, and patient IDs without any dataset-specific hardcoded strings.
    """
    mappings = {}
    tumor_tokens = {"tumor", "tumour", "cancer", "ca", "malignant", "carcinoma", "adenoma", "case", "primary", "disease"}
    normal_tokens = {"normal", "healthy", "cap", "adjacent", "control", "ctrl", "norm", "adj", "mucosa", "non-tumor", "nontumor"}

    sample_items = list(samples.items())

    for col in columns:
        col_str = str(col).strip()
        col_clean = re.sub(r'[\.\-_\s]+', '', col_str).lower()
        col_lower = col_str.lower()
        col_digits = "".join(c for c in col_str if c.isdigit())

        # Determine condition hint from column name
        is_normal_col = any(tok in col_lower for tok in normal_tokens)
        is_tumor_col = any(tok in col_lower for tok in tumor_tokens) and not is_normal_col

        matched_s_id = None
        matched_title = ""
        matched_chars = []

        # Pass 1: Direct exact match to GSM or Title
        for s_id, s_data in sample_items:
            t = s_data.get("title", "")
            t_clean = re.sub(r'[\.\-_\s]+', '', t).lower()
            if col_lower == s_id.lower() or col_clean == t_clean:
                matched_s_id = s_id
                matched_title = t
                matched_chars = s_data.get("characteristics", [])
                break

        # Pass 2: Numeric/Donor ID + Condition match across metadata
        if not matched_s_id and col_digits:
            for s_id, s_data in sample_items:
                t = s_data.get("title", "")
                chars = s_data.get("characteristics", [])
                full_text = f"{t} {' '.join(chars)}".lower()

                # Check if digits appear in full text (e.g. donor: 102548 or Pt 12)
                if col_digits in full_text:
                    is_norm_meta = any(tok in full_text for tok in normal_tokens)
                    is_tum_meta = any(tok in full_text for tok in tumor_tokens)

                    if (is_normal_col and is_norm_meta) or (is_tumor_col and is_tum_meta):
                        matched_s_id = s_id
                        matched_title = t
                        matched_chars = chars
                        break
                    elif not is_normal_col and not is_tumor_col:
                        matched_s_id = s_id
                        matched_title = t
                        matched_chars = chars
                        break

        # Pass 3: Token overlap
        if not matched_s_id:
            best_score = 0
            best_s_id = None
            col_words = set(re.findall(r'[a-zA-Z0-9]+', col_lower))
            for s_id, s_data in sample_items:
                t = s_data.get("title", "")
                meta_words = set(re.findall(r'[a-zA-Z0-9]+', f"{s_id} {t}".lower()))
                overlap = len(col_words.intersection(meta_words))
                if overlap > best_score:
                    best_score = overlap
                    best_s_id = s_id
                    matched_title = t
                    matched_chars = s_data.get("characteristics", [])
            if best_score >= 1:
                matched_s_id = best_s_id

        if matched_s_id:
            # Determine condition
            meta_str = f"{matched_title} {' '.join(matched_chars)}".lower()
            if any(w in meta_str for w in normal_tokens) or is_normal_col:
                cond = "Normal"
            elif any(w in meta_str for w in tumor_tokens) or is_tumor_col:
                cond = "Tumor"
            else:
                cond = "Tumor"  # Default assumption for oncology studies

            # Infer patient ID
            m_pat = re.search(r'(?:patient|donor|subject|case)[\s_:#]+([A-Za-z0-9_-]+)', meta_str, re.IGNORECASE)
            if m_pat:
                pat_val = m_pat.group(1).strip()
                digits = re.search(r'\d+', pat_val)
                pat_id = f"Patient_{int(digits.group(0)):03d}" if digits else f"Patient_{pat_val}"
            else:
                m_tn = re.search(r'\b(\d+)[tTnN]\b', meta_str)
                if m_tn:
                    pat_id = f"Patient_{int(m_tn.group(1)):03d}"
                elif col_digits:
                    pat_id = f"Patient_{int(col_digits):03d}"
                else:
                    pat_id = matched_s_id

            mappings[col_str] = {
                "sample_id": matched_s_id,
                "condition": cond,
                "patient_id": pat_id,
            }

    return mappings


# ---------------------------------------------------------------------------
# Universal Gene Identifier Translation
# ---------------------------------------------------------------------------

def translate_gene_identifiers(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """
    Harmonize matrix index into official HGNC Gene Symbols.
    Handles:
    - Probe IDs (GPL platform annotation)
    - Ensembl Gene IDs (ENSG...)
    - Existing Gene Symbols
    """
    idx_sample = [str(x).strip() for x in df.index[:25]]

    # Case 1: Ensembl Gene IDs (e.g. ENSG00000141510 or ENSG00000141510.12)
    if any(x.startswith("ENSG") for x in idx_sample):
        logger.info("Detected Ensembl Gene IDs in expression matrix. Cleaning IDs...")
        df.index = [str(x).split(".")[0].strip() for x in df.index]
        return df.groupby(df.index).mean()

    # Case 2: Microarray Probe IDs (e.g. 205239_at, 1007_s_at, ILMN_1725487)
    is_probe = any("_at" in x.lower() or "ilmn" in x.lower() for x in idx_sample) or all(x.isdigit() for x in idx_sample[:10])
    if is_probe and platform:
        logger.info(f"Detected microarray probe IDs. Mapping via platform {platform} annotation...")
        try:
            annot_path = os.path.join(config.DATA_DIR, f"{platform}.annot.gz")
            if not os.path.exists(annot_path):
                annot_url = f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{platform[:4]}nnn/{platform}/annot/{platform}.annot.gz"
                logger.info(f"Downloading {platform} platform annotation table...")
                _download_with_timeout(annot_url, annot_path, timeout=30)

            annot_skip = 0
            with gzip.open(annot_path, "rt", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if line.startswith("!platform_table_begin"):
                        annot_skip = i + 1
                        break

            annot_df = pd.read_csv(
                annot_path,
                sep="\t",
                skiprows=annot_skip,
                compression="gzip",
                low_memory=False,
                usecols=["ID", "Gene symbol"],
            )
            annot_df = annot_df.dropna(subset=["Gene symbol"])
            annot_df = annot_df[~annot_df["Gene symbol"].str.strip().isin(["", "---"])]
            annot_df["Gene symbol"] = annot_df["Gene symbol"].apply(lambda x: str(x).split("///")[0].strip())

            probe_to_gene = dict(zip(annot_df["ID"].astype(str), annot_df["Gene symbol"]))
            df["gene"] = df.index.astype(str).map(probe_to_gene)
            df = df.dropna(subset=["gene"]).set_index("gene")
            return df.groupby(df.index).mean()
        except Exception as e:
            logger.warning(f"Probe-to-gene translation failed: {e}. Keeping raw identifiers.")

    # Case 3: Gene Symbols (clean multi-mapping symbols like 'ACTB///ACTG1')
    clean_indices = [str(x).split("///")[0].strip() for x in df.index]
    df.index = clean_indices
    df = df[df.index != ""]
    return df.groupby(df.index).mean()


# ---------------------------------------------------------------------------
# Dynamic Scale & Value Normalization
# ---------------------------------------------------------------------------

def normalize_expression_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inspect expression matrix values and apply appropriate scale transformations:
    - Raw read counts (integers, max > 500) -> CPM normalization + log2(CPM + 1)
    - Continuous FPKM / TPM (floats, max > 50) -> log2(x + 1)
    - Existing log2 intensities (max <= 50) -> preserve without modification
    """
    if df.empty:
        return df

    max_val = df.values.max()

    # Check for raw integer counts
    sample_vals = df.iloc[:10, :min(5, df.shape[1])].values.flatten()
    is_mostly_integer = np.all(np.isclose(sample_vals, np.round(sample_vals))) and max_val > 500

    if is_mostly_integer:
        logger.info(f"Detected raw sequencing read counts (max={max_val:,.0f}). Computing log2(CPM + 1)...")
        lib_sizes = df.sum(axis=0)
        lib_sizes[lib_sizes == 0] = 1.0
        cpm = (df / lib_sizes) * 1e6
        return np.log2(cpm + 1)

    if max_val > 50:
        logger.info(f"Detected unlogged continuous expression values (max={max_val:.2f}). Applying log2(x + 1)...")
        return np.log2(df + 1)

    logger.info(f"Expression values are within standard log2 scale (max={max_val:.2f}). Skipping log2.")
    return df


# ---------------------------------------------------------------------------
# Main Universal Ingestion Function
# ---------------------------------------------------------------------------

def ingest_dataset(accession: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Autonomously ingest, parse, align, and normalize any GEO dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]:
        - expr_df: standardized gene expression DataFrame (genes x samples)
        - label_series: Series of Tumor/Normal condition labels, with attrs['patient_id']
    """
    accession = accession.strip().upper()
    logger.info(f"Universal AI Data Adapter ingesting: {accession}...")

    # Step 1: Fetch series matrix metadata
    meta = fetch_geo_metadata(accession)
    if not meta.get("success"):
        raise ValueError(f"Metadata ingestion failed for {accession}: {meta.get('error')}")

    samples = meta["samples"]
    platform = meta.get("platform", "GPL570")
    series_title = meta.get("series_title", "")
    local_path = meta.get("local_path")
    suppl_url = meta.get("suppl_url")

    df_genes = None

    # Step 2: Try reading embedded series matrix table
    if local_path and os.path.exists(local_path):
        try:
            with gzip.open(local_path, "rt", encoding="utf-8", errors="ignore") as f:
                matrix_lines = []
                in_table = False
                for line in f:
                    if line.startswith("!series_matrix_table_begin"):
                        in_table = True
                        continue
                    if line.startswith("!series_matrix_table_end"):
                        break
                    if in_table:
                        matrix_lines.append(line)

            if len(matrix_lines) > 2:
                from io import StringIO
                df = pd.read_csv(StringIO("".join(matrix_lines)), sep="\t", index_col=0, low_memory=False)
                df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
                if df.shape[0] > 0 and df.shape[1] > 0:
                    df_genes = df
                    logger.info(f"Parsed embedded series matrix table: {df.shape[0]} rows x {df.shape[1]} samples")
        except Exception as e:
            logger.warning(f"Could not parse embedded matrix: {e}")

    # Step 3: Fallback to supplementary count/expression matrix if needed
    if df_genes is None or len(df_genes) <= 1:
        if not suppl_url:
            raw_read_patterns = ["raw.tar", ".bam", ".sra", ".bw", ".bed", ".bigwig", ".cel", "_reads", "reads.txt", "fastq", "matrix.mtx", "barcodes.tsv"]
            for s_file in meta.get("supplementary_files", []):
                s_lower = s_file.lower()
                if any(ext in s_lower for ext in [".txt.gz", ".tsv.gz", ".csv.gz", ".txt", ".tsv", ".csv", ".xlsx", ".xls"]):
                    if not any(bad in s_lower for bad in raw_read_patterns):
                        suppl_url = s_file
                        break

        if suppl_url:
            logger.info(f"Downloading & parsing supplementary count/expression matrix: {suppl_url}...")
            http_url = suppl_url.replace("ftp://", "https://")
            suppl_filename = os.path.basename(suppl_url)
            suppl_local = os.path.join(config.DATA_DIR, f"{accession}_suppl_{suppl_filename}")

            if not os.path.exists(suppl_local):
                logger.info(f"  Downloading supplementary file from {http_url}...")
                _download_with_timeout(http_url, suppl_local, timeout=60)

            df_genes = load_raw_supplementary_matrix(suppl_local)
            logger.info(f"Parsed supplementary matrix: {df_genes.shape[0]} rows x {df_genes.shape[1]} columns")
        else:
            raise ValueError(
                f"No embedded series matrix table or supplementary count matrix available for {accession}."
            )

    # Step 4: Map and Align Sample Columns
    matrix_cols = list(df_genes.columns)
    logger.info("Aligning sample column headers with GEO metadata...")
    column_mappings = match_columns_fuzzy(matrix_cols, samples)

    # If fuzzy matching was incomplete (<80% coverage or missing comparative groups), query AI schema matcher
    matched_conds = {v.get("condition") for v in column_mappings.values()}
    if len(column_mappings) < len(matrix_cols) * 0.8 or len(matched_conds.intersection({"Tumor", "Normal"})) < 2:
        logger.info("Fuzzy column mapping incomplete. Querying AI schema matcher...")
        ai_mappings = match_columns_with_ai(matrix_cols, samples, accession, series_title)
        if ai_mappings:
            column_mappings.update(ai_mappings)

    # Build final aligned labels and rename columns
    col_rename = {}
    labels_dict = {}
    patient_dict = {}

    for col in matrix_cols:
        info = column_mappings.get(str(col))
        if info:
            gsm_id = info.get("sample_id", str(col))
            col_rename[col] = gsm_id
            labels_dict[gsm_id] = info.get("condition", "Tumor")
            patient_dict[gsm_id] = info.get("patient_id", gsm_id)
        else:
            col_rename[col] = str(col)
            labels_dict[str(col)] = "Tumor"
            patient_dict[str(col)] = str(col)

    df_genes = df_genes.rename(columns=col_rename)

    # Keep only columns with valid Tumor or Normal labels
    valid_samples = [s for s, cond in labels_dict.items() if cond in ("Tumor", "Normal") and s in df_genes.columns]
    if len(valid_samples) == 0:
        raise ValueError(f"Failed to align any valid comparative samples for {accession}.")

    df_genes = df_genes[valid_samples]
    label_series = pd.Series({s: labels_dict[s] for s in valid_samples}, name="condition")
    patient_series = pd.Series({s: patient_dict.get(s, s) for s in valid_samples}, name="patient_id")
    label_series.attrs["patient_id"] = patient_series

    # Step 5: Translate Gene Identifiers to official Gene Symbols
    df_genes = translate_gene_identifiers(df_genes, platform)

    # Step 6: Dynamic Value Scale Normalization
    df_genes = normalize_expression_values(df_genes)

    logger.info(
        f"Universal Adapter successfully processed {accession}: "
        f"{df_genes.shape[0]} genes x {df_genes.shape[1]} samples "
        f"({(label_series == 'Tumor').sum()} Tumor, {(label_series == 'Normal').sum()} Normal, "
        f"{patient_series.nunique()} patient groups)"
    )

    return df_genes, label_series
