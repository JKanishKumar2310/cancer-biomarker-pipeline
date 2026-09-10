"""
AI GEO Dataset Curator — Inspects NCBI GEO metadata and classifies samples.

Uses LLM (OpenRouter) when an API key is available, or an intelligent semantic
biomedical ontology parser as a reliable fallback.
"""
import os
import sys
import gzip
import json
import re
import urllib.request
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger


def fetch_geo_metadata(accession: str) -> dict:
    """
    Download and parse header metadata of a GEO series matrix.

    Returns
    -------
    dict with keys: 'accession', 'platform', 'title', 'samples', 'characteristics', 'success', 'error'
    """
    accession = accession.strip().upper()
    if not accession.startswith("GSE"):
        return {"success": False, "error": f"Invalid accession '{accession}'. Must start with GSE."}

    # Derive URL
    # NCBI structure: GSEnnn -> e.g. GSE8nnn for GSE8671, GSE19nnn for GSE19804
    num_part = accession[3:]
    if len(num_part) < 3:
        stub = "GSEnnn"
    else:
        stub = f"GSE{num_part[:-3]}nnn"

    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{accession}/matrix/{accession}_series_matrix.txt.gz"
    local_path = os.path.join(config.DATA_DIR, f"{accession}_series_matrix.txt.gz")

    if not os.path.exists(local_path):
        logger.info(f"Downloading {accession} series matrix metadata from NCBI...")
        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            return {"success": False, "error": f"Failed to download from {url}: {e}"}

    # Parse metadata header
    meta_lines = []
    platform = "GPL570"  # default fallback
    series_title = ""
    supplementary_files = []
    with gzip.open(local_path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Series_title"):
                series_title = line.strip().split("\t", 1)[-1].strip(' "')
            elif line.startswith("!Series_platform_id"):
                platform = line.strip().split("\t", 1)[-1].strip(' "')
            elif line.startswith("!Series_supplementary_file"):
                supp = line.strip().split("\t", 1)[-1].strip(' "')
                supplementary_files.append(supp)
            meta_lines.append(line)

    sample_ids, sample_titles, characteristics = [], [], []
    for line in meta_lines:
        if line.startswith("!Sample_geo_accession"):
            sample_ids = [x.strip(' "\t\r\n') for x in line.split("\t")[1:]]
        elif line.startswith("!Sample_title"):
            sample_titles = [x.strip(' "\t\r\n') for x in line.split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            characteristics.append([x.strip(' "\t\r\n') for x in line.split("\t")[1:]])

    sample_dict = {}
    for i, s_id in enumerate(sample_ids):
        t = sample_titles[i] if i < len(sample_titles) else ""
        chars = [ch[i] for ch in characteristics if i < len(ch) and not ch[i].lower().startswith("cel filename")]
        sample_dict[s_id] = {
            "title": t,
            "characteristics": chars,
        }

    return {
        "success": True,
        "accession": accession,
        "platform": platform,
        "series_title": series_title,
        "samples": sample_dict,
        "local_path": local_path,
        "supplementary_files": supplementary_files,
    }


def extract_sample_templates(samples: dict) -> dict:
    """
    Group samples into distinct naming templates by replacing digit sequences with '#'.
    This collapses hundreds of repetitive sample titles into 2-5 distinct templates.
    """
    templates = {}
    for s_id, data in samples.items():
        title = data["title"].strip()
        tmpl = re.sub(r"\d+", "#", title)
        if tmpl not in templates:
            templates[tmpl] = {
                "count": 0,
                "example_title": title,
                "chars": data.get("characteristics", [])[:3],
                "sample_ids": [],
            }
        templates[tmpl]["count"] += 1
        templates[tmpl]["sample_ids"].append(s_id)
    return templates


def curate_with_llm(accession: str, series_title: str, samples: dict, api_key: str) -> dict | None:
    """Query OpenRouter LLM to classify sample templates into Tumor vs Normal."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    templates = extract_sample_templates(samples)
    template_preview = {
        tmpl: f"{info['example_title']} | {' '.join(info['chars'])}"
        for tmpl, info in templates.items()
    }

    prompt = (
        f"You are an expert bioinformatician. Study GEO accession: {accession} ('{series_title}').\n"
        f"The cohort samples group into the following distinct title/metadata templates:\n"
        f"{json.dumps(template_preview, indent=2)}\n\n"
        "Return a JSON object with:\n"
        "1. 'cancer_type': Detected cancer or disease type name (e.g. 'Colorectal Adenoma', 'Non-Small Cell Lung Cancer', 'Breast Carcinoma').\n"
        "2. 'template_labels': A JSON dictionary mapping EVERY template key EXACTLY to 'Tumor' or 'Normal' or 'Exclude'.\n"
        "3. 'rationale': One sentence explaining how you distinguished Tumor vs Normal groups."
    )

    payload = {
        "model": "nvidia/nemotron-3.5-lightning:free",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as e:
            if attempt == 0:
                continue
            logger.warning(f"LLM curation query failed: {e}. Falling back to semantic parser.")
            return None


def _matches_term(term: str, text: str) -> bool:
    """Safely match a keyword or phrase against text, avoiding substring false positives."""
    term = term.strip().lower()
    text = text.lower()
    if not term or not text:
        return False
    # If term is 1-2 characters (e.g. 't', 'n', 'hc'), require boundary or numeric prefix (e.g., '2t', '165n')
    if len(term) <= 2:
        return bool(re.search(r'(?:\b|\d)' + re.escape(term) + r'\b', text))
    # If it's a multi-word phrase, check substring in text
    if " " in term:
        return term in text
    # Standard word boundary for single words >= 3 chars
    return bool(re.search(r'\b' + re.escape(term) + r'\b', text))


def _matches_any_term(terms: list, text: str) -> bool:
    return any(_matches_term(t, text) for t in terms if t)


def curate_dataset(accession: str) -> dict:
    """
    Autonomously inspect, curate, and classify any GEO dataset into Tumor vs Normal.
    """
    meta = fetch_geo_metadata(accession)
    if not meta.get("success"):
        return meta

    samples = meta["samples"]
    api_key = os.environ.get("OPENROUTER_API_KEY")
    llm_info = None
    if api_key:
        llm_info = curate_with_llm(accession, meta["series_title"], samples, api_key)

    cancer_type = meta["series_title"] or f"Cancer Cohort ({accession})"
    rationale = "Classified using semantic biomedical ontology matching."

    if llm_info:
        raw_ct = llm_info.get("cancer_type", cancer_type)
        if isinstance(raw_ct, dict):
            cancer_type = ", ".join([k for k in raw_ct.keys() if "normal" not in k.lower() and "control" not in k.lower()])
        elif isinstance(raw_ct, list):
            cancer_type = ", ".join([str(x) for x in raw_ct])
        elif isinstance(raw_ct, str):
            cancer_type = raw_ct
        rationale = f"AI curation ({llm_info.get('rationale', '')})"

    # Step 1: Assign via LLM template-level classification
    labels = {}
    templates = extract_sample_templates(samples)
    template_labels = llm_info.get("template_labels", {}) if llm_info else {}

    if template_labels:
        for tmpl, info in templates.items():
            assigned = template_labels.get(tmpl)
            if assigned in ("Tumor", "Normal"):
                for s_id in info["sample_ids"]:
                    labels[s_id] = assigned

    # Step 2: Fallback for any unassigned samples (or if LLM unavailable)
    unassigned = [s_id for s_id in samples if s_id not in labels]
    if unassigned:
        tumor_words = [
            "tumor", "tumour", "cancer", "malignant", "carcinoma", "adenoma", "polyp", "neoplasm",
            "crc", "gbm", "nsclc", "sclc", "luad", "lusc", "brca", "panc", "chol", "prad", "kirc",
            "glioma", "melanoma", "sarcoma",
        ]
        normal_words = [
            "normal", "healthy", "adjacent", "control", "ctrl", "non-tumor", "nontumor", "mucosa",
            "donor", "healthy control", "normal mucosa", "paired normal",
        ]
        for s_id in unassigned:
            data = samples[s_id]
            title = data["title"]
            chars = " ".join(data["characteristics"])
            full_text = f"{title} {chars}"

            is_normal = _matches_any_term(normal_words, full_text)
            is_tumor = _matches_any_term(tumor_words, full_text)

            if is_tumor and not is_normal:
                labels[s_id] = "Tumor"
            elif is_normal and not is_tumor:
                labels[s_id] = "Normal"
            elif is_tumor and is_normal:
                title_tumor = _matches_any_term(tumor_words, title)
                title_normal = _matches_any_term(normal_words, title)
                if title_tumor and not title_normal:
                    labels[s_id] = "Tumor"
                elif title_normal and not title_tumor:
                    labels[s_id] = "Normal"
                else:
                    labels[s_id] = "Normal" if "normal" in full_text.lower() else "Tumor"
            else:
                labels[s_id] = "Unknown"

    counts = pd.Series(labels).value_counts().to_dict()
    n_tumor = counts.get("Tumor", 0)
    n_normal = counts.get("Normal", 0)

    if n_tumor == 0 or n_normal == 0:
        return {
            "success": False,
            "error": f"Failed to detect comparative groups. Found: {counts}. Study may lack healthy controls or use unstandardized clinical codes.",
            "accession": accession,
            "counts": counts,
        }

    # Verify that the downloaded series matrix actually contains gene expression rows
    # (High-throughput RNA-seq series matrices often omit the data table, but provide a supplementary count matrix)
    has_expression_data = False
    suppl_url = None
    try:
        with gzip.open(meta["local_path"], "rt", encoding="utf-8", errors="ignore") as f:
            in_table = False
            row_count = 0
            for line in f:
                if line.startswith("!series_matrix_table_begin"):
                    in_table = True
                    continue
                if in_table:
                    if line.startswith("!series_matrix_table_end"):
                        break
                    row_count += 1
                    if row_count > 5:
                        has_expression_data = True
                        break
    except Exception:
        has_expression_data = True

    if not has_expression_data:
        # Check if an external supplementary count/expression matrix is available
        for s_file in meta.get("supplementary_files", []):
            s_lower = s_file.lower()
            if any(ext in s_lower for ext in [".txt.gz", ".tsv.gz", ".csv.gz", ".txt", ".tsv", ".csv"]):
                if not any(bad in s_lower for bad in ["raw.tar", ".bam", ".sra", ".bw", ".bed", ".bigwig", ".cel"]):
                    suppl_url = s_file
                    has_expression_data = True
                    logger.info(f"  Found supplementary expression matrix for {accession}: {suppl_url}")
                    break

    if not has_expression_data:
        return {
            "success": False,
            "error": f"Identified {n_tumor} Tumor vs {n_normal} Normal (Healthy Controls), but {accession} contains raw sequencing files without an embedded series matrix table or pre-computed gene count matrix in supplementary files. Please use standard cohorts with embedded matrices (e.g., GSE8671, GSE19804, GSE15852).",
            "accession": accession,
            "counts": counts,
        }

    # ── Write back to config so the entire downstream pipeline is aware ────
    config.GEO_ACCESSION = accession
    config.CANCER_TYPE = cancer_type
    logger.info(f"  config.GEO_ACCESSION → {accession}")
    logger.info(f"  config.CANCER_TYPE   → {cancer_type}")

    # ── Fetch cancer-specific known markers via AI ───────────────────────
    try:
        from src.ai_annotator import update_known_markers_for_cancer
        update_known_markers_for_cancer(cancer_type)
    except Exception as e:
        logger.warning(f"  Could not fetch AI markers: {e} — keeping existing KNOWN_MARKERS")

    return {
        "success": True,
        "accession": accession,
        "platform": meta["platform"],
        "cancer_type": cancer_type,
        "n_tumor": n_tumor,
        "n_normal": n_normal,
        "labels": labels,
        "rationale": rationale,
        "counts": counts,
        "suppl_url": suppl_url,
    }



if __name__ == "__main__":
    for test_acc in ["GSE8671", "GSE19804", "GSE15852"]:
        res = curate_dataset(test_acc)
        print(f"\n[{test_acc}] Status: {res['success']}, Platform: {res.get('platform')}")
        print(f"  Counts: {res.get('counts')}")
        print(f"  Rationale: {res.get('rationale')}")
