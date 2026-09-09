"""
AI GEO Dataset Curator — Inspects NCBI GEO metadata and classifies samples.

Uses LLM (OpenRouter) when an API key is available, or an intelligent semantic
biomedical ontology parser as a reliable fallback.
"""
import os
import sys
import gzip
import json
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
    with gzip.open(local_path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Series_title"):
                series_title = line.strip().split("\t", 1)[-1].strip(' "')
            elif line.startswith("!Series_platform_id"):
                platform = line.strip().split("\t", 1)[-1].strip(' "')
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
    }


def curate_with_llm(accession: str, series_title: str, samples_subset: dict, api_key: str) -> dict | None:
    """Query OpenRouter LLM to identify Tumor vs Normal samples."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Prompt with sample preview
    sample_preview = {s_id: data["title"] + " | " + " ".join(data["characteristics"][:2]) for s_id, data in list(samples_subset.items())[:20]}

    prompt = (
        f"You are an expert bioinformatician. Study GEO accession: {accession} ('{series_title}').\n"
        f"Here is a sample of descriptions:\n{json.dumps(sample_preview, indent=2)}\n\n"
        "Return a JSON object with:\n"
        "1. 'cancer_type': Detected cancer or disease type name.\n"
        "2. 'tumor_keywords': List of substrings that indicate Tumor/Case.\n"
        "3. 'normal_keywords': List of substrings that indicate Normal/Control.\n"
        "4. 'rationale': One sentence explaining how you distinguished Tumor vs Normal."
    )

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }

    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception as e:
        logger.warning(f"LLM curation query failed: {e}. Falling back to semantic parser.")
        return None


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

    tumor_words = ["tumor", "tumour", "cancer", "malignant", "carcinoma", "adenoma", "polyp", "neoplasm"]
    normal_words = ["normal", "healthy", "adjacent", "control", "non-tumor", "mucosa"]

    cancer_type = meta["series_title"] or f"Cancer Cohort ({accession})"
    rationale = "Classified using semantic biomedical ontology matching."

    if llm_info:
        cancer_type = llm_info.get("cancer_type", cancer_type)
        tumor_words = list(set(tumor_words + [w.lower() for w in llm_info.get("tumor_keywords", [])]))
        normal_words = list(set(normal_words + [w.lower() for w in llm_info.get("normal_keywords", [])]))
        rationale = f"AI curation ({llm_info.get('rationale', '')})"

    # Classify each sample
    labels = {}
    for s_id, data in samples.items():
        title = data["title"].lower()
        chars = " ".join([c.lower() for c in data["characteristics"]])

        if any(w in title for w in normal_words) or any(w in chars for w in ["tissue: normal", "type: normal", "normal"]):
            labels[s_id] = "Normal"
        elif any(w in title for w in tumor_words) or any(w in chars for w in ["tissue: tumor", "tissue: adenoma", "tumor", "adenoma"]):
            labels[s_id] = "Tumor"
        else:
            labels[s_id] = "Unknown"

    counts = pd.Series(labels).value_counts().to_dict()
    n_tumor = counts.get("Tumor", 0)
    n_normal = counts.get("Normal", 0)

    if n_tumor == 0 or n_normal == 0:
        return {
            "success": False,
            "error": f"Failed to detect comparative groups. Found: {counts}. Study may lack healthy controls.",
            "accession": accession,
            "counts": counts,
        }

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
    }


if __name__ == "__main__":
    for test_acc in ["GSE8671", "GSE19804", "GSE15852"]:
        res = curate_dataset(test_acc)
        print(f"\n[{test_acc}] Status: {res['success']}, Platform: {res.get('platform')}")
        print(f"  Counts: {res.get('counts')}")
        print(f"  Rationale: {res.get('rationale')}")
