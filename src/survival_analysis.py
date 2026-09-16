"""
Clinical Survival & Prognostic Modeling Module.

Harvests clinical follow-up time and event status from GEO metadata.
Generates Kaplan-Meier survival curves and computes Log-Rank test p-values
and Hazard Ratios for top biomarker genes.
"""
import os
import sys
import re
import pandas as pd
import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger


def extract_clinical_survival_data(samples_meta: dict) -> pd.DataFrame:
    """
    Scan sample characteristics for survival time and event status.

    Recognizes patterns like:
    - survival_months, os_months, survival_days, days_to_death, time_to_relapse, bcr_free_time, follow_up_months
    - vital_status, death, event, status, bcr (YES/NO), recurrence, relapsed
    """
    records = []

    time_patterns = [
        r"(?:survival|os|dfs|rfs|pfs|bcr_free|relapse_free|time|follow_up|futime)[_ -]?(?:months|time|days|years)?\s*[:=]?\s*([0-9.]+)",
        r"(?:overall survival|recurrence free|disease free)\s*[:=]?\s*([0-9.]+)",
        r"time\s*[:=]\s*([0-9.]+)",
    ]

    event_patterns = [
        r"(?:vital_status|status|death|deceased|event|relapse|recurrence|bcr)\s*[:=]?\s*(\w+)",
    ]

    for s_id, s_data in samples_meta.items():
        title = s_data.get("title", "")
        chars = " ".join(s_data.get("characteristics", []))
        full_text = f"{title} {chars}".lower()

        surv_time = np.nan
        surv_event = np.nan

        # 1. Parse survival time
        for pat in time_patterns:
            m = re.search(pat, full_text)
            if m:
                try:
                    val = float(m.group(1))
                    if val >= 0:
                        surv_time = val
                        break
                except ValueError:
                    pass

        # 2. Parse event status
        for pat in event_patterns:
            m = re.search(pat, full_text)
            if m:
                ev_str = m.group(1).lower()
                if any(w in ev_str for w in ["yes", "1", "dead", "deceased", "recurrence", "relapsed", "positive", "death"]):
                    surv_event = 1
                    break
                elif any(w in ev_str for w in ["no", "0", "alive", "living", "negative", "censor", "censored"]):
                    surv_event = 0
                    break

        if not np.isnan(surv_time):
            # If time is available but event is missing, assume default censored=0 or event=1 based on cohort context
            if np.isnan(surv_event):
                surv_event = 1 if "dead" in full_text else 0
            records.append({
                "sample_id": s_id,
                "time": surv_time,
                "event": int(surv_event)
            })

    if not records or len(records) < 10:
        return pd.DataFrame(columns=["sample_id", "time", "event"])

    return pd.DataFrame(records).set_index("sample_id")


def compute_kaplan_meier(time: np.ndarray, event: np.ndarray):
    """
    Compute non-parametric Kaplan-Meier survival curve coordinates.
    """
    order = np.argsort(time)
    time = time[order]
    event = event[order]

    unique_times, counts = np.unique(time, return_counts=True)
    n_at_risk = len(time)

    timeline = [0.0]
    survival_prob = [1.0]

    current_p = 1.0

    for t in unique_times:
        idx = (time == t)
        d_i = np.sum(event[idx])  # Number of events at time t
        n_i = n_at_risk           # Number at risk

        if n_i > 0:
            current_p *= (1.0 - d_i / n_i)

        timeline.append(float(t))
        survival_prob.append(float(current_p))

        n_at_risk -= len(idx)

    return timeline, survival_prob


def log_rank_test(
    time_high: np.ndarray,
    event_high: np.ndarray,
    time_low: np.ndarray,
    event_low: np.ndarray
) -> tuple[float, float, float]:
    """
    Compute two-sample Log-Rank test statistic, p-value, and approximate Hazard Ratio.
    """
    all_times = np.sort(np.unique(np.concatenate([time_high, time_low])))

    o_high, e_high = 0.0, 0.0
    v_high = 0.0

    n_high = len(time_high)
    n_low = len(time_low)

    for t in all_times:
        d_high = np.sum((time_high == t) & (event_high == 1))
        d_low = np.sum((time_low == t) & (event_low == 1))
        d_total = d_high + d_low

        r_high = np.sum(time_high >= t)
        r_low = np.sum(time_low >= t)
        r_total = r_high + r_low

        if r_total > 1 and d_total > 0:
            e_h = r_high * (d_total / r_total)
            o_high += d_high
            e_high += e_h

            var_t = (r_high * r_low * d_total * (r_total - d_total)) / (r_total**2 * (r_total - 1))
            v_high += var_t

    if v_high > 0:
        z = (o_high - e_high) / np.sqrt(v_high)
        chi2_stat = z**2
        p_value = float(1.0 - chi2.cdf(chi2_stat, df=1))
        hazard_ratio = (o_high / e_high) / (max(0.001, (len(event_high) - o_high) / max(0.001, len(event_high) - e_high))) if e_high > 0 else 1.0
    else:
        chi2_stat = 0.0
        p_value = 1.0
        hazard_ratio = 1.0

    return chi2_stat, p_value, max(0.01, min(100.0, hazard_ratio))


def run_survival_analysis(
    expr_df: pd.DataFrame,
    samples_meta: dict,
    top_genes: list[str],
    out_dir: str = None
) -> tuple[pd.DataFrame, dict]:
    """
    Run prognostic survival analysis across top candidate genes.
    """
    if out_dir is None:
        out_dir = config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    clin_df = extract_clinical_survival_data(samples_meta)
    if clin_df.empty:
        logger.info("  No longitudinal survival follow-up endpoints detected in this cohort metadata.")
        return pd.DataFrame(), {}

    # Intersect with expression sample IDs
    common_samples = clin_df.index.intersection(expr_df.columns)
    if len(common_samples) < 10:
        logger.info(f"  Insufficient matched clinical samples ({len(common_samples)}) for survival analysis.")
        return pd.DataFrame(), {}

    clin_df = clin_df.loc[common_samples]
    expr_sub = expr_df[common_samples]

    results = []
    curves = {}

    for gene in top_genes:
        if gene not in expr_sub.index:
            continue

        vals = expr_sub.loc[gene].values
        median_val = np.median(vals)

        high_idx = (vals >= median_val)
        low_idx = (vals < median_val)

        t_high = clin_df.loc[high_idx, "time"].values
        e_high = clin_df.loc[high_idx, "event"].values

        t_low = clin_df.loc[low_idx, "time"].values
        e_low = clin_df.loc[low_idx, "event"].values

        stat, p_val, hr = log_rank_test(t_high, e_high, t_low, e_low)

        time_h, surv_h = compute_kaplan_meier(t_high, e_high)
        time_l, surv_l = compute_kaplan_meier(t_low, e_low)

        curves[gene] = {
            "high": {"time": time_h, "survival": surv_h, "n": int(len(t_high))},
            "low": {"time": time_l, "survival": surv_l, "n": int(len(t_low))},
            "p_value": p_val,
            "hazard_ratio": hr
        }

        results.append({
            "gene": gene,
            "logrank_p_value": float(f"{p_val:.2e}") if p_val < 0.001 else round(p_val, 4),
            "hazard_ratio": round(hr, 2),
            "n_high": int(len(t_high)),
            "n_low": int(len(t_low)),
            "prognostic_significance": "Significant (p < 0.05)" if p_val < 0.05 else "Not Significant"
        })

    surv_df = pd.DataFrame(results).sort_values("logrank_p_value").reset_index(drop=True)
    csv_path = os.path.join(out_dir, "survival_analysis.csv")
    surv_df.to_csv(csv_path, index=False)
    logger.info(f"Clinical survival analysis complete: evaluated {len(surv_df)} genes (Saved to: {csv_path})")

    return surv_df, curves


def run_survival_pipeline(force_synthetic: bool = False) -> pd.DataFrame:
    """
    Run end-to-end clinical survival analysis pipeline for the active cohort or AI-selected survival cohort.
    """
    out_dir = config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Load consensus biomarkers
    cons_file = os.path.join(out_dir, "consensus_biomarkers.csv")
    if os.path.exists(cons_file):
        cons_df = pd.read_csv(cons_file)
        top_genes = cons_df["gene"].head(10).tolist()
    else:
        top_genes = ["TOP2A", "CDK1", "MKI67", "PCNA", "MYC", "TP53", "EGFR", "VEGFA"]

    if force_synthetic:
        np.random.seed(42)
        sample_ids = [f"GSM_SURV_{i}" for i in range(1, 61)]
        meta_mock = {
            s: {
                "title": f"Patient {s}",
                "characteristics": [
                    f"survival_months: {np.random.exponential(36):.1f}",
                    f"vital_status: {'dead' if np.random.rand() > 0.45 else 'alive'}"
                ]
            }
            for s in sample_ids
        }
        expr_mock = pd.DataFrame(np.random.randn(len(top_genes), len(sample_ids)), index=top_genes, columns=sample_ids)
        surv_df, _ = run_survival_analysis(expr_mock, meta_mock, top_genes, out_dir=out_dir)
        return surv_df

    # Load metadata and expression data for the active cohort
    try:
        from src.ai_geo_curator import fetch_geo_metadata
        from src.data_loader import load_data
        meta = fetch_geo_metadata(config.GEO_ACCESSION)
        if meta.get("success"):
            samples_meta = meta.get("samples", {})
            expr_df, _ = load_data(force_synthetic=False)
            surv_df, _ = run_survival_analysis(expr_df, samples_meta, top_genes, out_dir=out_dir)
            return surv_df
    except Exception as e:
        logger.warning(f"Survival pipeline fallback: {e}")

    return pd.DataFrame()


if __name__ == "__main__":
    # Test with synthetic survival data
    np.random.seed(42)
    sample_ids = [f"GSM{i}" for i in range(100, 150)]
    meta_mock = {
        s: {"title": f"Patient {s}", "characteristics": [f"survival_months: {np.random.exponential(24):.1f}", f"vital_status: {'dead' if np.random.rand() > 0.5 else 'alive'}"]}
        for s in sample_ids
    }
    expr_mock = pd.DataFrame(np.random.randn(5, 50), index=["TOP2A", "MYC", "TP53", "CDK1", "EGFR"], columns=sample_ids)

    df_res, crvs = run_survival_analysis(expr_mock, meta_mock, ["TOP2A", "MYC", "TP53", "CDK1", "EGFR"])
    print("Survival Analysis Test Results:")
    print(df_res.to_string(index=False))
