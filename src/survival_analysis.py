"""Endpoint-specific clinical survival analysis (all harmonized times are years)."""
import csv
import gzip
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger


_ENDPOINTS = {"overall": ("SURV_DEATH", "DEATH"), "relapse": ("SURV_RELAPSE", "RELAPSE")}
_ANALYSIS_COLUMNS = ["gene", "logrank_p_value", "hazard_ratio", "n_high", "n_low", "prognostic_significance"]
_METRICS = {"pvalue": "p_value", "hazard_ratio": "hazard_ratio", "hr_lower": "hr_lower",
            "hr_upper": "hr_upper", "cox_hr": "cox_hr", "cox_pvalue": "cox_pvalue",
            "5yr_high": "surv_5yr_high", "5yr_low": "surv_5yr_low"}
_VALIDATION_COLUMNS = (["gene"] + [f"{ep}_{metric}" for ep in ("os", "rfs") for metric in _METRICS]
                       + ["os_fdr", "rfs_fdr", "prognostic_status", "cohort_accession", "data_provenance",
                          "p_value", "hazard_ratio"])


def _key(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _time_field(key, accession=None):
    """Match complete endpoint labels, never substrings or an unqualified 'time'."""
    # These historical standardized columns are explicitly defined in years.
    if key in {"surv_death", "surv_relapse"}:
        return ("overall" if key == "surv_death" else "relapse"), 1.0
    if accession == "GSE39582" and key == "rfs_delay":
        return "relapse", 1 / 12.0  # documented CIT cohort unit
    for endpoint, stems in {
        "overall": ("os", "overall_survival", "survival", "follow_up", "followup"),
        "relapse": ("rfs", "relapse_free", "relapse_free_survival", "recurrence_free", "recurrence_free_survival", "time_to_relapse"),
    }.items():
        for stem in stems:
            match = re.fullmatch(re.escape(stem) + r"(?:_time|_delay)?_(days?|months?|years?)", key)
            if match:
                unit = match.group(1)
                return endpoint, 1 / 365.25 if unit.startswith("day") else 1 / 12 if unit.startswith("month") else 1.0
    if key == "days_to_death":
        return "overall", 1 / 365.25
    return None


def _event_field(key):
    if key in {"os_event", "os_status", "overall_survival_event", "overall_survival_status", "vital_status", "death", "deceased"}:
        return "overall"
    if key in {"rfs_event", "rfs_status", "relapse", "relapsed", "recurrence", "relapse_status", "recurrence_status"}:
        return "relapse"
    # DFS, PFS and biochemical recurrence are distinct endpoints, not OS or RFS.
    return None


def _event_value(value, endpoint):
    value = str(value).strip().lower()
    positive = {"1", "1.0", "yes", "event"}
    negative = {"0", "0.0", "no", "censored", "censor"}
    if endpoint == "overall":
        positive |= {"dead", "deceased", "death", "1:dead", "1:deceased"}
        negative |= {"alive", "living", "0:alive", "0:living"}
    else:
        positive |= {"relapsed", "recurrence", "recurred"}
        negative |= {"no recurrence", "no relapse", "relapse-free", "recurrence-free"}
    if value in positive:
        return 1.0
    if value in negative:
        return 0.0
    return np.nan


def _harmonize_clinical(samples_meta, accession=None):
    records = []
    for sample_id, sample in samples_meta.items():
        characteristics = sample.get("characteristics", [])
        if isinstance(characteristics, dict):
            fields = list(characteristics.items())
        else:
            if isinstance(characteristics, str):
                characteristics = [characteristics]
            fields = [re.split(r"\s*[:=]\s*", text, maxsplit=1)
                      for text in (characteristics or []) if isinstance(text, str) and re.search(r"[:=]", text)]
        values = {ep: {"time": [], "event": []} for ep in _ENDPOINTS}
        for label, value in fields:
            key = _key(label)
            field = _time_field(key, accession)
            if field:
                endpoint, factor = field
                try:
                    number = float(value) * factor
                except (ValueError, TypeError):
                    number = np.nan
                values[endpoint]["time"].append(number if np.isfinite(number) and number >= 0 else np.nan)
            endpoint = _event_field(key)
            if endpoint:
                values[endpoint]["event"].append(_event_value(value, endpoint))
        record = {"sample_id": sample_id}
        for endpoint, columns in _ENDPOINTS.items():
            for kind, column in zip(("time", "event"), columns):
                entries = values[endpoint][kind]
                # Conflicting or explicitly unknown observations remain missing.
                record[column] = entries[0] if entries and np.allclose(entries, entries[0]) else np.nan
        records.append(record)
    return pd.DataFrame(records, columns=["sample_id", "SURV_DEATH", "DEATH", "SURV_RELAPSE", "RELAPSE"]).set_index("sample_id")


def extract_clinical_survival_data(samples_meta: dict, outcome: str = "overall", accession: str = None) -> pd.DataFrame:
    """Extract complete, explicitly recorded OS or RFS pairs, with time in years.

    Sample titles, generic status/time, and other endpoints cannot supply an event.
    Unknown units, missing events, or conflicting annotations are not imputed.
    """
    if outcome not in _ENDPOINTS:
        raise ValueError(f"Unsupported survival outcome: {outcome}")
    clinical = _harmonize_clinical(samples_meta, accession)
    columns = _ENDPOINTS[outcome]
    return clinical.loc[:, list(columns)].rename(columns=dict(zip(columns, ("time", "event")))).dropna()


def _valid_survival(time, event):
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=float)
    if time.ndim != 1 or event.ndim != 1 or time.shape != event.shape:
        raise ValueError("Durations and events must be equal-length one-dimensional arrays")
    valid = np.isfinite(time) & (time >= 0) & np.isin(event, [0, 1])
    return time[valid], event[valid]


def compute_kaplan_meier(time: np.ndarray, event: np.ndarray):
    """Product-limit estimate; tied censorings leave risk only after tied events."""
    time, event = _valid_survival(time, event)
    n_at_risk = len(time)
    timeline, survival_prob = [0.0], [1.0]
    current_p = 1.0
    for t in np.unique(time):
        idx = time == t
        current_p *= 1.0 - np.count_nonzero(event[idx] == 1) / n_at_risk
        timeline.append(float(t))
        survival_prob.append(float(current_p))
        n_at_risk -= int(np.count_nonzero(idx))
    return timeline, survival_prob


def fit_cox_ph_model(durations, events, gene_expr, covariates=None):
    """Cox PH estimate per expression unit, with undefined results on failed fits."""
    undefined = dict.fromkeys(["cox_coef", "cox_hr", "cox_hr_lower", "cox_hr_upper", "cox_pvalue"], np.nan)
    undefined["converged"] = False
    d, e, g = (np.asarray(x, dtype=float) for x in (durations, events, gene_expr))
    valid = np.isfinite(d) & (d >= 0) & np.isin(e, [0, 1]) & np.isfinite(g)
    exog = pd.DataFrame({"gene": g})
    if covariates is not None:
        for column in covariates:
            exog[column] = pd.to_numeric(covariates[column], errors="coerce").to_numpy()
        valid &= np.isfinite(exog).all(axis=1).to_numpy()
    d, e, exog = d[valid], e[valid], exog.loc[valid]
    if len(d) < 3 or np.sum(e) < 2 or exog["gene"].nunique() < 2:
        return undefined
    try:
        # Non-convergence/separation must not masquerade as finite clinical HRs.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = PHReg(d, exog, status=e, ties="breslow")
            result = model.fit(disp=False)
            coef, se = float(result.params[0]), float(result.bse[0])
            hr, lower, upper = np.exp([coef, *result.conf_int()[0]])
            if (not np.all(np.isfinite([coef, se, hr, lower, upper])) or se <= 0
                    or np.max(np.abs(model.score(result.params))) > 1e-5
                    or np.min(np.linalg.eigvalsh(-model.hessian(result.params))) <= 1e-8):
                return undefined
        return {"cox_coef": coef, "cox_hr": float(hr), "cox_hr_lower": float(lower),
                "cox_hr_upper": float(upper), "cox_pvalue": float(result.pvalues[0]), "converged": True}
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, Warning):
        return undefined


def _logrank_and_hr(time_high, event_high, time_low, event_low):
    time_high, event_high = _valid_survival(time_high, event_high)
    time_low, event_low = _valid_survival(time_low, event_low)
    observed, expected, variance = 0.0, 0.0, 0.0
    for t in np.unique(np.concatenate([time_high[event_high == 1], time_low[event_low == 1]])):
        dh = np.count_nonzero((time_high == t) & (event_high == 1))
        dl = np.count_nonzero((time_low == t) & (event_low == 1))
        rh, rl = np.count_nonzero(time_high >= t), np.count_nonzero(time_low >= t)
        total, deaths = rh + rl, dh + dl
        if total > 1:
            observed += dh
            expected += rh * deaths / total
            variance += rh * rl * deaths * (total - deaths) / (total**2 * (total - 1))
    stat = (observed - expected)**2 / variance if variance > 0 else 0.0
    p_value = float(chi2.sf(stat, df=1))
    # Separate binary-group Cox fit: HR means high-expression hazard / low hazard.
    cox = fit_cox_ph_model(np.r_[time_high, time_low], np.r_[event_high, event_low],
                           np.r_[np.ones(len(time_high)), np.zeros(len(time_low))])
    return float(stat), p_value, cox["cox_hr"], cox["cox_hr_lower"], cox["cox_hr_upper"]


def log_rank_test(time_high, event_high, time_low, event_low) -> tuple[float, float, float]:
    """Return log-rank chi-square, stable tail p-value, and binary Cox high/low HR."""
    return _logrank_and_hr(time_high, event_high, time_low, event_low)[:3]


def _survival_cohort_info():
    selected = getattr(config, "SURVIVAL_COHORT", None)
    if isinstance(selected, dict) and selected.get("accession"):
        return dict(selected)
    acc, cancer = getattr(config, "GEO_ACCESSION", ""), getattr(config, "CANCER_TYPE", "").lower()
    if "colon" in cancer or "colorectal" in cancer or acc == "GSE8671":
        return {"accession": "GSE39582", "platform": "GPL570"}
    if "lung" in cancer or "nsclc" in cancer or acc == "GSE19804":
        return {"accession": "GSE31210", "platform": "GPL570"}
    if "breast" in cancer or acc == "GSE15852":
        return {"accession": "GSE1456", "platform": "GPL96"}
    # Never silently substitute an unrelated cancer's survival cohort.
    return {"accession": acc}


def _create_synthetic_survival_cohort():
    rng = np.random.default_rng(getattr(config, "RANDOM_SEED", 42))
    genes = list(dict.fromkeys(list(getattr(config, "KNOWN_MARKERS", [])) + ["TOP2A", "CDK1", "MKI67", "MYC", "TP53"]))
    samples = [f"SYNTHETIC_{i}" for i in range(60)]
    expr = pd.DataFrame(rng.normal(size=(len(genes), 60)), index=genes, columns=samples)
    clinical = pd.DataFrame({"SURV_DEATH": rng.exponential(5, 60), "DEATH": rng.integers(0, 2, 60),
                             "SURV_RELAPSE": rng.exponential(4, 60), "RELAPSE": rng.integers(0, 2, 60)}, index=samples)
    return expr, clinical


def load_survival_cohort(force_synthetic: bool = False):
    """Load the selected survival matrix, independently of discovery labels/config.

    Uses the established accession-specific expression/clinical caches, or the
    cohort's local GEO series matrix. Missing authentic data fails closed; there
    is no implicit synthetic fallback or tumor-vs-normal filtering.
    """
    if force_synthetic:
        return _create_synthetic_survival_cohort()
    info = _survival_cohort_info()
    accession = info["accession"]
    if not re.fullmatch(r"GSE\d+", accession):
        raise ValueError("A valid GEO survival cohort accession is required")
    expr_cache = os.path.join(config.DATA_DIR, f"{accession}_expression.csv")
    clin_cache = os.path.join(config.DATA_DIR, f"{accession}_clinical.csv")
    if os.path.exists(expr_cache) and os.path.exists(clin_cache):
        expression = pd.read_csv(expr_cache, index_col=0)
        clinical = pd.read_csv(clin_cache, index_col=0)
        if not set(clinical.columns).intersection({"SURV_DEATH", "SURV_RELAPSE"}):
            clinical = _harmonize_clinical({s: {"characteristics": row.to_dict()} for s, row in clinical.iterrows()}, accession)
        return expression, clinical

    platform = info.get("platform") or {"GSE1456": "GPL96", "GSE31210": "GPL570", "GSE39582": "GPL570"}.get(accession)
    matrix_file = os.path.join(config.DATA_DIR, f"{accession}_series_matrix.txt.gz")
    alternate = os.path.join(config.DATA_DIR, f"{accession}-{platform}_series_matrix.txt.gz")
    if not os.path.exists(matrix_file) and os.path.exists(alternate):
        matrix_file = alternate
    if not os.path.exists(matrix_file):
        raise RuntimeError(f"Authentic survival cohort {accession} is not available in data/")
    samples, characteristics, skiprows = [], [], None
    with gzip.open(matrix_file, "rt", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            fields = next(csv.reader([line], delimiter="\t"))
            if not fields or not fields[0].strip():
                continue  # blank lines carry no metadata
            if fields[0] == "!series_matrix_table_begin":
                skiprows = i + 1
                break
            if fields[0] == "!Sample_geo_accession":
                samples = fields[1:]
            elif fields[0] == "!Sample_characteristics_ch1":
                characteristics.append(fields[1:])
            elif fields[0] == "!Series_platform_id":
                declared = fields[1]
                if platform and declared != platform:
                    raise ValueError(f"Survival matrix platform {declared} differs from selected {platform}")
                platform = declared
    if skiprows is None or not samples or any(len(row) != len(samples) for row in characteristics):
        raise ValueError("Malformed survival series matrix metadata")
    clinical = _harmonize_clinical({s: {"characteristics": [row[i] for row in characteristics]}
                                    for i, s in enumerate(samples)}, accession)
    expression = pd.read_csv(matrix_file, sep="\t", skiprows=skiprows, index_col=0, comment="!")
    expression = expression.apply(pd.to_numeric, errors="coerce")
    if platform:
        from src.ai_geo_curator import fetch_gpl_probe_mapping
        mapping = fetch_gpl_probe_mapping(platform)
        if mapping:
            expression.index = expression.index.astype(str).str.strip().map(mapping)
            expression = expression.loc[expression.index.notna()].groupby(level=0).mean()
    # Do not cache newly inferred events: retain missing outcomes as missing.
    return expression, clinical


def evaluate_biomarker_survival(gene: str, outcome: str = "overall", expr_df=None, clinical_df=None) -> dict:
    """Dashboard contract: KM timelines in years and explicit high/low median cutoff."""
    if outcome not in _ENDPOINTS:
        return {"error": f"Unsupported survival outcome: {outcome}"}
    if (expr_df is None) != (clinical_df is None):
        return {"error": "Expression and clinical data must be supplied together"}
    if expr_df is None:
        try:
            expr_df, clinical_df = load_survival_cohort()
        except Exception as exc:
            return {"error": f"Survival cohort unavailable: {exc}"}
    if gene not in expr_df.index:
        return {"error": f"Gene {gene} not found in validation cohort"}
    if not expr_df.columns.is_unique or not clinical_df.index.is_unique:
        return {"error": "Duplicate patient IDs in survival cohort"}
    time_col, event_col = _ENDPOINTS[outcome]
    if time_col not in clinical_df or event_col not in clinical_df:
        return {"error": f"Outcome {outcome} columns not available"}
    samples = clinical_df.index.intersection(expr_df.columns)
    time = pd.to_numeric(clinical_df.loc[samples, time_col], errors="coerce").to_numpy(dtype=float)
    event = pd.to_numeric(clinical_df.loc[samples, event_col], errors="coerce").to_numpy(dtype=float)
    values = expr_df.loc[gene, samples]
    if isinstance(values, pd.DataFrame):
        values = values.apply(pd.to_numeric, errors="coerce").mean(axis=0)
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time) & (time >= 0) & np.isin(event, [0, 1]) & np.isfinite(values)
    time, event, values = time[valid], event[valid], values[valid]
    if len(time) < 10:
        return {"error": "Insufficient matched clinical samples (minimum 10)"}
    cutoff = float(np.median(values))
    high = values >= cutoff
    if min(np.count_nonzero(high), np.count_nonzero(~high)) < 2:
        return {"error": "Insufficient expression variation for two survival groups"}
    th, sh = compute_kaplan_meier(time[high], event[high])
    tl, sl = compute_kaplan_meier(time[~high], event[~high])
    _, p, hr, lower, upper = _logrank_and_hr(time[high], event[high], time[~high], event[~high])
    continuous = fit_cox_ph_model(time, event, values)

    def at_five(timeline, survival, durations):
        # No extrapolation beyond observed follow-up unless survival is already zero.
        if durations.max() < 5 and survival[-1] > 0:
            return np.nan
        return float(np.asarray(survival)[np.asarray(timeline) <= 5][-1] * 100)

    return {"gene": gene, "outcome": outcome, "n_total": len(time), "n_high": int(high.sum()),
            "n_low": int((~high).sum()), "median_cutoff": cutoff, "p_value": p,
            "hazard_ratio": hr, "hr_lower": lower, "hr_upper": upper,
            "cox_hr": continuous["cox_hr"], "cox_hr_lower": continuous["cox_hr_lower"],
            "cox_hr_upper": continuous["cox_hr_upper"], "cox_pvalue": continuous["cox_pvalue"],
            "surv_5yr_high": at_five(th, sh, time[high]), "surv_5yr_low": at_five(tl, sl, time[~high]),
            "km_high": {"timeline": th, "survival": sh}, "km_low": {"timeline": tl, "survival": sl}}


def _clear_outputs(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(columns=_ANALYSIS_COLUMNS).to_csv(os.path.join(out_dir, "survival_analysis.csv"), index=False)
    pd.DataFrame(columns=_VALIDATION_COLUMNS).to_csv(os.path.join(out_dir, "survival_validation.csv"), index=False)


def _evaluate_candidates(expr_df, clinical_df, top_genes, out_dir, accession, provenance):
    records, analysis, curves = [], [], {}
    for gene in dict.fromkeys(top_genes):
        outcomes = {ep: evaluate_biomarker_survival(gene, outcome, expr_df, clinical_df)
                    for ep, outcome in (("os", "overall"), ("rfs", "relapse"))}
        if all("error" in result for result in outcomes.values()):
            continue
        record = {"gene": gene, "cohort_accession": accession, "data_provenance": provenance}
        for ep, result in outcomes.items():
            record.update({f"{ep}_{key}": result.get(value, np.nan) for key, value in _METRICS.items()})
        record.update(p_value=record["os_pvalue"], hazard_ratio=record["os_hazard_ratio"])
        records.append(record)
        result = outcomes["os"]
        if "error" not in result:
            curves[gene] = dict(result, high={"time": result["km_high"]["timeline"], "survival": result["km_high"]["survival"], "n": result["n_high"]},
                                low={"time": result["km_low"]["timeline"], "survival": result["km_low"]["survival"], "n": result["n_low"]})
            analysis.append({"gene": gene, "logrank_p_value": result["p_value"], "hazard_ratio": result["hazard_ratio"],
                             "n_high": result["n_high"], "n_low": result["n_low"],
                             "prognostic_significance": "Significant (p < 0.05)" if result["p_value"] < .05 else "Not Significant"})
    validation = pd.DataFrame(records, columns=_VALIDATION_COLUMNS)
    for ep in ("os", "rfs"):
        valid = validation[f"{ep}_pvalue"].notna()
        if valid.any():
            validation.loc[valid, f"{ep}_fdr"] = multipletests(validation.loc[valid, f"{ep}_pvalue"].astype(float), method="fdr_bh")[1]
    if len(validation):
        validation["prognostic_status"] = np.where(validation["os_fdr"] < .05, "Prognostic (FDR < 0.05)",
                                                    np.where(validation["os_pvalue"] < .05, "Nominally Significant (p < 0.05)", "Not Significant"))
    validation = validation.sort_values("os_pvalue").reset_index(drop=True)
    summary = pd.DataFrame(analysis, columns=_ANALYSIS_COLUMNS).sort_values("logrank_p_value").reset_index(drop=True)
    validation.to_csv(os.path.join(out_dir, "survival_validation.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "survival_analysis.csv"), index=False)
    return validation, summary, curves


def run_survival_analysis(expr_df: pd.DataFrame, samples_meta: dict, top_genes: list[str], out_dir: str = None):
    """Analyze supplied metadata, retaining the newer summary/curve API."""
    out_dir = out_dir or config.RESULTS_DIR
    _clear_outputs(out_dir)
    clinical = _harmonize_clinical(samples_meta, getattr(config, "GEO_ACCESSION", None))
    _, summary, curves = _evaluate_candidates(expr_df, clinical, top_genes, out_dir,
                                               getattr(config, "GEO_ACCESSION", ""), "supplied_metadata")
    return summary, curves


def run_survival_pipeline(force_synthetic: bool = False) -> pd.DataFrame:
    """Write OS/RFS validation schema, isolating simulations and clearing stale runs."""
    out_dir = os.fspath(config.RESULTS_DIR)
    if force_synthetic and os.path.basename(os.path.normpath(out_dir)) != "synthetic_test":
        out_dir = os.path.join(out_dir, "synthetic_test")
    _clear_outputs(out_dir)
    try:
        top_genes = list(getattr(config, "KNOWN_MARKERS", []))[:15]
        consensus_path = os.path.join(out_dir, "consensus_biomarkers.csv")
        if os.path.exists(consensus_path) and os.path.getsize(consensus_path):
            consensus = pd.read_csv(consensus_path)
            if "gene" in consensus:
                top_genes = consensus["gene"].dropna().astype(str).head(30).tolist()
        expression, clinical = load_survival_cohort(force_synthetic=force_synthetic)
        accession = "SYNTHETIC" if force_synthetic else _survival_cohort_info()["accession"]
        validation, _, _ = _evaluate_candidates(expression, clinical, top_genes, out_dir, accession,
                                                 "synthetic_simulation" if force_synthetic else "authentic_clinical_geo")
        return validation
    except Exception as exc:
        _clear_outputs(out_dir)
        logger.warning(f"Survival analysis unavailable: {exc}")
        return pd.DataFrame(columns=_VALIDATION_COLUMNS)


if __name__ == "__main__":
    print(run_survival_pipeline(force_synthetic=True).to_string(index=False))
