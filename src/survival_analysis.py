"""
Clinical Survival Analysis Engine — Kaplan-Meier Curves & Log-Rank Tests.

Evaluates biomarker genes against real patient survival outcomes (OS and RFS)
using disease-matched clinical cohorts:
  - Colorectal: GSE39582 (French CIT Cohort, n=566, GPL570)
  - Lung (NSCLC): GSE31210 (Okayama et al., n=226, GPL570)
  - Breast: GSE1456 (Stockholm Cohort, n=159, GPL96)

Includes:
  - Non-parametric Kaplan-Meier survival curves S(t)
  - 2-sample Log-Rank test with 95% Confidence Intervals for Hazard Ratios (HR)
  - Cox Proportional Hazards regression (via statsmodels PHReg)
  - Benjamini-Hochberg FDR multiple testing correction
  - Strict isolation of synthetic test outputs to results/synthetic_test/
"""
import os
import sys
import gzip
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger


def _create_synthetic_survival_cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a minimal synthetic survival cohort for CI or offline environments."""
    np.random.seed(config.RANDOM_SEED)
    samples = [f"Patient_{i+1:03d}" for i in range(60)]
    genes = list(dict.fromkeys(config.KNOWN_MARKERS + ["CLIC5", "TNNC1", "TOP2A", "CDK1", "EPCAM", "CDH3", "FXYD1", "DHRS11"]))
    expr = pd.DataFrame(np.random.normal(7.0, 1.5, size=(len(genes), len(samples))), index=genes, columns=samples)
    clin = pd.DataFrame({
        "SURV_DEATH": np.random.uniform(0.5, 10.0, size=len(samples)),
        "DEATH": np.random.choice([0, 1], size=len(samples), p=[0.6, 0.4]),
        "SURV_RELAPSE": np.random.uniform(0.5, 10.0, size=len(samples)),
        "RELAPSE": np.random.choice([0, 1], size=len(samples), p=[0.7, 0.3]),
    }, index=samples)
    return expr, clin


def load_survival_cohort(force_synthetic: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load clinical survival cohort matching current cancer type.
    Strictly disease-matched: Colorectal -> GSE39582, Lung -> GSE31210, Breast -> GSE1456.
    """
    if force_synthetic:
        logger.info("Using synthetic survival cohort (test mode)...")
        return _create_synthetic_survival_cohort()

    surv_info = getattr(config, "SURVIVAL_COHORT", None)
    target_acc = surv_info.get("accession") if isinstance(surv_info, dict) else None

    if not target_acc:
        acc = getattr(config, "GEO_ACCESSION", "")
        ct = getattr(config, "CANCER_TYPE", "").lower()
        if "colon" in ct or "colorectal" in ct or acc == "GSE8671":
            target_acc = "GSE39582"
            cohort_name = "GSE39582 French CIT Colorectal Cohort (n=566)"
            platform = "GPL570"
        elif "lung" in ct or "nsclc" in ct or acc == "GSE19804":
            target_acc = "GSE31210"
            cohort_name = "GSE31210 Clinical Survival Cohort (NSCLC, n=226)"
            platform = "GPL570"
        elif "breast" in ct or acc == "GSE15852":
            target_acc = "GSE1456"
            cohort_name = "GSE1456 Stockholm Breast Cancer (n=159)"
            platform = "GPL96"
        else:
            target_acc = "GSE39582"
            cohort_name = "GSE39582 Survival Cohort"
            platform = "GPL570"
    else:
        cohort_name = surv_info.get("label", f"{target_acc} Survival Cohort")
        platform = surv_info.get("platform", "GPL570")

    expr_cache = os.path.join(config.DATA_DIR, f"{target_acc}_expression.csv")
    clin_cache = os.path.join(config.DATA_DIR, f"{target_acc}_clinical.csv")

    # If requested cohort is cached, load it directly
    if os.path.exists(expr_cache) and os.path.exists(clin_cache):
        logger.info(f"Loading cached {cohort_name}...")
        expr_df = pd.read_csv(expr_cache, index_col=0)
        clinical_df = pd.read_csv(clin_cache, index_col=0)
        return expr_df, clinical_df

    # Check for series matrix file
    matrix_file = os.path.join(config.DATA_DIR, f"{target_acc}_series_matrix.txt.gz")
    if not os.path.exists(matrix_file) and target_acc == "GSE1456":
        alt_matrix = os.path.join(config.DATA_DIR, "GSE1456-GPL96_series_matrix.txt.gz")
        if os.path.exists(alt_matrix):
            matrix_file = alt_matrix

    annot_file = os.path.join(config.DATA_DIR, f"{platform}.annot.gz")

    if not os.path.exists(matrix_file):
        if force_synthetic:
            logger.warning(f"Survival cohort {target_acc} not found. Using synthetic test survival cohort.")
            return _create_synthetic_survival_cohort()
        raise RuntimeError(
            f"Authentic clinical survival cohort '{target_acc}' ({cohort_name}) is not available in data/. "
            "Survival analysis requires disease-matched patient clinical outcome data."
        )

    logger.info(f"Parsing {target_acc} series matrix and clinical outcomes...")
    meta_lines = []
    skiprows = 0
    with gzip.open(matrix_file, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("!series_matrix_table_begin"):
                skiprows = i + 1
                break
            meta_lines.append(line)

    sample_ids = []
    sample_titles = []
    characteristics = []
    for line in meta_lines:
        if line.startswith("!Sample_geo_accession"):
            sample_ids = [x.strip(' "\t\r\n') for x in line.split("\t")[1:]]
        elif line.startswith("!Sample_title"):
            sample_titles = [x.strip(' "\t\r\n') for x in line.split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            characteristics.append([x.strip(' "\t\r\n') for x in line.split("\t")[1:]])

    clin_dict = {"sample_id": sample_ids}
    if sample_titles:
        clin_dict["title"] = sample_titles

    for ch_row in characteristics:
        if len(ch_row) > 0 and ":" in ch_row[0]:
            var_name = ch_row[0].split(":")[0].strip()
            vals = [c.split(":", 1)[1].strip() if ":" in c else c for c in ch_row]
            clin_dict[var_name] = vals

    clinical_df = pd.DataFrame(clin_dict).set_index("sample_id")

    # Harmonize survival endpoints across datasets
    if target_acc == "GSE39582":
        # GSE39582: os.delay (months), os.event (0/1), rfs.delay (months), rfs.event (0/1)
        if "os.delay (months)" in clinical_df.columns:
            clinical_df["SURV_DEATH"] = pd.to_numeric(clinical_df["os.delay (months)"], errors="coerce") / 12.0
        if "os.event" in clinical_df.columns:
            clinical_df["DEATH"] = pd.to_numeric(clinical_df["os.event"], errors="coerce").fillna(0).astype(int)
        if "rfs.delay" in clinical_df.columns:
            clinical_df["SURV_RELAPSE"] = pd.to_numeric(clinical_df["rfs.delay"], errors="coerce") / 12.0
        if "rfs.event" in clinical_df.columns:
            clinical_df["RELAPSE"] = pd.to_numeric(clinical_df["rfs.event"], errors="coerce").fillna(0).astype(int)
        if "age.at.diagnosis (year)" in clinical_df.columns:
            clinical_df["AGE"] = pd.to_numeric(clinical_df["age.at.diagnosis (year)"], errors="coerce")
        if "tnm.stage" in clinical_df.columns:
            clinical_df["STAGE"] = clinical_df["tnm.stage"]
    else:
        # GSE1456 / GSE31210
        for col in ["SURV_DEATH", "SURV_RELAPSE"]:
            if col in clinical_df.columns:
                clinical_df[col] = pd.to_numeric(clinical_df[col], errors="coerce")
        for col in ["DEATH", "DEATH_BC", "RELAPSE"]:
            if col in clinical_df.columns:
                clinical_df[col] = pd.to_numeric(clinical_df[col], errors="coerce").fillna(0).astype(int)

    # Read expression table from series matrix
    logger.info(f"Reading {target_acc} expression values...")
    df = pd.read_csv(
        matrix_file, compression="gzip", skiprows=skiprows, sep="\t", index_col=0, comment="!"
    )
    df = df[~df.index.astype(str).str.startswith("!")]
    df = df.apply(pd.to_numeric, errors="coerce")

    # Map probe IDs using platform annotations
    if os.path.exists(annot_file):
        logger.info(f"Mapping {target_acc} probes via {platform} annotations...")
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
        expr_df = df.groupby(df.index).mean()
    else:
        logger.warning(f"Platform annotation file {annot_file} not found; using raw probe IDs.")
        expr_df = df

    # Match common samples
    common = expr_df.columns.intersection(clinical_df.index)
    expr_df = expr_df[common]
    clinical_df = clinical_df.loc[common]

    # Convert raw intensity values to log2 scale if needed
    if expr_df.values.max() > 50:
        expr_df = np.log2(expr_df + 1)

    # Cache for fast reuse
    expr_df.to_csv(expr_cache)
    clinical_df.to_csv(clin_cache)
    logger.info(f"{target_acc} parsed & cached: {expr_df.shape[0]} genes × {expr_df.shape[1]} patients")
    return expr_df, clinical_df


def compute_kaplan_meier(
    durations: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Compute non-parametric Kaplan-Meier survival curve coordinates S(t).
    """
    valid = (~np.isnan(durations)) & (~np.isnan(events)) & (durations >= 0)
    durations = durations[valid]
    events = events[valid]

    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]

    unique_times = np.unique(durations)

    timeline = [0.0]
    survival_prob = [1.0]
    current_p = 1.0
    censored_times = []

    for t in unique_times:
        mask = (durations == t)
        d_i = np.sum(events[mask] == 1)  # number of events
        c_i = np.sum(events[mask] == 0)  # number censored
        n_i = np.sum(durations >= t)     # number at risk

        if c_i > 0:
            censored_times.extend([t] * c_i)

        if n_i > 0 and d_i > 0:
            current_p *= (1.0 - (d_i / n_i))

        timeline.append(float(t))
        survival_prob.append(float(current_p))

    return np.array(timeline), np.array(survival_prob), censored_times


def log_rank_test(
    dur_1: np.ndarray, ev_1: np.ndarray,
    dur_2: np.ndarray, ev_2: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Perform 2-sample Log-Rank statistical test and calculate Hazard Ratio
    with mathematically rigorous 95% Confidence Intervals.

    Returns
    -------
    chi2_stat : float
    p_value : float
    hazard_ratio : float
    hr_lower : float
        95% CI lower bound
    hr_upper : float
        95% CI upper bound
    """
    all_dur = np.concatenate([dur_1, dur_2])
    all_ev = np.concatenate([ev_1, ev_2])

    unique_event_times = np.sort(np.unique(all_dur[all_ev == 1]))
    if len(unique_event_times) == 0:
        return 0.0, 1.0, 1.0, 1.0, 1.0

    o1_total = 0.0
    e1_total = 0.0
    v_total = 0.0

    o2_total = 0.0
    e2_total = 0.0

    for t in unique_event_times:
        d1 = np.sum((dur_1 == t) & (ev_1 == 1))
        d2 = np.sum((dur_2 == t) & (ev_2 == 1))
        d = d1 + d2

        n1 = np.sum(dur_1 >= t)
        n2 = np.sum(dur_2 >= t)
        n = n1 + n2

        if n > 1 and d > 0:
            e1 = n1 * (d / n)
            e2 = n2 * (d / n)

            o1_total += d1
            e1_total += e1
            o2_total += d2
            e2_total += e2

            v = (n1 * n2 * d * (n - d)) / ((n ** 2) * (n - 1))
            v_total += v

    if v_total <= 0:
        return 0.0, 1.0, 1.0, 1.0, 1.0

    z = (o1_total - e1_total) / np.sqrt(v_total)
    chi2_stat = z ** 2
    p_val = 1.0 - stats.chi2.cdf(chi2_stat, df=1)

    # Hazard Ratio calculation
    if e1_total > 0 and e2_total > 0 and o2_total > 0 and o1_total > 0:
        hr = (o1_total / e1_total) / (o2_total / e2_total)
        # 95% Confidence Interval for Hazard Ratio: SE = sqrt(1 / v_total)
        se_ln_hr = np.sqrt(1.0 / v_total)
        hr_lower = float(np.exp(np.log(hr) - 1.96 * se_ln_hr))
        hr_upper = float(np.exp(np.log(hr) + 1.96 * se_ln_hr))
    else:
        hr = 1.0
        hr_lower = 1.0
        hr_upper = 1.0

    return float(chi2_stat), float(p_val), float(hr), float(hr_lower), float(hr_upper)


def fit_cox_ph_model(
    durations: np.ndarray,
    events: np.ndarray,
    gene_expr: np.ndarray,
    covariates: pd.DataFrame = None,
) -> dict:
    """
    Fit Cox Proportional Hazards regression using statsmodels PHReg.

    Evaluates biomarker as continuous predictor with covariate adjustment.
    """
    valid = (~np.isnan(durations)) & (~np.isnan(events)) & (~np.isnan(gene_expr)) & (durations > 0)
    d = durations[valid]
    e = events[valid].astype(int)
    g = gene_expr[valid]

    if np.sum(e) < 3 or np.var(g) < 1e-6:
        return {
            "cox_coef": np.nan,
            "cox_hr": np.nan,
            "cox_hr_lower": np.nan,
            "cox_hr_upper": np.nan,
            "cox_pvalue": np.nan,
            "converged": False,
        }

    try:
        exog = pd.DataFrame({"gene": g})
        if covariates is not None:
            for col in covariates.columns:
                c_vals = covariates.loc[valid, col].values
                exog[col] = pd.to_numeric(c_vals, errors="coerce")
            exog = exog.fillna(exog.median())

        model = PHReg(endog=d, status=e, exog=exog)
        res = model.fit(disp=False)

        coef = float(res.params[0])
        hr = float(np.exp(coef))
        ci = res.conf_int()
        hr_lower = float(np.exp(ci[0, 0]))
        hr_upper = float(np.exp(ci[0, 1]))
        pval = float(res.pvalues[0])

        return {
            "cox_coef": coef,
            "cox_hr": hr,
            "cox_hr_lower": hr_lower,
            "cox_hr_upper": hr_upper,
            "cox_pvalue": pval,
            "converged": True,
        }
    except Exception:
        return {
            "cox_coef": np.nan,
            "cox_hr": np.nan,
            "cox_hr_lower": np.nan,
            "cox_hr_upper": np.nan,
            "cox_pvalue": np.nan,
            "converged": False,
        }


def evaluate_biomarker_survival(
    gene: str,
    outcome: str = "overall",
    expr_df: pd.DataFrame = None,
    clinical_df: pd.DataFrame = None,
) -> dict:
    """
    Evaluate survival curves, log-rank test, and Cox PH model for a biomarker gene.
    """
    if expr_df is None or clinical_df is None:
        expr_df, clinical_df = load_survival_cohort()

    if gene not in expr_df.index:
        return {"error": f"Gene {gene} not found in validation cohort"}

    time_col = "SURV_DEATH" if outcome == "overall" else "SURV_RELAPSE"
    event_col = "DEATH" if outcome == "overall" else "RELAPSE"

    if time_col not in clinical_df.columns or event_col not in clinical_df.columns:
        return {"error": f"Outcome {outcome} columns ({time_col}, {event_col}) not available"}

    valid_mask = clinical_df[time_col].notna() & clinical_df[event_col].notna()
    samples = clinical_df[valid_mask].index.intersection(expr_df.columns)

    durations = np.asarray(clinical_df.loc[samples, time_col]).flatten().astype(float)
    events = np.asarray(clinical_df.loc[samples, event_col]).flatten().astype(int)

    val = expr_df.loc[gene, samples]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[0]
    gene_expr = np.asarray(val).flatten().astype(float)

    # Median cutoff for Kaplan-Meier stratification
    median_val = float(np.median(gene_expr))
    high_mask = gene_expr >= median_val
    low_mask = ~high_mask

    dur_high, ev_high = durations[high_mask], events[high_mask]
    dur_low, ev_low = durations[low_mask], events[low_mask]

    t_high, s_high, cens_high = compute_kaplan_meier(dur_high, ev_high)
    t_low, s_low, cens_low = compute_kaplan_meier(dur_low, ev_low)

    chi2, p_val, hr, hr_lower, hr_upper = log_rank_test(dur_high, ev_high, dur_low, ev_low)

    # Cox Proportional Hazards regression (continuous model)
    cox_res = fit_cox_ph_model(durations, events, gene_expr)

    # 5-year survival rates (t=5.0)
    def _surv_at_5yr(times, probs):
        t_arr = np.array(times)
        valid = t_arr <= 5.0
        return probs[valid][-1] * 100 if np.any(valid) else probs[-1] * 100

    s5_high = _surv_at_5yr(t_high, s_high)
    s5_low = _surv_at_5yr(t_low, s_low)

    return {
        "gene": gene,
        "outcome": outcome,
        "n_total": len(samples),
        "n_high": int(np.sum(high_mask)),
        "n_low": int(np.sum(low_mask)),
        "median_cutoff": float(median_val),
        "p_value": float(p_val),
        "hazard_ratio": float(hr),
        "hr_lower": float(hr_lower),
        "hr_upper": float(hr_upper),
        "cox_hr": float(cox_res["cox_hr"]) if not np.isnan(cox_res["cox_hr"]) else float(hr),
        "cox_hr_lower": float(cox_res["cox_hr_lower"]) if not np.isnan(cox_res["cox_hr_lower"]) else float(hr_lower),
        "cox_hr_upper": float(cox_res["cox_hr_upper"]) if not np.isnan(cox_res["cox_hr_upper"]) else float(hr_upper),
        "cox_pvalue": float(cox_res["cox_pvalue"]) if not np.isnan(cox_res["cox_pvalue"]) else float(p_val),
        "surv_5yr_high": float(s5_high),
        "surv_5yr_low": float(s5_low),
        "km_high": {"timeline": t_high.tolist(), "survival": s_high.tolist()},
        "km_low": {"timeline": t_low.tolist(), "survival": s_low.tolist()},
    }


def run_survival_pipeline(force_synthetic: bool = False) -> pd.DataFrame:
    """
    Run survival analysis across candidate biomarkers with Benjamini-Hochberg FDR
    correction and strict output isolation.
    """
    acc = getattr(config, "GEO_ACCESSION", "")
    ct = getattr(config, "CANCER_TYPE", "").lower()
    if "colon" in ct or "colorectal" in ct or acc == "GSE8671":
        cohort_label = "GSE39582 Colorectal Cohort"
    elif "lung" in ct or "nsclc" in ct or acc == "GSE19804":
        cohort_label = "GSE31210 Lung Cohort"
    else:
        cohort_label = "GSE1456 Breast Cohort"

    logger.info("=" * 60)
    logger.info(f"CLINICAL SURVIVAL ANALYSIS ({cohort_label})")
    logger.info("=" * 60)

    expr_df, clinical_df = load_survival_cohort(force_synthetic=force_synthetic)

    # Load consensus biomarkers
    consensus_path = os.path.join(config.RESULTS_DIR, "consensus_biomarkers.csv")
    test_genes = []
    if os.path.exists(consensus_path) and os.path.getsize(consensus_path) > 0:
        try:
            cons_df = pd.read_csv(consensus_path)
            if "gene" in cons_df.columns:
                test_genes = cons_df["gene"].tolist()[:30]
        except Exception:
            pass
    if not test_genes:
        test_genes = config.KNOWN_MARKERS[:15]

    # Ensure hallmark cancer markers are included
    for g in config.KNOWN_MARKERS:
        if g not in test_genes and g in expr_df.index:
            test_genes.append(g)

    records = []
    logger.info(f"Evaluating survival for {len(test_genes)} candidate biomarkers...")

    for g in test_genes:
        if g in expr_df.index:
            os_res = evaluate_biomarker_survival(g, outcome="overall", expr_df=expr_df, clinical_df=clinical_df)
            rfs_res = evaluate_biomarker_survival(g, outcome="relapse", expr_df=expr_df, clinical_df=clinical_df)

            if "error" not in os_res:
                records.append({
                    "gene": g,
                    "os_pvalue": os_res["p_value"],
                    "os_hazard_ratio": os_res["hazard_ratio"],
                    "os_hr_lower": os_res["hr_lower"],
                    "os_hr_upper": os_res["hr_upper"],
                    "os_cox_hr": os_res["cox_hr"],
                    "os_cox_pvalue": os_res["cox_pvalue"],
                    "os_5yr_high": os_res["surv_5yr_high"],
                    "os_5yr_low": os_res["surv_5yr_low"],
                    "rfs_pvalue": rfs_res.get("p_value", np.nan),
                    "rfs_hazard_ratio": rfs_res.get("hazard_ratio", np.nan),
                    "rfs_hr_lower": rfs_res.get("hr_lower", np.nan),
                    "rfs_hr_upper": rfs_res.get("hr_upper", np.nan),
                    "rfs_cox_hr": rfs_res.get("cox_hr", np.nan),
                    "rfs_cox_pvalue": rfs_res.get("cox_pvalue", np.nan),
                    "rfs_5yr_high": rfs_res.get("surv_5yr_high", np.nan),
                    "rfs_5yr_low": rfs_res.get("surv_5yr_low", np.nan),
                })

    surv_df = pd.DataFrame(records)
    if len(surv_df) > 0:
        # Benjamini-Hochberg FDR Multiple Testing Correction
        valid_os = surv_df["os_pvalue"].notna()
        if valid_os.sum() > 0:
            _, os_fdr, _, _ = multipletests(surv_df.loc[valid_os, "os_pvalue"], method="fdr_bh")
            surv_df.loc[valid_os, "os_fdr"] = os_fdr
        else:
            surv_df["os_fdr"] = np.nan

        valid_rfs = surv_df["rfs_pvalue"].notna()
        if valid_rfs.sum() > 0:
            _, rfs_fdr, _, _ = multipletests(surv_df.loc[valid_rfs, "rfs_pvalue"], method="fdr_bh")
            surv_df.loc[valid_rfs, "rfs_fdr"] = rfs_fdr
        else:
            surv_df["rfs_fdr"] = np.nan

        def _prognostic_class(row):
            fdr = row.get("os_fdr", np.nan)
            p = row.get("os_pvalue", np.nan)
            if not np.isnan(fdr) and fdr < 0.05:
                return "Prognostic (FDR < 0.05)"
            elif not np.isnan(p) and p < 0.05:
                return "Nominally Significant (p < 0.05)"
            else:
                return "Not Significant"

        surv_df["prognostic_status"] = surv_df.apply(_prognostic_class, axis=1)
        surv_df["cohort_accession"] = getattr(config, "SURVIVAL_COHORT", {}).get("accession", cohort_label.split()[0])
        surv_df["data_provenance"] = "synthetic_simulation" if force_synthetic else "authentic_clinical_geo"

        surv_df = surv_df.sort_values("os_pvalue")

        out_dir = os.path.join(config.RESULTS_DIR, "synthetic_test") if force_synthetic else config.RESULTS_DIR
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "survival_validation.csv")
        surv_df.to_csv(out_path, index=False)
        logger.info(f"Saved survival validation metrics -> {out_path}")

        logger.info("\nTop Prognostic Biomarkers by Patient Survival:")
        for _, row in surv_df.head(6).iterrows():
            fdr_str = f"FDR={row['os_fdr']:.3f}" if not np.isnan(row.get("os_fdr", np.nan)) else "FDR=NA"
            logger.info(
                f"  {row['gene']:<10} OS p={row['os_pvalue']:.2e} ({fdr_str})  "
                f"HR={row['os_hazard_ratio']:.2f} [95% CI {row['os_hr_lower']:.2f}-{row['os_hr_upper']:.2f}]  "
                f"Status: {row['prognostic_status']}"
            )

    return surv_df


if __name__ == "__main__":
    df = run_survival_pipeline()
