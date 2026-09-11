"""
AI Copilot Engine — Interactive Oncology & Bioinformatics Assistant.

Provides real-time intelligent chat capabilities grounded in the
pipeline's discovered biomarkers, clinical survival, pathways, and drug mappings.
"""
import os
import sys
import json
import urllib.request
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import logger

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


def get_api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY")


def build_pipeline_context() -> str:
    """
    Assemble current computational telemetry and findings into a dense context block.
    """
    accession = getattr(config, "GEO_ACCESSION", "Unknown")
    cancer_type = getattr(config, "CANCER_TYPE", "Oncology Cohort")
    surv_info = getattr(config, "SURVIVAL_COHORT", {})
    val_info = getattr(config, "VALIDATION_COHORT", {})

    context_lines = [
        f"STUDY TELEMETRY:",
        f"- Active Discovery Cohort: {accession}",
        f"- Cancer Type: {cancer_type}",
        f"- Survival Cohort: {surv_info.get('label', 'GSE31210 / GSE1456')}",
        f"- External Validation Cohort: {val_info.get('label', 'GSE18842 / GSE42568')}",
        "",
    ]

    # Consensus Biomarkers
    consensus_path = os.path.join(config.RESULTS_DIR, "consensus_biomarkers.csv")
    if os.path.exists(consensus_path):
        df_c = pd.read_csv(consensus_path)
        top_c = df_c.head(10)
        context_lines.append("TOP CONSENSUS BIOMARKERS (DE + Random Forest ML):")
        for _, r in top_c.iterrows():
            gene = r.get("gene", "Unknown")
            reg = r.get("regulation", "Unknown")
            fc = r.get("log2FC", 0.0)
            pval = r.get("adj_pvalue", 1.0)
            score = r.get("ensemble_score", 0.0)
            context_lines.append(f"  • {gene}: {reg} (log2FC={fc:+.2f}, FDR={pval:.2e}, ML Score={score:.4f})")
        context_lines.append("")

    # Survival Validation
    surv_path = os.path.join(config.RESULTS_DIR, "survival_validation.csv")
    if os.path.exists(surv_path):
        df_s = pd.read_csv(surv_path)
        context_lines.append("PROGNOSTIC SURVIVAL MARKERS (Kaplan-Meier / Log-Rank Test):")
        sig_s = df_s[df_s["p_value"] < 0.05].head(6) if "p_value" in df_s.columns else df_s.head(6)
        for _, r in sig_s.iterrows():
            gene = r.get("gene", "Unknown")
            p = r.get("p_value", 1.0)
            hr = r.get("hazard_ratio", 1.0)
            context_lines.append(f"  • {gene}: p={p:.2e}, Hazard Ratio={hr:.2f}")
        context_lines.append("")

    # Pathways
    pathway_path = os.path.join(config.RESULTS_DIR, "pathway_enrichment.csv")
    if os.path.exists(pathway_path):
        df_p = pd.read_csv(pathway_path)
        context_lines.append("TOP ENRICHED PATHWAYS (GO & KEGG):")
        for _, r in df_p.head(6).iterrows():
            term = r.get("Term", r.get("pathway", "Unknown"))
            pval = r.get("Adjusted_P_value", r.get("p_value", 0.05))
            context_lines.append(f"  • {term} (FDR={pval:.2e})")
        context_lines.append("")

    # Targeted Drugs
    drug_path = os.path.join(config.RESULTS_DIR, "drug_actionability.csv")
    if os.path.exists(drug_path):
        df_d = pd.read_csv(drug_path)
        context_lines.append("ACTIONABLE TARGETED ONCOLOGY DRUGS:")
        for _, r in df_d.head(8).iterrows():
            gene = r.get("gene", "Unknown")
            drugs = r.get("approved_drugs", r.get("drugs", "Investigational"))
            mech = r.get("mechanism", "Targeted inhibition")
            context_lines.append(f"  • {gene} → {drugs} ({mech})")
        context_lines.append("")

    # External Validation
    ext_path = os.path.join(config.RESULTS_DIR, "external_validation_metrics.csv")
    if os.path.exists(ext_path):
        df_e = pd.read_csv(ext_path)
        if len(df_e) > 0:
            m = df_e.iloc[0]
            context_lines.append(
                f"EXTERNAL COHORT VALIDATION: Accuracy={m.get('test_accuracy', 0)*100:.1f}%, "
                f"ROC-AUC={m.get('roc_auc', 0):.4f}, Sensitivity={m.get('sensitivity', 0)*100:.1f}%, "
                f"Specificity={m.get('specificity', 0)*100:.1f}%"
            )
            context_lines.append("")

    return "\n".join(context_lines)


def query_ai_copilot(user_query: str, chat_history: list = None) -> str:
    """
    Query the AI Oncologist Copilot with grounded telemetry and multi-turn context.
    """
    api_key = get_api_key()
    context = build_pipeline_context()

    if not api_key:
        return (
            f"### ℹ️ Offline Analytical Summary\n\n"
            f"*OpenRouter API key is not configured. Here is the direct pipeline telemetry for **{getattr(config, 'CANCER_TYPE', 'Current Study')}**:*\n\n"
            f"```\n{context}\n```\n\n"
            f"> [!TIP]\n"
            f"> Add `OPENROUTER_API_KEY` to `.env` to unlock interactive conversational reasoning, "
            f"therapeutic hypothesis generation, and wet-lab validation designs!"
        )

    system_prompt = (
        "You are an elite translational oncologist and expert bioinformatics AI Copilot built into the "
        "Cancer Biomarker Discovery & Validation Platform.\n"
        "You are assisting a cancer research scientist who is investigating real clinical genomics datasets.\n\n"
        "GROUNDING CONTEXT (Current Pipeline Telemetry):\n"
        f"{context}\n\n"
        "INSTRUCTIONS:\n"
        "1. Base your answer directly on the loaded dataset, consensus biomarkers, survival outcomes, and drug mappings shown above.\n"
        "2. Provide deep, biologically rigorous explanations (molecular mechanisms, downstream signaling cascades, prognostic implications).\n"
        "3. Format cleanly with bullet points, bold gene names (e.g. **CDH3**, **AXIN2**), and concise summary tables when helpful.\n"
        "4. Keep responses high-impact, direct, and under 200 words so the clinical scientist gets instant answers.\n"
        "5. When asked for experimental validation, list 2-3 specific assays (RT-qPCR, Western blot, IHC, siRNA knockdown)."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Include recent history for conversation continuity (up to 4 turns)
    if chat_history:
        for turn in chat_history[-4:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_query})

    payload = json.dumps({
        "model": DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": 350,
        "temperature": 0.2,
    }).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://cancer-biomarker-pipeline.local",
        "X-Title": "Cancer Biomarker Copilot",
    }

    last_err = None
    for attempt in range(2):
        try:
            req = urllib.request.Request(OPENROUTER_API_URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            if attempt == 0:
                import time
                time.sleep(0.5)
                continue

    logger.warning(f"AI Copilot request failed after retry: {last_err}")
    return (
        f"### ℹ️ Analytical Telemetry Report\n\n"
        f"> **Notice:** The AI gateway was momentarily unavailable (`{str(last_err)}`).\n"
        f"> Here is the live telemetry and biomarker findings extracted directly from your results:\n\n"
        f"{context}"
    )
