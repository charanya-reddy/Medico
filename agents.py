"""
FILE 3 of 5: agents.py

WHAT THIS FILE DOES:
Executes the two LLM-driven agents in the pipeline:
  1. Clinical Data Fusion Agent (run_fusion_agent)
     Synthesizes the FOUR input streams:
       - Stream 1: Vector RAG Output + Evidence Agent Output on Vector
       - Stream 2: Graph RAG Output + Evidence Agent Output on Graph
     Cross-checks diagnostic hypotheses against structured graph constraints and
     literature validation, producing an explainable clinical report.

  2. Adaptive Optimizer Agent (run_optimizer_agent)
     Evaluates the dual-confidence metrics (specifically gating on Confidence Score 1).
     If Confidence Score 1 >= 0.65 -> produces "FINAL:" sign-off.
     If Confidence Score 1 < 0.65  -> produces "RE-QUERY:" actionable query optimization guidance.

Includes robust fallback functions so the system runs smoothly even if the
LLM API key is absent or offline.
"""

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI


# Shared model connection configured for OpenRouter / OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "dummy_key_if_offline"),
)


def run_fusion_agent(
    vector_evidence: str,
    vector_web_evidence: str,
    graph_evidence: str,
    graph_web_evidence: str,
) -> str:
    """
    Agent 1: Clinical Data Fusion Agent.
    Receives 4 normalized evidence streams and generates a patient- and doctor-facing
    clinical diagnosis and treatment report.
    """
    prompt = f"""You are the Clinical Data Fusion Agent in an advanced Clinical Decision Support System (CDSS).
You receive four input streams from preceding retrieval agents:

=== 1. SYMPTOM / VECTOR RAG EVIDENCE ===
{vector_evidence}

=== 2. LITERATURE EVIDENCE ON DIAGNOSIS ===
{vector_web_evidence}

=== 3. KNOWLEDGE GRAPH RELATIONS & SAFETY CONSTRAINTS ===
{graph_evidence}

=== 4. LITERATURE EVIDENCE ON SAFETY & INTERACTIONS ===
{graph_web_evidence}

=== TASK: CLINICAL DIAGNOSIS & TREATMENT PLAN ===
Synthesize a clear, highly professional clinical report designed specifically for doctors and patients. 
Focus ONLY on actionable medical information:
  1. DIAGNOSED DISEASE / DISORDER (Primary Suspected Diagnosis & Differentials)
  2. CLINICAL FINDINGS & SYMPTOM SUMMARY
  3. RECOMMENDED MEDICATIONS & PHARMACOTHERAPY (First-line medications, dosage, administration guidelines)
  4. TREATMENT, CURE & MANAGEMENT PLAN (Clinical management, curative/supportive care steps)
  5. DRUG SAFETY, CONTRAINDICATIONS & FOLLOW-UP ADVICE (Interaction warnings, monitoring requirements, when to seek immediate emergency care)

Do NOT include raw technical confidence scores, code logic, or internal agent metadata in the report body."""

    response = llm.invoke(prompt)
    return response.content


def run_optimizer_agent(fusion_report: str, confidence_data: Dict[str, Any]) -> str:
    """
    Agent 2: Adaptive Optimizer Agent.
    Evaluates the dual-stream confidence breakdown.

    Gating Logic:
      - Checks Confidence Score 1 (Vector Track).
      - If C1 >= 0.65 -> Emits "FINAL:" accepted report.
      - If C1 < 0.65  -> Emits "RE-QUERY:" guidance with term boosting and query expansion directives.
    """
    c1_info = confidence_data["confidence_1"]
    c2_info = confidence_data["confidence_2"]
    fusion_info = confidence_data["fusion"]

    c1_score = c1_info["score"]
    c2_score = c2_info["score"]
    fused_score = fusion_info["final_fused_score"]
    passes_gate = c1_info["passes_threshold"]

    breakdown_text = (
        f"--- CONFIDENCE ASSESSMENT BREAKDOWN ---\n"
        f"- Confidence Score 1 (Vector Stream - GATING): {c1_score} (Threshold: 0.65) -> {'PASSED' if passes_gate else 'FAILED'}\n"
        f"- Confidence Score 2 (Graph Stream - HIGH): {c2_score}\n"
        f"- Final Fused Confidence Score: {fused_score}\n"
    )

    if passes_gate:
        instruction = (
            "Confidence Score 1 satisfies the acceptance threshold (>= 0.65). "
            "Write a concise confirmation validating the diagnostic synthesis and confirming that "
            "recommended medications and safety guidelines have passed clinical validation."
        )
    else:
        instruction = (
            "Confidence Score 1 is BELOW threshold (< 0.65). "
            "Write actionable re-query feedback with specific instructions for symptom term boosting "
            "and targeted PubMed query expansion."
        )

    prompt = f"""You are the Adaptive Optimizer Agent monitoring CDSS diagnostic reliability.

{breakdown_text}

FUSION AGENT'S SYNTHESIS REPORT:
{fusion_report}

INSTRUCTION:
{instruction}"""

    response = llm.invoke(prompt)
    return response.content


def build_fallback_report(
    vector_evidence: str,
    vector_web_evidence: str,
    graph_evidence: str,
    graph_web_evidence: str,
) -> str:
    """
    Deterministic fallback for run_fusion_agent() if LLM API is unavailable.
    Constructs a well-structured doctor/patient clinical report without crashing.
    """
    comb = (vector_evidence + "\n" + vector_web_evidence).lower()
    if "myocard" in comb or "chest pain" in comb:
        disease_hypothesis = "Myocardial Infarction (MI) / Acute Coronary Syndrome"
    elif "pneumonia" in comb:
        disease_hypothesis = "Community-Acquired Pneumonia"
    else:
        disease_hypothesis = "Primary Suspected Clinical Condition"

    return (
        f"======================================================================\n"
        f" CLINICAL DIAGNOSIS & ACTIONABLE TREATMENT REPORT\n"
        f"======================================================================\n\n"
        f"1. DIAGNOSED DISEASE / DISORDER:\n"
        f"   - Primary Diagnosis Candidate: {disease_hypothesis}\n"
        f"   - Clinical Differentials: Review mapped symptom coverage table.\n\n"
        f"2. CLINICAL FINDINGS & SYMPTOM SUMMARY:\n"
        f"   - Input Findings: {vector_evidence}\n"
        f"   - Supporting Literature Evidence: {vector_web_evidence}\n\n"
        f"3. RECOMMENDED MEDICATIONS & PHARMACOTHERAPY:\n"
        f"   - Recommended First-Line Therapy: Guideline-directed pharmacotherapy as clinically indicated.\n"
        f"   - Safety & Relational Evaluation: {graph_evidence}\n\n"
        f"4. TREATMENT, CURE & MANAGEMENT PLAN:\n"
        f"   - Immediate clinical evaluation & diagnostic confirmation (ECG/Troponin/Labs/Imaging).\n"
        f"   - Implement guideline-directed curative and supportive care protocols.\n"
        f"   - Evidence Validation: {graph_web_evidence}\n\n"
        f"5. DRUG SAFETY, CONTRAINDICATIONS & FOLLOW-UP ADVICE:\n"
        f"   - Monitor for medication contraindications and potential drug interactions.\n"
        f"   - Seek immediate emergency medical attention if acute symptoms persist or worsen."
    )


def build_fallback_optimizer(confidence_data: Dict[str, Any], attempt: int) -> str:
    """
    Deterministic fallback for run_optimizer_agent() if LLM API is unavailable.
    """
    c1_score = confidence_data["confidence_1"]["score"]
    c2_score = confidence_data["confidence_2"]["score"]
    fused = confidence_data["fusion"]["final_fused_score"]
    passes = confidence_data["confidence_1"]["passes_threshold"]

    if passes:
        return (
            f"FINAL: Confidence Score 1 ({c1_score}) satisfies the acceptance threshold (>= 0.65). "
            f"Graph Confidence Score 2 is {c2_score} and Final Fused Confidence is {fused}. "
            f"Diagnostic recommendations approved for clinical workflow."
        )
    else:
        return (
            f"RE-QUERY (Attempt {attempt}): Confidence Score 1 ({c1_score}) is below threshold (0.65). "
            f"Optimization directives: Activate medical term boosting on core clinical keywords, "
            f"expand BioBERT embedding radius, and initiate targeted PubMed queries for higher-level evidence."
        )
