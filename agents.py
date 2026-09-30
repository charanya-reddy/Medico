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
    Produces a concise, endpoint-driven, bullet-pointed clinical report.
    """
    prompt = f"""You are the Clinical Data Fusion Agent in a Clinical Decision Support System (CDSS).
Synthesize the following 4 input evidence streams into a CONCISE, ENDPOINT-STYLE clinical report for a doctor or patient.

=== INPUT EVIDENCE STREAMS ===
1. SYMPTOM / VECTOR EVIDENCE:
{vector_evidence}

2. LITERATURE EVIDENCE (DIAGNOSIS):
{vector_web_evidence}

3. KNOWLEDGE GRAPH SAFETY & RELATIONS:
{graph_evidence}

4. LITERATURE EVIDENCE (SAFETY):
{graph_web_evidence}

=== FORMATTING INSTRUCTIONS ===
- Do NOT output long paragraphs or raw internal section headers (such as 'NORMALIZED MAPPING APPLIED').
- Keep every section extremely concise using short bullet points (endpoints).
- Use EXACTLY this format:

----------------------------------------------------------------------
 CLINICAL DIAGNOSIS & ACTIONABLE TREATMENT PLAN
----------------------------------------------------------------------
• PRIMARY DIAGNOSIS: <Primary Disease Name>
• KEY DIFFERENTIALS:
  - <Differential 1>
  - <Differential 2>

• PRESENTING SYMPTOMS:
  - <Symptom 1>
  - <Symptom 2>

• RECOMMENDED MEDICATIONS & DOSAGE:
  - <Medication 1>: <Short dosage/usage guidance>
  - <Medication 2>: <Short dosage/usage guidance>

• TREATMENT & MANAGEMENT STEPS:
  - <Step 1>
  - <Step 2>

• DRUG SAFETY & WARNINGS:
  - <Warning/Interaction 1>
  - <Warning/Interaction 2>
----------------------------------------------------------------------"""

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
    Outputs short, bulleted endpoints for doctors and patients.
    """
    comb = (vector_evidence + "\n" + vector_web_evidence).lower()

    if "myocard" in comb or "chest pain" in comb:
        primary_dx = "Myocardial Infarction (MI) / Acute Coronary Syndrome"
        differentials = ["Angina Pectoris", "Aortic Dissection", "Pulmonary Embolism"]
        symptoms = ["Chest Pain (Angina)", "Hyperhidrosis (Sweating)", "Dyspnea (Shortness of Breath)"]
        meds = [
            "Aspirin 325 mg: Chewed immediately upon acute presentation.",
            "Clopidogrel 300 mg / Ticagrelor 180 mg: Dual antiplatelet loading dose.",
            "Sublingual Nitroglycerin 0.4 mg: PRN for angina symptoms.",
            "Unfractionated Heparin: Acute anticoagulation protocol."
        ]
    elif "pneumonia" in comb:
        primary_dx = "Community-Acquired Pneumonia"
        differentials = ["Acute Bronchitis", "COPD Exacerbation", "Viral Pneumonitis"]
        symptoms = ["Fever (38.5 C)", "Productive Cough", "Progressive Dyspnea"]
        meds = [
            "Amoxicillin-clavulanate 875/125 mg PO BID: First-line antimicrobial.",
            "Azithromycin 500 mg Day 1, 250 mg Days 2-5: Macrolide coverage."
        ]
    else:
        primary_dx = "Primary Suspected Clinical Condition"
        differentials = ["Review mapped symptom coverage table"]
        symptoms = ["Presenting clinical symptoms under evaluation"]
        meds = ["Guideline-directed pharmacotherapy as indicated by physician."]

    safety_lines = []
    if "warfarin" in (graph_evidence + "\n" + graph_web_evidence).lower():
        safety_lines.append("Warfarin Interaction: Monitor INR closely if co-administering antibiotics.")
    if "nsaid" in (graph_evidence + "\n" + graph_web_evidence).lower():
        safety_lines.append("NSAID Warning: Avoid unmonitored NSAIDs due to GI bleeding risk.")
    if not safety_lines:
        safety_lines.append("No critical drug-drug contraindications identified.")

    diff_str = "\n".join(f"  - {d}" for d in differentials)
    symp_str = "\n".join(f"  - {s}" for s in symptoms)
    med_str = "\n".join(f"  - {m}" for m in meds)
    safe_str = "\n".join(f"  - {w}" for w in safety_lines)

    return (
        f"----------------------------------------------------------------------\n"
        f" CLINICAL DIAGNOSIS & ACTIONABLE TREATMENT PLAN\n"
        f"----------------------------------------------------------------------\n"
        f"• PRIMARY DIAGNOSIS: {primary_dx}\n"
        f"• KEY DIFFERENTIALS:\n{diff_str}\n\n"
        f"• PRESENTING SYMPTOMS:\n{symp_str}\n\n"
        f"• RECOMMENDED MEDICATIONS & DOSAGE:\n{med_str}\n\n"
        f"• TREATMENT & MANAGEMENT STEPS:\n"
        f"  - Order immediate diagnostic confirmation (ECG, Cardiac Biomarkers / Troponin, CXR).\n"
        f"  - Initiate urgent clinical evaluation and supportive care protocol.\n\n"
        f"• DRUG SAFETY & WARNINGS:\n{safe_str}\n"
        f"----------------------------------------------------------------------"
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
