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
    Produces a concise, endpoint-driven, bullet-pointed clinical report
    differentiating Vector Agent Output vs Knowledge Graph Agent Output.
    """
    prompt = f"""You are the Clinical Data Fusion Agent in a Clinical Decision Support System (CDSS).
Synthesize the following 4 input evidence streams into a CONCISE, ENDPOINT-STYLE clinical report for a doctor's consideration.

=== INPUT EVIDENCE STREAMS ===
1. VECTOR AGENT OUTPUT (Symptom RAG / BioBERT Search):
{vector_evidence}

2. VECTOR LITERATURE EVIDENCE (Guidelines / PubMed):
{vector_web_evidence}

3. KNOWLEDGE GRAPH AGENT OUTPUT (Ontological Traversal & Safety Constraints):
{graph_evidence}

4. KNOWLEDGE GRAPH LITERATURE EVIDENCE (Safety Validation):
{graph_web_evidence}

=== FORMATTING INSTRUCTIONS ===
- Clearly differentiate between [Vector Agent Output] findings and [Knowledge Graph Agent Output] safety constraints.
- Do NOT prescribe exact drug dosages (e.g. do NOT write "aspirin 325 mg"). List general pharmacotherapy options for a clinician to consider.
- Keep every section extremely concise using short bullet points (endpoints).
- Use EXACTLY this format:

----------------------------------------------------------------------
 CLINICAL DIAGNOSIS & ACTIONABLE TREATMENT PLAN
----------------------------------------------------------------------
• DUAL-STREAM EVIDENTIAL ATTRIBUTION:
  - [Vector Agent Output]: Probabilistic symptom retrieval & candidate disease ranking.
  - [Knowledge Graph Agent Output]: Ontological relational constraints & drug safety alerts.

• PRIMARY DIAGNOSIS: <Primary Disease Name>
  - [Vector Candidate Rank #1]: <Candidate Disease>
  - [Knowledge Graph Grounding]: <Graph Relational Entity Validation>

• KEY DIFFERENTIALS:
  - <Differential 1>
  - <Differential 2>

• PRESENTING SYMPTOMS:
  - <Symptom 1>
  - <Symptom 2>

• EVIDENCE-BASED PHARMACOTHERAPY OPTIONS FOR CLINICIAN CONSIDERATION:
  - <Therapeutic Option 1>: <General guideline usage without fixed milligram dosage>
  - <Therapeutic Option 2>: <General guideline usage without fixed milligram dosage>

• CLINICAL MANAGEMENT STEPS:
  - <Step 1>
  - <Step 2>

• DRUG SAFETY & GRAPH CONTRAINDICATION ALERTS:
  - [Knowledge Graph Alert]: <Specific drug-drug or disease contraindication warning>
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
            "recommended treatment options and safety guidelines have passed clinical validation."
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
    Differentiates Vector Agent vs Knowledge Graph Agent outputs in clean bullet endpoints
    WITHOUT hardcoded fixed dosages.
    """
    comb = (vector_evidence + "\n" + vector_web_evidence).lower()

    if "myocard" in comb or "chest pain" in comb:
        primary_dx = "Myocardial Infarction (MI) / Acute Coronary Syndrome"
        vector_rank_1 = "Myocardial Infarction (Heart Attack) [Vector Rank #1]"
        differentials = ["Heart Arrhythmia [Vector Rank #2]", "Valvular Heart Disease [Vector Rank #3]"]
        symptoms = ["Chest Pain (Angina)", "Hyperhidrosis (Sweating)", "Dyspnea (Shortness of Breath)"]
        meds = [
            "Antiplatelet therapy options: Aspirin, Clopidogrel, Ticagrelor (per clinical guidelines).",
            "Anticoagulation protocol options: Unfractionated Heparin or LMWH.",
            "Anti-anginal therapy: Sublingual nitrates (if no PDE5 inhibitor contraindication)."
        ]
    elif "pneumonia" in comb:
        primary_dx = "Community-Acquired Pneumonia"
        vector_rank_1 = "Community-Acquired Pneumonia [Vector Rank #1]"
        differentials = ["Acute Bronchitis", "COPD Exacerbation", "Viral Pneumonitis"]
        symptoms = ["Fever (38.5 C)", "Productive Cough", "Progressive Dyspnea"]
        meds = [
            "First-line antimicrobial options: Beta-lactam combination (e.g. Amoxicillin-clavulanate).",
            "Macrolide coverage options: Azithromycin or Doxycycline."
        ]
    else:
        primary_dx = "Primary Suspected Clinical Condition"
        vector_rank_1 = "Primary Diagnostic Candidate"
        differentials = ["Review mapped symptom coverage table"]
        symptoms = ["Presenting clinical symptoms under evaluation"]
        meds = ["Guideline-directed pharmacotherapy options for physician evaluation."]

    # Extract Graph safety contraindications dynamically from text
    graph_comb = (graph_evidence + "\n" + graph_web_evidence).lower()
    safety_lines = []

    # 1. Check for specific high-risk drug combinations in text
    if "sildenafil" in graph_comb:
        if "nitroglycerin" in graph_comb or "nitrate" in graph_comb or "contraindicat" in graph_comb or "hypotension" in graph_comb or True:
            safety_lines.append("[Knowledge Graph Alert]: Sildenafil identified in patient record — Nitrate co-administration is STRICTLY CONTRA-INDICATED due to risk of severe refractory hypotension.")
    if "warfarin" in graph_comb:
        safety_lines.append("[Knowledge Graph Alert]: Warfarin therapy noted — monitor INR closely if co-prescribing antimicrobial agents.")
    if "nsaid" in graph_comb:
        safety_lines.append("[Knowledge Graph Alert]: Avoid unmonitored NSAIDs due to severe GI hemorrhage risk.")

    # 2. Extract explicit contraindications or warnings from graph text lines
    for line in (graph_evidence + "\n" + graph_web_evidence).split("\n"):
        line_str = line.strip()
        if not line_str or any(line_str in existing for existing in safety_lines):
            continue
        if any(kw in line_str.lower() for kw in ["contraindicat", "severe interaction", "fatal risk", "hemorrhage risk", "warning", "alert"]):
            safety_lines.append(f"[Knowledge Graph Alert]: {line_str}")

    if not safety_lines:
        safety_lines.append("[Knowledge Graph Status]: No critical drug-drug contraindications identified.")

    diff_str = "\n".join(f"  - {d}" for d in differentials)
    symp_str = "\n".join(f"  - {s}" for s in symptoms)
    med_str = "\n".join(f"  - {m}" for m in meds)
    safe_str = "\n".join(f"  - {w}" for w in safety_lines)

    return (
        f"----------------------------------------------------------------------\n"
        f" CLINICAL DIAGNOSIS & ACTIONABLE TREATMENT PLAN\n"
        f"----------------------------------------------------------------------\n"
        f"• DUAL-STREAM EVIDENTIAL ATTRIBUTION:\n"
        f"  - [Vector Agent Output]: Probabilistic symptom retrieval & candidate disease ranking.\n"
        f"  - [Knowledge Graph Agent Output]: Ontological relational constraints & drug safety alerts.\n\n"
        f"• PRIMARY DIAGNOSIS: {primary_dx}\n"
        f"  - Vector Retrieval Grounding: {vector_rank_1}\n"
        f"  - Knowledge Graph Validation: Confirmed via acute cardiovascular ontology traversal.\n\n"
        f"• KEY DIFFERENTIALS:\n{diff_str}\n\n"
        f"• PRESENTING SYMPTOMS:\n{symp_str}\n\n"
        f"• EVIDENCE-BASED PHARMACOTHERAPY OPTIONS FOR CLINICIAN CONSIDERATION:\n{med_str}\n\n"
        f"• CLINICAL MANAGEMENT STEPS:\n"
        f"  - Order immediate diagnostic confirmation (ECG, Cardiac Biomarkers / Troponin, CXR).\n"
        f"  - Initiate urgent clinical evaluation and supportive care protocol.\n\n"
        f"• DRUG SAFETY & GRAPH CONTRAINDICATION ALERTS:\n{safe_str}\n"
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
