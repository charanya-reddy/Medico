"""
FILE 3 of 5: fusion_agents.py

WHAT THIS FILE DOES:
Executes the two LLM-driven agents in the pipeline:
  1. Clinical Data Fusion Agent (run_fusion_agent)
     Synthesizes the FOUR input streams:
       - Stream 1: Vector RAG Output + Evidence Agent Output on Vector
       - Stream 2: Graph RAG Output + Evidence Agent Output on Graph
     Cross-checks diagnostic hypotheses against structured graph constraints and
     literature validation, producing an explainable clinical report.

  2. Adaptive Optimizer Agent (run_optimizer_agent)
     Reports the dual-confidence metrics and whether Confidence Score 1 meets
     the threshold. It does not trigger upstream queries or repeat the pipeline.

Includes robust fallback functions so the system runs smoothly even if the
LLM API key is absent or offline.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()
if not os.getenv("OPENROUTER_API_KEY"):
    print("[CONFIG] OPENROUTER_API_KEY is missing. AI calls may fail; fallback output will be used.")


# Shared model connection configured for OpenRouter / OpenAI
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL") or "gpt-4o-mini",
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
- Treat Knowledge Graph disease associations, warning text, and severity labels as source data. Preserve which disease each warning applies to, and do not upgrade, downgrade, or infer severity.
- Preserve distinctions such as "minor or unknown severity," "moderate severity, use with care," and "should NOT be used in this disease." "No known clash" does not mean guaranteed safe.
- Do not add drug safety claims that are absent from the Knowledge Graph inputs.
- Do not introduce treatment drugs that are absent from the evidence inputs. Keep Knowledge Graph treatments labeled as options for a doctor, not prescriptions.
- Preserve each Vector and Evidence Agent hypothesis label. Evidence returned for another hypothesis must not be presented as support for the primary diagnosis.
- Preserve every item in the supplied `kg_safety_messages` list. Include its type, drug, target, severity, and message exactly as supplied; do not infer or change severity.
- Interpret safety types as supplied: `contraindication.target` is a disease; `drug_clash.target` is another current medicine; `treatment_clash.target` is the current medicine it clashes with; `treatment_clash_summary` is a grouped summary for one disease and must not be split into invented individual clashes.
- Treat retrieval-source failures as evidence limitations, not clinical findings. Similarity or retrieval scores are search signals, not diagnostic probabilities.
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

• EVIDENCE LIMITATIONS:
  - <Hypothesis mismatches, retrieval failures, or other supplied limitations>

• PRESENTING SYMPTOMS:
  - <Symptom 1>
  - <Symptom 2>

• EVIDENCE-BASED PHARMACOTHERAPY OPTIONS FOR CLINICIAN CONSIDERATION:
  - <Therapeutic Option 1>: <General guideline usage without fixed milligram dosage>
  - <Therapeutic Option 2>: <General guideline usage without fixed milligram dosage>

• CLINICAL MANAGEMENT STEPS:
  - <Step 1>
  - <Step 2>

• KNOWLEDGE GRAPH MEDICATION SAFETY FINDINGS:
  - <Finding with its disease association and recorded severity preserved>
----------------------------------------------------------------------"""

    response = llm.invoke(prompt)
    return response.content


def run_optimizer_agent(fusion_report: str, confidence_data: Dict[str, Any]) -> str:
    """
    Agent 2: Adaptive Optimizer Agent.
    Evaluates the dual-stream confidence breakdown.

    Threshold reporting:
      - Checks Confidence Score 1 (Vector Track).
      - If C1 >= 0.65 -> Reports that the score meets the threshold.
      - If C1 < 0.65  -> Reports that the score is below the threshold.
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
        f"- Confidence Score 1 (Vector Stream): {c1_score} (Threshold: 0.65) -> {'PASSED' if passes_gate else 'BELOW THRESHOLD'}\n"
        f"- Confidence Score 2 (Graph Stream - HIGH): {c2_score}\n"
        f"- Final Fused Confidence Score: {fused_score}\n"
    )

    if passes_gate:
        instruction = (
            "Confidence Score 1 meets the acceptance threshold (>= 0.65). "
            "Briefly report the scores and their threshold status. Do not claim that medical findings "
            "or treatments are clinically validated."
        )
    else:
        instruction = (
            "Confidence Score 1 is below threshold (< 0.65). "
            "Briefly report the scores and state that this single-pass result is below threshold."
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
    Fallback for run_fusion_agent() if the LLM API is unavailable.
    Reports the failure and preserves only the supplied evidence streams.
    """
    streams = (
        ("Vector input", vector_evidence),
        ("Vector literature input", vector_web_evidence),
        ("Knowledge graph input", graph_evidence),
        ("Knowledge graph literature input", graph_web_evidence),
    )
    supplied = [f"{label}:\n{text}" for label, text in streams if text and text != "No evidence provided."]
    if not supplied:
        return "AI report unavailable"
    return "AI report unavailable\n\nInput data:\n" + "\n\n".join(supplied)


def build_fallback_optimizer(confidence_data: Dict[str, Any], attempt: int) -> str:
    """
    Return a clear unavailable status if run_optimizer_agent() fails.
    """
    return "AI report unavailable"
