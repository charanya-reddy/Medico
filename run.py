"""
FILE 5 of 5: run.py

WHAT THIS FILE DOES:
Executes the Clinical Data Fusion & Adaptive Optimizer pipeline.

Outputs a clean, doctor- and patient-focused medical diagnosis and treatment report,
prefixed by a 1-2 line system status summary at the very top.
"""

import sys
import io

# Ensure UTF-8 output encoding across Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from graph import build_pipeline


# ============================================================================
# REALISTIC CLINICAL TEST DATA (MATCHING TEAM AGENT INPUT FORMATS)
# ============================================================================

# INPUT 1: Symptom RAG / Vector Output (Symptom Normalization & Candidate Table)
GRAPH_SYMPTOM_RAG_INPUT = """======================================================================
INPUT SYMPTOMS
======================================================================
  - chest pain
  - sweating
  - shortness of breath

======================================================================
NORMALIZED MAPPING APPLIED
======================================================================
  'chest pain'                   -> 'Chest pain'  
  'sweating'                     -> 'Hyperhidrosis'  
  'shortness of breath'          -> 'Dyspnea'

======================================================================
TOP CANDIDATE DISEASES (of 347 total)
======================================================================
 matched   coverage   total_symptoms  disease
       3      35.0%                9  Myocardial Infarction
       3      23.1%               13  thymic carcinoma
       3      17.6%               17  Hodgkins lymphoma"""

# INPUT 2: Evidence Agent Output (Structured JSON schema with hypothesis & evidence array)
EVIDENCE_AGENT_INPUT = {
    "hypothesis": "myocardial infarction",
    "broadened": False,
    "evidence": [
        {
            "tier_label": "Systematic Review / Meta-Analysis",
            "score": 100.0,
            "title": "A meta-analysis and systematic review of myocardial infarction-induced cardiomyocyte proliferation in adult mouse heart.",
            "summary": "The proliferation capacity of adult cardiomyocytes is very limited in the normal adult mammalian heart.",
            "url": "https://pubmed.ncbi.nlm.nih.gov/39736615/",
            "source": "PubMed"
        },
        {
            "tier_label": "Systematic Review / Meta-Analysis",
            "score": 100.0,
            "title": "Comparison of Long-Term Outcomes of Patients With Myocardia Infarction (MI) With Non-obstructive Coronary Arteries and MI With Obstructive Coronary Arteries: A Systematic Review and Meta-Analysis.",
            "summary": "The aim of this study was to compare long-term outcomes in patients with myocardial infarction with non-obstructive coronary arteries (MINOCA) and patients with myocardial infarction with obstructive coronary arteries (MIOCA).",
            "url": "https://pubmed.ncbi.nlm.nih.gov/37692745/",
            "source": "PubMed"
        }
    ]
}

# INPUT 3: Knowledge Graph Relational Safety Constraints
GRAPH_EVIDENCE_INPUT = {
    "summary": (
        "Knowledge Graph Traversal: Patient presents with acute Chest Pain, Hyperhidrosis, and Dyspnea. "
        "Primary relational entity: Myocardial Infarction / Acute Coronary Syndrome. "
        "Recommended acute pharmacotherapy: Dual antiplatelet therapy (Aspirin + P2Y12 inhibitor), "
        "Sublingual Nitroglycerin, and Unfractionated Heparin."
    ),
    "relations": [
        "Patient -> presents_with -> Chest Pain (angina)",
        "Patient -> presents_with -> Hyperhidrosis (sweating)",
        "Patient -> presents_with -> Dyspnea (shortness of breath)",
        "Myocardial Infarction -> first_line_medication -> Aspirin 325mg + Clopidogrel 300mg",
        "Myocardial Infarction -> contraindicates -> NSAID monotherapy without gastroprotection"
    ],
    "structural_confidence": 0.94
}

# INPUT 4: Literature Evidence on Graph Findings
GRAPH_WEB_EVIDENCE_INPUT = (
    "ACC/AHA & ESC Clinical Guidelines: Prompt administration of chewing aspirin (162-325 mg) "
    "and P2Y12 inhibitor upon presentation with suspected acute coronary syndrome significantly reduces mortality. "
    "Immediate emergency ECG and troponin diagnostic confirmation mandated."
)


# ============================================================================
# OPTIONAL LOW-CONFIDENCE SIMULATION
# ============================================================================
LOW_CONFIDENCE_INPUT = "Vague pleuritic discomfort and fatigue. Uncertain etiology."
LOW_CONFIDENCE_EVIDENCE = ["Anecdotal case report on atypical discomfort."]


def main():
    test_loop = "--test-loop" in sys.argv or "-t" in sys.argv

    pipeline = build_pipeline()

    initial_state = {
        "raw_vector_evidence": LOW_CONFIDENCE_INPUT if test_loop else GRAPH_SYMPTOM_RAG_INPUT,
        "raw_vector_web_evidence": LOW_CONFIDENCE_EVIDENCE if test_loop else EVIDENCE_AGENT_INPUT,
        "raw_graph_evidence": GRAPH_EVIDENCE_INPUT,
        "raw_graph_web_evidence": GRAPH_WEB_EVIDENCE_INPUT,
        "attempt": 1,
    }

    final_state = pipeline.invoke(initial_state)

    # Extract confidence scores
    conf = final_state["confidence"]
    c1 = conf["confidence_1"]["score"]
    c2 = conf["confidence_2"]["score"]
    passes = conf["confidence_1"]["passes_threshold"]
    attempts = final_state.get("attempt", 1) - 1

    # Print 1-2 line concise system status at the VERY TOP as requested
    print("=" * 70)
    if passes:
        print(f"[STATUS] Validation Complete | Dual Confidence Scores: C1 = {c1} (Passed >= 0.65), C2 = {c2} | Re-query Iterations Required: {attempts}")
    else:
        print(f"[STATUS] Low Confidence Detected | Dual Confidence: C1 = {c1} (Failed < 0.65), C2 = {c2} | Re-query Iterations Triggered: {attempts + 1}")
    print("=" * 70)

    # Print Doctor & Patient-facing Clinical Diagnosis & Treatment Plan
    print("\n" + final_state["fusion_report"])


if __name__ == "__main__":
    main()
