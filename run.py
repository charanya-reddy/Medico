"""
FILE 5 of 5: run.py

WHAT THIS FILE DOES:
Executes the Clinical Data Fusion & Adaptive Optimizer pipeline.

Outputs a concise, doctor- and patient-focused medical diagnosis and treatment report,
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
# REALISTIC CLINICAL TEST DATA (EXACT TEAM AGENT OUTPUT SCHEMAS)
# ============================================================================

# INPUT 1: Symptom RAG / Vector Database Agent Output (Structured JSON dictionary)
GRAPH_SYMPTOM_RAG_INPUT = {
    "agent": "symptom_rag",
    "status": "success",
    "query_context": {
        "diseases": ["Myocardial Infarction"],
        "symptoms": ["chest pain", "sweating", "shortness of breath"],
        "medications": ["Sildenafil"],
        "tests": [],
        "procedures": []
    },
    "diagnostic_candidates": [
        {
            "condition": "Myocardial Infarction (Heart Attack)",
            "rank": 1,
            "justification": "Evidence 2 explicitly describes classic signs and symptoms of a heart attack as crushing substernal chest pain, shortness of breath, and sweating, directly matching the patient's presentation.",
            "supporting_evidence": [2, 5]
        },
        {
            "condition": "Heart Arrhythmia",
            "rank": 2,
            "justification": "Evidence 1 identifies chest pain or discomfort and shortness of breath as potential symptoms of heart arrhythmias.",
            "supporting_evidence": [1]
        },
        {
            "condition": "Valvular Heart Disease",
            "rank": 3,
            "justification": "Evidence 4 notes that valvular heart disease can present with symptoms including chest pain and shortness of breath.",
            "supporting_evidence": [4]
        }
    ],
    "evidence": [
        {
            "content": "Classic signs and symptoms of a heart attack include crushing, substernal chest pain, pain in your shoulders or arms, shortness of breath, and sweating.",
            "source": "symptom-disease-train-dataset.csv",
            "source_type": "symptom-disease-dataset",
            "relevance_score": 0.5996
        }
    ],
    "metadata": {"evidence_count": 5},
    "limitations": "The retrieved passages do not contain information regarding the clinical risks or drug interactions associated with Sildenafil in a patient presenting with myocardial infarction.",
    "error": None
}

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
        "Critical Medication Alert: Patient is currently taking Sildenafil. "
        "Severe Contraindication: Concomitant administration of Nitrates (e.g. Sublingual Nitroglycerin) with Sildenafil is strictly contra-indicated due to risk of fatal refractory hypotension."
    ),
    "relations": [
        "Patient -> presents_with -> Chest Pain (angina)",
        "Patient -> presents_with -> Hyperhidrosis (sweating)",
        "Patient -> presents_with -> Dyspnea (shortness of breath)",
        "Patient -> current_medication -> Sildenafil",
        "Sildenafil -> severe_contraindication -> Nitroglycerin (fatal hypotension risk)"
    ],
    "structural_confidence": 0.95
}

# INPUT 4: Literature Evidence on Graph Findings
GRAPH_WEB_EVIDENCE_INPUT = (
    "FDA Drug Safety Warning & AHA Guidelines: Co-administration of PDE5 inhibitors (Sildenafil) "
    "with organic nitrates (Nitroglycerin) produces severe, life-threatening hypotension. "
    "Nitrates must not be administered within 24 hours of Sildenafil use."
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
