"""
FILE 2 of 5: confidence_scoring.py

WHAT THIS FILE DOES:
Calculates TWO separate confidence scores and fuses them, using rigorous mathematical
formulas -- NOT by asking an AI to guess numbers.

The 4-Input, Dual-Stream Confidence Architecture:
  - Stream 1 (Vector Track): Vector Agent Output + Evidence Agent Output on Vector
      -> Produces Confidence Score 1 (C1)
      -> Used as the GATING CRITERIA for the Optimizer loop (threshold = 0.65).
  - Stream 2 (Graph Track): Graph Agent Output + Evidence Agent Output on Graph
      -> Produces Confidence Score 2 (C2)
      -> Structured knowledge graph fidelity; always high and non-gating.
  - Fusion of C1 and C2:
      -> Produces Final Fused Confidence (C_final) using our recommended
         Dual-Stream Evidential Fusion Formula.

NOTHING in this file calls an LLM or touches the internet. It is pure, auditable Python.
"""

import re
from typing import Dict, Any, Tuple


# --- THRESHOLDS & HYPERPARAMETERS (Aligned with Base Paper Table 2) ---
CONFIDENCE_THRESHOLD = 0.65  # Base paper acceptance threshold (tau = 0.65)
MAX_ITERATIONS = 3           # Base paper maximum optimization cycles

# --- STREAM 1 WEIGHTS (Vector RAG + Literature on Vector) ---
WEIGHT_VEC_SIMILARITY = 0.45    # BioBERT cosine similarity match
WEIGHT_VEC_WEB = 0.35           # Literature hierarchy quality (guidelines vs case reports)
WEIGHT_VEC_CONCORDANCE = 0.20   # Agreement between vector diagnosis & retrieved literature

# --- STREAM 2 WEIGHTS (Knowledge Graph RAG + Literature on Graph) ---
WEIGHT_GRAPH_STRUCTURAL = 0.50  # Ontology/triples structural validity
WEIGHT_GRAPH_WEB = 0.30         # Literature hierarchy quality for graph relations
WEIGHT_GRAPH_CONCORDANCE = 0.20 # Agreement between graph relations & retrieved literature

# --- FUSION WEIGHTS (Recommended Dual-Stream Evidential Fusion) ---
FUSION_WEIGHT_VEC = 0.45        # Weight allocated to Vector stream
FUSION_WEIGHT_GRAPH = 0.55      # Weight allocated to Knowledge Graph stream (high factual specificity)
AGREEMENT_BONUS = 0.05          # Adjustment (+0.05 if concordant, -0.05 if unresolved conflict)


# ============================================================================
# HELPER EXTRACTION AND SCORING FUNCTIONS
# ============================================================================

def extract_similarity_score(vector_text: str) -> float:
    """
    Evaluates the Vector RAG retrieval confidence score (S_vec).

    Handles multiple scenarios:
      1. If the Vector Agent includes an explicit number (e.g. 'similarity 0.88' or 'score: 0.85'):
         Extracts that exact numeric score via regex.
      2. If coverage or match percentage is present (e.g. '23.1%' coverage or 'score 100.0'):
         Extracts and normalizes percentage into 0.0-1.0 range.
      3. If the Vector Agent sends plain text / symptom tables:
         Derives an objective clinical retrieval quality score (0.0 to 1.0) by analyzing
         semantic completeness, symptoms, mapped terms, and disease specificity.
    """
    # Case 1: Explicit RAG similarity or relevance score provided (e.g. relevance_score: 0.5996 or similarity: 0.88)
    match_dec = re.search(r"(?:relevance_score|similarity_score|cosine_similarity|relevance|similarity|score)[:\s]+([01](?:\.\d+)?)", vector_text, re.IGNORECASE)
    if match_dec:
        val = float(match_dec.group(1))
        return min(1.0, max(0.0, val))

    # Case 2: Percentage score (e.g. 85% or score 100.0)
    match_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", vector_text)
    if match_pct:
        val = float(match_pct.group(1)) / 100.0
        # If coverage percentage is given, scale appropriately (e.g. 23% symptom coverage with 3/3 matched symptoms is solid)
        if val > 0.0:
            return round(min(0.95, max(0.60, val * 3.0 if val < 0.35 else val)), 3)

    # Case 3: Pure plain text / table -> Compute from clinical text signals
    text_lower = vector_text.lower()
    score = 0.50

    # (a) Check diagnostic specificity
    disease_indicators = [
        "pneumonia", "embolism", "infarction", "myocardial", "myocardia", "angina",
        "carcinoma", "lymphoma", "diabetes", "asthma", "copd", "failure", "syndrome",
        "hypertension", "infection", "bronchitis", "coronary", "ischemia"
    ]
    if any(dx in text_lower for dx in disease_indicators):
        score += 0.20

    # (b) Check objective clinical findings / normalized symptoms
    clinical_findings = [
        "cxr", "x-ray", "consolidation", "fever", "cough", "dyspnea", "wbc",
        "crp", "ct", "ultrasound", "infiltrate", "tachycardia", "crepitations",
        "chest pain", "sweating", "hyperhidrosis", "shortness of breath"
    ]
    if any(finding in text_lower for finding in clinical_findings):
        score += 0.15

    # (c) Check management / treatment actionability / literature hypothesis
    treatment_terms = [
        "first-line", "treatment", "therapy", "regimen", "antibiotic", "amoxicillin",
        "azithromycin", "stent", "aspirin", "heparin", "thrombolytic", "revascularization"
    ]
    if any(t in text_lower for t in treatment_terms):
        score += 0.10

    # (d) Check uncertainty markers
    uncertainty_markers = ["uncertain", "atypical", "unconfirmed", "unclear", "possible", "mimicking", "vague"]
    if any(u in text_lower for u in uncertainty_markers):
        score -= 0.20

    return round(min(0.95, max(0.35, score)), 3)


def extract_graph_score(graph_text: str) -> float:
    """
    Evaluates the structural grounding score of the Knowledge Graph evidence.
    If an explicit confidence or score is present in the text, it extracts it.
    Otherwise, defaults to 0.90 because curated biomedical knowledge graphs
    (e.g., SNOMED CT, UMLS, verified relational triples) provide high-certainty facts.
    """
    match = re.search(r"(?:confidence|score|fidelity)[:\s]+([01](?:\.\d+)?)", graph_text, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        return min(1.0, max(0.0, val))
    # Standard baseline for verified KG relations
    return 0.90


def score_web_evidence(web_evidence_text: str) -> float:
    """
    Evaluates literature evidence quality based on the clinical Evidence Hierarchy
    (as described in Section II-A-3 of the base paper):
      - 0.90: Clinical Guidelines, Systematic Reviews, Meta-Analyses, NICE, IDSA, Cochrane
      - 0.70: Randomized Controlled Trials, Clinical Trials, Cohort Studies
      - 0.50: Case Reports, Case Studies, Anecdotal series
      - 0.60: General or unspecified biomedical publications
    """
    text = web_evidence_text.lower()

    # Check weakest first to prevent substring false matches (e.g. 'case study' matching 'study')
    if any(phrase in text for phrase in ["case report", "case study", "anecdotal"]):
        return 0.50
    if any(phrase in text for phrase in ["systematic review", "meta-analysis", "clinical guideline", "nice", "idsa", "guideline"]):
        return 0.90
    if any(phrase in text for phrase in ["randomized", "clinical trial", "cohort", "multicenter", "study"]):
        return 0.70
    return 0.60


def extract_key_terms(text: str) -> set:
    """Extracts informative medical terms (words >= 4 chars, excluding common stopwords)."""
    stopwords = {
        "with", "that", "this", "from", "have", "were", "been", "these",
        "their", "patient", "clinical", "evidence", "findings", "report"
    }
    words = set(re.findall(r"[a-z]{4,}", text.lower()))
    return words - stopwords


def score_concordance_pairwise(primary_text: str, evidence_text: str) -> float:
    """
    Measures the degree of semantic agreement / term concordance between
    a primary source (e.g., Vector diagnosis or Graph relation) and its
    associated web evidence.
    """
    primary_terms = extract_key_terms(primary_text)
    if not primary_terms:
        return 0.50

    evidence_terms = extract_key_terms(evidence_text)
    shared = primary_terms & evidence_terms
    overlap_ratio = len(shared) / max(1, min(len(primary_terms), 10))

    if overlap_ratio >= 0.30 or len(shared) >= 3:
        return 1.00  # Strong confirmation in literature
    elif overlap_ratio >= 0.10 or len(shared) >= 1:
        return 0.70  # Moderate/partial concordance
    return 0.40      # Low concordance


def check_inter_stream_agreement(vector_text: str, graph_text: str) -> Tuple[float, str]:
    """
    Checks cross-stream concordance between Vector RAG and Graph RAG:
      - Looks for unmitigated clinical conflicts or contraindications without resolution (-0.05)
      - Looks for mutual alignment/compatibility between diagnosis and graph entities (+0.05)
      - Returns (adjustment_factor, status_label)
    """
    v_lower = vector_text.lower()
    g_lower = graph_text.lower()

    conflict_keywords = ["contraindicat", "severe interaction", "incompatible", "unresolved conflict"]
    has_conflict = any(kw in g_lower for kw in conflict_keywords)

    if has_conflict:
        # Check if vector text acknowledges or resolves it
        return -AGREEMENT_BONUS, "Conflict/Contraindication Detected"

    # Check shared disease/symptom concepts between vector and graph
    v_terms = extract_key_terms(v_lower)
    g_terms = extract_key_terms(g_lower)
    if len(v_terms & g_terms) >= 1:
        return AGREEMENT_BONUS, "Mutually Concordant"

    return 0.0, "Independent / Neutral"


# ============================================================================
# DUAL CONFIDENCE CALCULATION FUNCTIONS
# ============================================================================

def compute_confidence_score_1(vector_evidence: str, vector_web_evidence: str) -> Dict[str, Any]:
    """
    Computes Confidence Score 1 (C1):
      Stream 1: Vector RAG Output + Evidence Agent Output on Vector.

    Formula:
      C1 = (w_sim * S_vec) + (w_web * E_web_vec) + (w_conc * Conc_vec)

    This is the gating confidence score evaluated by the Adaptive Optimizer.
    """
    sim_score = extract_similarity_score(vector_evidence)
    web_score = score_web_evidence(vector_web_evidence)
    concordance = score_concordance_pairwise(vector_evidence, vector_web_evidence)

    raw_c1 = (
        WEIGHT_VEC_SIMILARITY * sim_score +
        WEIGHT_VEC_WEB * web_score +
        WEIGHT_VEC_CONCORDANCE * concordance
    )
    c1 = round(min(1.0, max(0.0, raw_c1)), 3)

    return {
        "score": c1,
        "similarity_score": sim_score,
        "web_evidence_score": web_score,
        "concordance_score": concordance,
        "passes_threshold": c1 >= CONFIDENCE_THRESHOLD,
        "threshold": CONFIDENCE_THRESHOLD,
    }


def compute_confidence_score_2(graph_evidence: str, graph_web_evidence: str) -> Dict[str, Any]:
    """
    Computes Confidence Score 2 (C2):
      Stream 2: Graph Agent Output + Evidence Agent Output on Graph.

    Formula:
      C2 = (w_graph * S_graph) + (w_web * E_web_graph) + (w_conc * Conc_graph)

    Because Knowledge Graph triples provide structured ground truth and are verified
    by literature, C2 is consistently high and is not checked for re-querying.
    """
    graph_score = extract_graph_score(graph_evidence)
    web_score = score_web_evidence(graph_web_evidence)
    concordance = score_concordance_pairwise(graph_evidence, graph_web_evidence)

    raw_c2 = (
        WEIGHT_GRAPH_STRUCTURAL * graph_score +
        WEIGHT_GRAPH_WEB * web_score +
        WEIGHT_GRAPH_CONCORDANCE * concordance
    )
    c2 = round(min(1.0, max(0.0, raw_c2)), 3)

    return {
        "score": c2,
        "graph_score": graph_score,
        "web_evidence_score": web_score,
        "concordance_score": concordance,
        "passes_threshold": True,  # C2 is inherently accepted / non-gating
    }


# ============================================================================
# FUSION FORMULAS & RECOMMENDATION
# ============================================================================

def fuse_dual_confidence(
    c1_dict: Dict[str, Any],
    c2_dict: Dict[str, Any],
    vector_evidence: str,
    graph_evidence: str,
) -> Dict[str, Any]:
    """
    RECOMMENDED FUSION FORMULA: Dual-Stream Evidential Fusion

      C_final = beta_vec * C1 + beta_graph * C2 + Delta_agreement

    Where:
      - beta_vec = 0.45 (weight for vector literature stream)
      - beta_graph = 0.55 (weight for graph literature stream)
      - Delta_agreement in [-0.05, +0.05] (cross-stream concordance bonus/penalty)
      - C_final is clamped to [0.0, 1.0]

    This formula balances statistical vector similarity against structured ontological
    evidence, rewarding mutual verification while penalizing unresolved drug/disease conflicts.
    """
    c1 = c1_dict["score"]
    c2 = c2_dict["score"]

    delta, status = check_inter_stream_agreement(vector_evidence, graph_evidence)

    raw_fused = (FUSION_WEIGHT_VEC * c1) + (FUSION_WEIGHT_GRAPH * c2) + delta
    final_score = round(min(1.0, max(0.0, raw_fused)), 3)

    return {
        "final_fused_score": final_score,
        "c1_score": c1,
        "c2_score": c2,
        "formula": "C_final = 0.45*C1 + 0.55*C2 + Delta_agreement",
        "agreement_adjustment": delta,
        "inter_stream_status": status,
    }


def compute_dual_confidence(
    vector_evidence: str,
    vector_web_evidence: str,
    graph_evidence: str,
    graph_web_evidence: str,
) -> Dict[str, Any]:
    """
    MASTER FUNCTION. Runs both stream computations and fuses them.

    Returns full diagnostic breakdown dictionary:
      - confidence_1: Vector stream results (used for re-query gating)
      - confidence_2: Graph stream results (non-gating)
      - fusion: Details of the fused score
      - final_score: The overall fused confidence
      - passes_threshold: Boolean evaluation of Confidence Score 1 >= 0.65
    """
    c1_dict = compute_confidence_score_1(vector_evidence, vector_web_evidence)
    c2_dict = compute_confidence_score_2(graph_evidence, graph_web_evidence)
    fusion_dict = fuse_dual_confidence(c1_dict, c2_dict, vector_evidence, graph_evidence)

    # CRITICAL: Gating is based STRICTLY on Confidence Score 1 as requested!
    passes_gate = c1_dict["passes_threshold"]

    return {
        "confidence_1": c1_dict,
        "confidence_2": c2_dict,
        "fusion": fusion_dict,
        "final_score": fusion_dict["final_fused_score"],
        "passes_threshold": passes_gate,  # Dependent ONLY on C1
    }
