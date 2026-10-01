"""
FILE 2 of 5: confidence_scoring.py

WHAT THIS FILE DOES:
Calculates TWO separate confidence scores and fuses them, using rigorous mathematical
formulas -- NOT by asking an AI to guess numbers.

The 4-Input, Dual-Stream Confidence Architecture:
  - Stream 1 (Vector Track): Vector Agent Output + Evidence Agent Output on Vector
      -> Produces Confidence Score 1 (C1)
      -> Compared with the acceptance threshold (0.65) for reporting.
  - Stream 2 (Graph Track): Graph Agent Output + Evidence Agent Output on Graph
      -> Produces Confidence Score 2 (C2)
      -> Reports structured graph fidelity and its supplied literature support.
  - Fusion of C1 and C2:
      -> Produces Final Fused Confidence (C_final) using our recommended
         Dual-Stream Evidential Fusion Formula.

NOTHING in this file calls an LLM or touches the internet. It is pure, auditable Python.
"""

import json
import re
from typing import Dict, Any, Tuple


# --- THRESHOLDS & HYPERPARAMETERS (Aligned with Base Paper Table 2) ---
CONFIDENCE_THRESHOLD = 0.65  # Base paper acceptance threshold (tau = 0.65)

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
    # Structured Vector Agent payloads wrap keys in JSON quotes, so parse the
    # actual retrieval field before falling back to text patterns.
    payload = _load_payload(vector_text)
    def find_retrieval_scores(value: Any):
        if isinstance(value, dict):
            for key in ("relevance_score", "similarity_score", "cosine_similarity", "vector_similarity"):
                score = value.get(key)
                if isinstance(score, (int, float)):
                    yield float(score)
            for nested in value.values():
                yield from find_retrieval_scores(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from find_retrieval_scores(nested)

    scores = list(find_retrieval_scores(payload))
    if scores:
        # Vector retrieval results are normally sorted best-first. Taking the
        # strongest supplied score also handles equivalent unsorted schemas.
        return min(1.0, max(0.0, max(scores)))

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
    text = (web_evidence_text or "").strip().lower()
    # Missing retrieval is missing evidence, not a generic publication.
    if not text or text == "no evidence provided." or text == "no evidence provided":
        return 0.0

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


def _load_payload(text: str) -> Any:
    """Parse normalized JSON text, returning None for ordinary prose."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _condition_from_record(record: Dict[str, Any]) -> str:
    """Read a disease label from common Vector Agent field names."""
    for key in ("condition", "condition_name", "disease", "disease_name", "diagnosis", "name", "label"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _ranked_candidates(payload: Any) -> list:
    """Return candidate records in rank order across common payload layouts."""
    if isinstance(payload, list):
        records = [item for item in payload if isinstance(item, dict) and _condition_from_record(item)]
    elif isinstance(payload, dict):
        records = []
        for key in ("diagnostic_candidates", "candidates", "predictions", "diseases", "diagnoses"):
            value = payload.get(key)
            if isinstance(value, list):
                records.extend(item for item in value if isinstance(item, dict) and _condition_from_record(item))
        if not records:
            direct = _condition_from_record(payload)
            if direct:
                records = [payload]
    else:
        records = []

    def rank_value(record: Dict[str, Any]) -> float:
        for key in ("rank", "position", "index"):
            try:
                return float(record[key])
            except (KeyError, TypeError, ValueError):
                continue
        return float("inf")

    return sorted(records, key=rank_value)


def _hypothesis_records(payload: Any) -> list:
    """Extract labeled evidence groups from common Evidence Agent schemas."""
    found = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return

        label = next((value.get(key) for key in ("hypothesis", "query", "search_term", "condition", "disease")
                      if isinstance(value.get(key), str) and value.get(key).strip()), None)
        evidence = next((value.get(key) for key in ("evidence", "articles", "papers", "results", "items")
                         if isinstance(value.get(key), list)), None)
        if label and evidence is not None:
            found.append({"hypothesis": label.strip(), "record": value})
            return

        for key in ("hypothesis_results", "evidence_results", "results", "hypotheses", "queries"):
            nested = value.get(key)
            if isinstance(nested, (list, dict)):
                visit(nested)

    visit(payload)
    return found


def _normalized_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _label_terms(text: str) -> set:
    qualifiers = {"acute", "chronic", "mild", "moderate", "severe", "recurrent", "unspecified", "suspected"}
    return extract_key_terms(_normalized_label(text)) - qualifiers


def _labels_match(candidate: str, hypothesis: str) -> bool:
    """Conservative lexical match; never infer a medical synonym."""
    primary = _normalized_label(candidate)
    evidence = _normalized_label(hypothesis)
    if not primary or not evidence:
        return False
    if primary == evidence or f" {evidence} " in f" {primary} " or f" {primary} " in f" {evidence} ":
        return True

    primary_terms = _label_terms(primary)
    evidence_terms = _label_terms(evidence)
    shared = primary_terms & evidence_terms
    shortest = min(len(primary_terms), len(evidence_terms))
    return bool(shortest and len(shared) >= 2 and len(shared) / shortest >= 0.75)


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
    g_lower = graph_text.lower()

    conflict_keywords = ["contraindicat", "severe interaction", "incompatible", "unresolved conflict"]
    has_conflict = any(kw in g_lower for kw in conflict_keywords)

    if has_conflict:
        # Check if vector text acknowledges or resolves it
        return -AGREEMENT_BONUS, "Conflict/Contraindication Detected"

    # Award a bonus only when the rank-1 Vector diagnosis is explicitly
    # grounded in the graph text. Shared symptoms or medication names alone
    # do not establish diagnostic agreement.
    vector_payload = _load_payload(vector_text)
    ranked = _ranked_candidates(vector_payload)
    primary_candidate = _condition_from_record(ranked[0]) if ranked else None

    if primary_candidate:
        candidate_terms = extract_key_terms(primary_candidate)
        graph_terms = extract_key_terms(g_lower)
        if _labels_match(primary_candidate, g_lower) or len(candidate_terms & graph_terms) >= 2:
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

    This score is compared with the configured threshold by the Optimizer.
    """
    sim_score = extract_similarity_score(vector_evidence)

    vector_payload = _load_payload(vector_evidence)
    ranked_candidates = _ranked_candidates(vector_payload)
    primary_candidate = _condition_from_record(ranked_candidates[0]) if ranked_candidates else None

    web_payload = _load_payload(vector_web_evidence)
    grouped_evidence = _hypothesis_records(web_payload)
    evidence_hypotheses = [group["hypothesis"] for group in grouped_evidence]
    matched_groups = [
        group for group in grouped_evidence
        if primary_candidate and _labels_match(primary_candidate, group["hypothesis"])
    ]
    matched_hypotheses = list(dict.fromkeys(group["hypothesis"] for group in matched_groups))

    if grouped_evidence and primary_candidate:
        # Only evidence grouped under a matching hypothesis can contribute to
        # the primary candidate's C1 literature and concordance terms.
        if matched_groups:
            matched_texts = [json.dumps(group["record"], ensure_ascii=False) for group in matched_groups]
            web_scores = [score_web_evidence(text) for text in matched_texts]
            web_score = max(web_scores, default=0.0)
            concordance = score_concordance_pairwise(primary_candidate, "\n".join(matched_texts))
        else:
            web_score = 0.0
            concordance = 0.40
    elif web_payload is not None and isinstance(web_payload, dict) and any(
        isinstance(web_payload.get(key), list) for key in ("hypothesis_results", "evidence", "results", "articles")
    ):
        # Structured but unlabeled literature cannot safely be assigned to a
        # diagnosis, so retain it for reporting but do not award C1 support.
        web_score = 0.0
        concordance = 0.40
    else:
        # Backwards-compatible handling for plain prose evidence inputs.
        web_score = score_web_evidence(vector_web_evidence)
        concordance = score_concordance_pairwise(primary_candidate or vector_evidence, vector_web_evidence)

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
        "evidence_hypothesis_check": {
            "primary_candidate": primary_candidate,
            "evidence_hypotheses": evidence_hypotheses,
            "matched_hypotheses": matched_hypotheses,
            "unmatched_hypotheses": [
                hypothesis for hypothesis in evidence_hypotheses
                if hypothesis not in matched_hypotheses
            ],
        },
    }


def compute_confidence_score_2(graph_evidence: str, graph_web_evidence: str) -> Dict[str, Any]:
    """
    Computes Confidence Score 2 (C2):
      Stream 2: Graph Agent Output + Evidence Agent Output on Graph.

    Formula:
      C2 = (w_graph * S_graph) + (w_web * E_web_graph) + (w_conc * Conc_graph)

    C2 reports graph grounding and its supplied literature evidence independently.
    """
    graph_score = extract_graph_score(graph_evidence)
    web_score = score_web_evidence(graph_web_evidence)
    concordance = (
        score_concordance_pairwise(graph_evidence, graph_web_evidence)
        if web_score > 0.0
        else 0.40
    )

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
      - confidence_1: Vector stream results (reported against the threshold)
      - confidence_2: Graph stream results (non-gating)
      - fusion: Details of the fused score
      - final_score: The overall fused confidence
      - passes_threshold: Boolean evaluation of Confidence Score 1 >= 0.65
    """
    c1_dict = compute_confidence_score_1(vector_evidence, vector_web_evidence)
    c2_dict = compute_confidence_score_2(graph_evidence, graph_web_evidence)
    fusion_dict = fuse_dual_confidence(c1_dict, c2_dict, vector_evidence, graph_evidence)

    # Threshold status is based on Confidence Score 1; C2 remains informational.
    passes_gate = c1_dict["passes_threshold"]

    return {
        "confidence_1": c1_dict,
        "confidence_2": c2_dict,
        "fusion": fusion_dict,
        "final_score": fusion_dict["final_fused_score"],
        "passes_threshold": passes_gate,  # Dependent ONLY on C1
    }
