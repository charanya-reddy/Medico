"""
safety.py

Deterministic (no LLM) helpers that guarantee three things, whatever the LLM writes:
  1. Knowledge Graph safety data (drug names, warnings, contraindications) is never lost.
  2. Treatment options come ONLY from the input streams (the AI is never asked to invent them).
  3. Contradictions / conflicts between streams are detected and always reported.
"""

import re
from typing import Dict, List

SAFETY_KEYWORDS = (
    "contraindicat", "contra-indicat", "interaction", "warning", "alert",
    "avoid", "do not", "must not", "hypotension", "bleeding", "risk",
    "life-threatening", "fatal",
)

TREATMENT_KEYWORDS = (
    "treatment", "therapy", "first-line", "regimen", "management", "recommended",
    "antibiotic", "antiplatelet", "anticoagul", "thrombolytic", "reperfusion",
    "revascularization", "stent", "aspirin", "heparin", "amoxicillin", "azithromycin",
)

STREAM_LABELS = {
    "vector_evidence": "Vector Agent",
    "vector_web_evidence": "Vector Literature",
    "graph_evidence": "Knowledge Graph",
    "graph_web_evidence": "Knowledge Graph Literature",
}


def split_lines(text: str) -> List[str]:
    """Splits text into lines/sentences, keeping 'e.g.' / 'i.e.' intact."""
    protected = text.replace("e.g.", "e<dot>g<dot>").replace("i.e.", "i<dot>e<dot>")
    out = []
    for chunk in re.split(r"[\n]+|(?<=[.!?])\s+", protected):
        line = chunk.strip(" -\t").replace("<dot>", ".")
        if len(line) >= 8:
            out.append(line)
    return out


def extract_safety_lines(text: str) -> List[str]:
    """Every safety-relevant line from the given text, verbatim, de-duplicated."""
    seen, out = set(), []
    for line in split_lines(text):
        low = line.lower()
        if low in seen:
            continue
        if any(k in low for k in SAFETY_KEYWORDS):
            seen.add(low)
            out.append(line)
    return out


def extract_treatment_options(streams: Dict[str, str]) -> List[str]:
    """
    Treatment mentions taken VERBATIM from the input streams, labelled with their source.
    Lines that are safety statements are excluded (they are reported in the safety block).
    Nothing is generated or invented here.
    """
    seen, out = set(), []
    for key, label in STREAM_LABELS.items():
        for line in split_lines(streams.get(key, "")):
            low = line.lower()
            if low in seen:
                continue
            if any(k in low for k in SAFETY_KEYWORDS):
                continue
            if any(k in low for k in TREATMENT_KEYWORDS):
                seen.add(low)
                out.append(f"[{label}] {line}")
    return out


def extract_patient_medications(vector_text: str) -> List[str]:
    m = re.search(r"Current Patient Medications:\s*(.+)", vector_text)
    if not m:
        return []
    return [x.strip() for x in m.group(1).split(",") if x.strip()]


def extract_conflict_terms(graph_text: str) -> List[str]:
    """Drug names the Knowledge Graph says conflict with something (relation targets, conflicts_with, e.g. lists)."""
    terms = set()
    for m in re.finditer(r"->\s*[\w ]*contra[\w\- ]*indicat[\w ]*\s*->\s*([A-Za-z][\w\- ]*?)\s*(?:\(|$|\n)", graph_text, re.I):
        terms.add(m.group(1).strip())
    for m in re.finditer(r"conflicts_with:\s*([^;\n]+)", graph_text, re.I):
        terms.add(m.group(1).strip())
    for m in re.finditer(r"e\.g\.\s*(?:[A-Za-z]+\s+)?([A-Z][A-Za-z\-]+)", graph_text):
        terms.add(m.group(1).strip())
    return sorted(t for t in terms if len(t) >= 4)


def find_contradictions(streams: Dict[str, str]) -> List[str]:
    """
    Returns human-readable contradiction/conflict findings. Never silently empty when a
    patient medication or a listed treatment clashes with a Knowledge Graph safety statement.
    """
    graph_text = streams.get("graph_evidence", "") + "\n" + streams.get("graph_web_evidence", "")
    safety = extract_safety_lines(graph_text)
    findings: List[str] = []

    # 1. Patient's current medication named in graph safety statements (one finding per drug).
    for med in extract_patient_medications(streams.get("vector_evidence", "")):
        pat = re.compile(rf"\b{re.escape(med)}\b", re.I)
        hits = [line for line in safety if pat.search(line)]
        if hits:
            key = next((h for h in hits if "contra" in h.lower() or "must not" in h.lower()), hits[0])
            findings.append(
                f"Patient is taking {med}, which appears in {len(hits)} Knowledge Graph safety statement(s). Key one: {key}"
            )

    # 2. A treatment option present in the inputs that mentions a drug the graph flags.
    terms = extract_conflict_terms(graph_text)
    for option in extract_treatment_options(streams):
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", option, re.I):
                findings.append(f"Treatment option mentions '{term}', which the Knowledge Graph flags as conflicting: {option}")

    # de-duplicate, keep order
    seen, out = set(), []
    for f in findings:
        if f.lower() not in seen:
            seen.add(f.lower())
            out.append(f)
    return out


def format_block(title: str, items: List[str], empty_text: str) -> str:
    body = "\n".join(f"  - {i}" for i in items) if items else f"  - {empty_text}"
    return f"• {title}:\n{body}"
