"""
FILE 1 of 5: evidence_adapter.py

WHAT THIS FILE DOES:
In the 6-agent clinical pipeline, the Fusion Agent receives FOUR inputs:
  1. Vector Agent Output (Symptom RAG candidates from BioBERT vector search)
  2. Evidence Agent Output on Vector (web literature/guidelines validating the vector candidates)
  3. Graph Agent Output (Knowledge Graph RAG relations, interactions, contraindications)
  4. Evidence Agent Output on Graph (web literature/guidelines validating the graph findings)

Because upstream agents and teammates might output data in varying shapes
(plain string, JSON/dict with different keys, list of findings, etc.) AND
with different variable names, this adapter:
  1. Detects whichever key names teammates use in the LangGraph state.
  2. Normalizes ANY input shape into clean, readable plain text.

Nothing downstream ever needs to worry about input formats.
"""

from typing import Any, Dict, List, Union


def to_plain_text(raw_evidence: Any) -> str:
    """
    Takes ANYTHING a teammate's agent might return, and turns it into
    one clean, standardized plain string.

    Handles:
      - Plain string                        -> returned trimmed
      - Dictionary (JSON object)            -> extracts recognized text keys
                                               ('text', 'summary', 'content', 'answer',
                                                'result', 'findings', 'relations', 'evidence', 'diagnosis')
      - List of strings or dictionaries     -> formats and joins items
      - None / empty string                  -> returns safe informative placeholder
      - Any other data type                 -> safe string conversion without crashing
    """
    if raw_evidence is None or raw_evidence == "":
        return "No evidence provided."

    # Case 1: Plain string
    if isinstance(raw_evidence, str):
        return raw_evidence.strip()

    # Case 2: Dictionary
    if isinstance(raw_evidence, dict):
        # Specific handler for Evidence Agent structured JSON schema
        if "evidence" in raw_evidence and isinstance(raw_evidence["evidence"], list):
            hypo = raw_evidence.get("hypothesis", "")
            lines = [f"Hypothesis Investigated: {hypo}"] if hypo else []
            for idx, item in enumerate(raw_evidence["evidence"], 1):
                if isinstance(item, dict):
                    title = item.get("title", "")
                    tier = item.get("tier_label", "")
                    summary = item.get("summary", "")
                    score = item.get("score", "")
                    source = item.get("source", "")
                    lines.append(f"[{idx}] {tier} (Score: {score}) - {title}\nSummary: {summary} (Source: {source})")
                else:
                    lines.append(f"- {to_plain_text(item)}")
            return "\n\n".join(lines)

        # Look for standard keys holding textual descriptions
        for key in ("text", "summary", "content", "answer", "result", "findings", "diagnosis", "output"):
            if key in raw_evidence and raw_evidence[key]:
                return to_plain_text(raw_evidence[key])

        # If it's a structured graph dictionary with a list of relations or entities
        if "relations" in raw_evidence and isinstance(raw_evidence["relations"], list):
            rel_lines = [f"- {to_plain_text(r)}" for r in raw_evidence["relations"]]
            base_summary = raw_evidence.get("summary", "")
            if base_summary:
                return f"{base_summary}\nRelations:\n" + "\n".join(rel_lines)
            return "Knowledge Graph Relations:\n" + "\n".join(rel_lines)

        # Fallback: readable key-value dump so no information is lost
        kv_pairs = [f"{k}: {to_plain_text(v)}" for k, v in raw_evidence.items() if v]
        return "\n".join(kv_pairs)

    # Case 3: List
    if isinstance(raw_evidence, list):
        pieces = [to_plain_text(item) for item in raw_evidence]
        return "\n".join(f"- {p}" if not p.startswith("- ") else p for p in pieces if p)

    # Case 4: Any unexpected object
    return str(raw_evidence).strip()


def normalize_all(
    vector_evidence: Any,
    vector_web_evidence: Any,
    graph_evidence: Any,
    graph_web_evidence: Any,
) -> Dict[str, str]:
    """
    Convenience function: runs all 4 input streams through the adapter at once.

    Returns:
      A dictionary with 4 clean plain-text strings:
        - "vector_evidence"
        - "vector_web_evidence"
        - "graph_evidence"
        - "graph_web_evidence"
    """
    return {
        "vector_evidence": to_plain_text(vector_evidence),
        "vector_web_evidence": to_plain_text(vector_web_evidence),
        "graph_evidence": to_plain_text(graph_evidence),
        "graph_web_evidence": to_plain_text(graph_web_evidence),
    }


def extract_from_state(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Auto-detects whatever variable names teammates choose to put in the LangGraph state.
    Allows teammates to use any common alias without breaking the pipeline.
    """
    VECTOR_ALIASES = [
        "raw_vector_evidence", "vector_agent_output", "vector_output",
        "vector_evidence", "vector_rag", "vector_agent", "symptom_rag_output"
    ]
    EVIDENCE_ON_VECTOR_ALIASES = [
        "raw_vector_web_evidence", "evidence_on_vector", "vector_web_evidence",
        "web_evidence_on_vector", "evidence_vector", "web_evidence_vector", "evidence_rag"
    ]
    GRAPH_ALIASES = [
        "raw_graph_evidence", "graph_agent_output", "graph_output",
        "graph_evidence", "graph_rag", "graph_agent"
    ]
    EVIDENCE_ON_GRAPH_ALIASES = [
        "raw_graph_web_evidence", "evidence_on_graph", "graph_web_evidence",
        "web_evidence_on_graph", "evidence_graph", "web_evidence_graph"
    ]

    def find_first(aliases: List[str]) -> Any:
        for key in aliases:
            if key in state and state[key] is not None and state[key] != "":
                return state[key]
        return None

    raw_vec = find_first(VECTOR_ALIASES)
    raw_vec_web = find_first(EVIDENCE_ON_VECTOR_ALIASES)
    raw_graph = find_first(GRAPH_ALIASES)
    raw_graph_web = find_first(EVIDENCE_ON_GRAPH_ALIASES)

    return normalize_all(raw_vec, raw_vec_web, raw_graph, raw_graph_web)
