"""
FILE 4 of 5: graph.py

WHAT THIS FILE DOES:
Structures the Fusion Agent and Optimizer Agent as clean, modular LangGraph NODES.

This allows:
  1. Running this 2-node pipeline standalone for testing (python run.py).
  2. Directly importing `fusion_node` and `optimizer_node` into your team's master
     6-agent LangGraph pipeline with zero friction.

Architecture:
  - Node 1: `fusion_node`    -> Adapts 4 inputs, computes 3 confidence scores (pure Python),
                                generates synthesized clinical report (LLM/fallback).
  - Node 2: `optimizer_node` -> Reports threshold status (LLM/fallback).
  - The standalone graph runs once from fusion_node to optimizer_node and then ends.
"""

import sys
from typing import TypedDict, Any, Dict
from langgraph.graph import StateGraph, END

from evidence_adapter import extract_from_state
from confidence_scoring import compute_dual_confidence
from agents import (
    run_fusion_agent,
    run_optimizer_agent,
    build_fallback_report,
    build_fallback_optimizer,
)


# --- STATE: Shared notebook across all agents in the LangGraph ---
class PipelineState(TypedDict, total=False):
    # Common input keys teammates might send
    raw_vector_evidence: Any
    vector_agent_output: Any
    vector_output: Any
    vector_evidence: Any
    symptom_rag_output: Any

    raw_vector_web_evidence: Any
    evidence_on_vector: Any
    vector_web_evidence: Any
    web_evidence_on_vector: Any
    evidence_vector: Any

    raw_graph_evidence: Any
    graph_agent_output: Any
    graph_output: Any
    graph_evidence: Any
    graph_rag: Any

    raw_graph_web_evidence: Any
    evidence_on_graph: Any
    graph_web_evidence: Any
    web_evidence_on_graph: Any
    evidence_graph: Any

    # Agent outputs
    fusion_report: str
    confidence: Dict[str, Any]
    optimizer_response: str
    attempt: int
    fallback_used: bool


# ============================================================================
# AGENT NODE 1: CLINICAL DATA FUSION AGENT
# ============================================================================
def fusion_node(state: PipelineState) -> Dict[str, Any]:
    """
    LangGraph Node for the Clinical Data Fusion Agent.

    Steps:
      1. Auto-extracts & normalizes the 4 inputs from state (regardless of key names).
      2. Computes the 3 confidence scores (C1, C2, C_fused) using PURE PYTHON (no LLM).
      3. Calls the Fusion LLM to synthesize the multi-source clinical report.
    """
    # Step 1: Normalize whatever keys are in state into clean plain strings
    normalized = extract_from_state(state)
    vec = normalized["vector_evidence"]
    vec_web = normalized["vector_web_evidence"]
    graph = normalized["graph_evidence"]
    graph_web = normalized["graph_web_evidence"]

    # Step 2: Pure Python mathematical confidence calculation (instant, reproducible, no LLM)
    confidence = compute_dual_confidence(
        vector_evidence=vec,
        vector_web_evidence=vec_web,
        graph_evidence=graph,
        graph_web_evidence=graph_web,
    )

    # Step 3: LLM multi-source synthesis report (with error logging fallback)
    fallback_used = False
    try:
        report = run_fusion_agent(
            vector_evidence=vec,
            vector_web_evidence=vec_web,
            graph_evidence=graph,
            graph_web_evidence=graph_web,
        )
    except Exception as error:
        sys.stderr.write(f"[LLM WARNING / ERROR] Fusion Agent AI call failed ({error}). Using deterministic fallback synthesis.\n")
        report = build_fallback_report(vec, vec_web, graph, graph_web)
        fallback_used = True

    return {
        **normalized,
        "fusion_report": report,
        "confidence": confidence,
        "fallback_used": fallback_used,
    }


# ============================================================================
# AGENT NODE 2: ADAPTIVE OPTIMIZER AGENT
# ============================================================================
def optimizer_node(state: PipelineState) -> Dict[str, Any]:
    """
    LangGraph Node for the Adaptive Optimizer Agent.

    Steps:
      1. Reads the confidence dictionary (specifically evaluates Confidence Score 1).
      2. If C1 >= 0.65: Generates FINAL acceptance confirmation.
         If C1 < 0.65:  Reports that the result is below the threshold.
      3. Preserves the attempt state value for integration compatibility.
    """
    attempt = state.get("attempt", 1)
    confidence = state["confidence"]
    fusion_report = state["fusion_report"]

    try:
        response = run_optimizer_agent(fusion_report, confidence)
        fallback_used = state.get("fallback_used", False)
    except Exception as error:
        sys.stderr.write(f"[LLM WARNING / ERROR] Optimizer Agent AI call failed ({error}). Using deterministic fallback response.\n")
        response = build_fallback_optimizer(confidence, attempt)
        fallback_used = True

    return {
        "optimizer_response": response,
        "attempt": attempt,
        "fallback_used": fallback_used,
    }


# ============================================================================
# STANDALONE LANGGRAPH BUILDER
# ============================================================================
def build_pipeline():
    """
    Builds the clean 2-node LangGraph pipeline:
        fusion_node -> optimizer_node -> END
    """
    graph = StateGraph(PipelineState)

    # 1. Add the two agent nodes
    graph.add_node("fusion_node", fusion_node)
    graph.add_node("optimizer_node", optimizer_node)

    # Connect the nodes in one pass and end.
    graph.set_entry_point("fusion_node")
    graph.add_edge("fusion_node", "optimizer_node")
    graph.add_edge("optimizer_node", END)

    return graph.compile()
