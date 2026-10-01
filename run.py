"""
FILE 5 of 5: run.py

WHAT THIS FILE DOES:
Executes the Clinical Data Fusion & Adaptive Optimizer pipeline.

Prints pipeline status and the fusion report or its unavailable message.
"""

import json
import sys
from pathlib import Path

# Ensure UTF-8 output encoding across Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from graph import build_pipeline


# Optional low-confidence sample data for a single-pass run.
LOW_CONFIDENCE_INPUT = "Vague pleuritic discomfort and fatigue. Uncertain etiology."
LOW_CONFIDENCE_EVIDENCE = ["Anecdotal case report on atypical discomfort."]


def main():
    low_confidence_demo = "--low-confidence" in sys.argv or "-l" in sys.argv

    pipeline = build_pipeline()

    initial_state = json.loads(
        Path(__file__).with_name("example_input.json").read_text(encoding="utf-8")
    )
    if low_confidence_demo:
        initial_state["raw_vector_evidence"] = LOW_CONFIDENCE_INPUT
        initial_state["raw_vector_web_evidence"] = LOW_CONFIDENCE_EVIDENCE
    initial_state.setdefault("attempt", 1)

    final_state = pipeline.invoke(initial_state)

    # Extract confidence scores
    conf = final_state["confidence"]
    c1 = conf["confidence_1"]["score"]
    c2 = conf["confidence_2"]["score"]
    passes = conf["confidence_1"]["passes_threshold"]
    # Print concise pipeline status.
    print("=" * 70)
    if passes:
        print(f"[STATUS] Confidence check passed | C1 = {c1} (>= 0.65), C2 = {c2}")
    else:
        print(f"[STATUS] Confidence check below threshold | C1 = {c1} (< 0.65), C2 = {c2}")
    print(f"[STATUS] Fallback used: {str(final_state.get('fallback_used', False)).lower()}")
    print("=" * 70)

    # Print Doctor & Patient-facing Clinical Diagnosis & Treatment Plan
    print("\n" + final_state["fusion_report"])


if __name__ == "__main__":
    main()
