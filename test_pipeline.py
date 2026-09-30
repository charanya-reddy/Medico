"""
Unit test suite for the 4-Input Dual-Confidence Fusion & Optimizer CDSS Pipeline.
"""

import unittest
from evidence_adapter import to_plain_text, normalize_all
from confidence_scoring import (
    compute_confidence_score_1,
    compute_confidence_score_2,
    fuse_dual_confidence,
    compute_dual_confidence,
    CONFIDENCE_THRESHOLD,
    MAX_ITERATIONS,
)
from graph import build_pipeline


class TestEvidenceAdapter(unittest.TestCase):
    def test_string_input(self):
        text = "Pneumonia with similarity 0.85"
        self.assertEqual(to_plain_text(text), text)

    def test_dict_input(self):
        data = {"summary": "Patient has AFib", "score": 0.9}
        self.assertEqual(to_plain_text(data), "Patient has AFib")

    def test_list_input(self):
        data = ["Guideline A", "Trial B"]
        res = to_plain_text(data)
        self.assertIn("Guideline A", res)
        self.assertIn("Trial B", res)

    def test_none_and_empty(self):
        self.assertEqual(to_plain_text(None), "No evidence provided.")
        self.assertEqual(to_plain_text(""), "No evidence provided.")

    def test_normalize_all_4_inputs(self):
        res = normalize_all(
            vector_evidence="Vector text",
            vector_web_evidence=["Guideline 1", "Guideline 2"],
            graph_evidence={"summary": "Graph text"},
            graph_web_evidence="FDA text",
        )
        self.assertEqual(res["vector_evidence"], "Vector text")
        self.assertIn("Guideline 1", res["vector_web_evidence"])
        self.assertEqual(res["graph_evidence"], "Graph text")
        self.assertEqual(res["graph_web_evidence"], "FDA text")


class TestConfidenceScoring(unittest.TestCase):
    def test_confidence_1_high(self):
        vec = "Community-acquired pneumonia (similarity 0.90)."
        web = "Clinical guideline: Amoxicillin recommended first-line for pneumonia."
        c1 = compute_confidence_score_1(vec, web)
        self.assertGreaterEqual(c1["score"], CONFIDENCE_THRESHOLD)
        self.assertTrue(c1["passes_threshold"])

    def test_confidence_1_low(self):
        vec = "Unknown condition (similarity 0.40)."
        web = "Anecdotal case report on rare presentation."
        c1 = compute_confidence_score_1(vec, web)
        self.assertLess(c1["score"], CONFIDENCE_THRESHOLD)
        self.assertFalse(c1["passes_threshold"])

    def test_confidence_2_always_passes(self):
        graph = "Knowledge graph traversal: Warfarin interaction with amoxicillin."
        web = "Systematic review confirming increased INR."
        c2 = compute_confidence_score_2(graph, web)
        self.assertTrue(c2["passes_threshold"])
        self.assertGreaterEqual(c2["score"], 0.80)

    def test_dual_confidence_gating(self):
        # Case where C1 passes
        res_pass = compute_dual_confidence(
            vector_evidence="Pneumonia (similarity 0.92)",
            vector_web_evidence="Clinical guideline and meta-analysis on pneumonia",
            graph_evidence="Atrial fibrillation on warfarin",
            graph_web_evidence="Systematic review of warfarin monitoring",
        )
        self.assertTrue(res_pass["passes_threshold"])
        self.assertIn("final_fused_score", res_pass["fusion"])

        # Case where C1 fails (even if C2 is very high)
        res_fail = compute_dual_confidence(
            vector_evidence="Undifferentiated fever (similarity 0.35)",
            vector_web_evidence="Anecdotal case report",
            graph_evidence="Known AFib managed with warfarin (confidence: 0.95)",
            graph_web_evidence="Systematic review of anticoagulants",
        )
        # Gating is strictly C1
        self.assertFalse(res_fail["passes_threshold"])
        self.assertLess(res_fail["confidence_1"]["score"], CONFIDENCE_THRESHOLD)
        self.assertTrue(res_fail["confidence_2"]["passes_threshold"])


class TestLangGraphPipeline(unittest.TestCase):
    def test_pipeline_build(self):
        pipeline = build_pipeline()
        self.assertIsNotNone(pipeline)


if __name__ == "__main__":
    unittest.main()
