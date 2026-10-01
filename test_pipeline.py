"""
Unit test suite for the 4-Input Dual-Confidence Fusion & Optimizer CDSS Pipeline.
"""

import json
import unittest
from unittest.mock import patch
from evidence_adapter import to_plain_text, normalize_all
from confidence_scoring import (
    check_inter_stream_agreement,
    compute_confidence_score_1,
    compute_confidence_score_2,
    extract_similarity_score,
    score_web_evidence,
    fuse_dual_confidence,
    compute_dual_confidence,
    CONFIDENCE_THRESHOLD,
)
from graph import build_pipeline, fusion_node


class TestEvidenceAdapter(unittest.TestCase):
    def test_string_input(self):
        text = "Pneumonia with similarity 0.85"
        self.assertEqual(to_plain_text(text), text)

    def test_dict_input(self):
        data = {"summary": "Patient has AFib", "score": 0.9}
        result = to_plain_text(data)
        self.assertIn("Patient has AFib", result)

    def test_list_input(self):
        data = ["Guideline A", "Trial B"]
        res = to_plain_text(data)
        self.assertIn("Guideline A", res)
        self.assertIn("Trial B", res)

    def test_none_and_empty(self):
        self.assertEqual(to_plain_text(None), "No evidence provided.")
        self.assertEqual(to_plain_text(""), "No evidence provided.")

    def test_kg_safety_data_preserved(self):
        """Regression test: KG safety fields must NOT be dropped."""
        data = {
            "summary": "Patient presents with MI. Sildenafil in record.",
            "relations": [
                "Patient -> current_medication -> Sildenafil",
                "Sildenafil -> severe_contraindication -> Nitroglycerin",
            ],
            "contraindications": "Nitrate co-administration with Sildenafil is strictly contra-indicated.",
            "warnings": "Risk of fatal refractory hypotension.",
        }
        result = to_plain_text(data)
        self.assertIn("Sildenafil", result)
        self.assertIn("Nitroglycerin", result)
        self.assertIn("contra-indicated", result)
        self.assertIn("hypotension", result)
        self.assertIn("Knowledge Graph Relations", result)

    def test_normalize_all_4_inputs(self):
        res = normalize_all(
            vector_evidence="Vector text",
            vector_web_evidence=["Guideline 1", "Guideline 2"],
            graph_evidence={"summary": "Graph text"},
            graph_web_evidence="FDA text",
        )
        self.assertEqual(res["vector_evidence"], "Vector text")
        self.assertIn("Guideline 1", res["vector_web_evidence"])
        self.assertIn("Graph text", res["graph_evidence"])
        self.assertEqual(res["graph_web_evidence"], "FDA text")

    def test_structured_agent_payloads_keep_all_fields(self):
        vector = {
            "status": "success",
            "query_context": {"diseases": ["MI"], "tests": []},
            "diagnostic_candidates": [{"condition": "Myocardial Infarction", "rank": 1}],
            "evidence": [{"content": "finding", "url": "https://example.test", "relevance_score": 0.7}],
            "metadata": {"evidence_count": 1},
        }
        web = {
            "hypothesis_results": [{"hypothesis": "type 2 diabetes", "evidence": [{"url": "https://example.test"}]}],
            "retrieval_warnings": ["HTTP 429; continuing"],
        }
        normalized = normalize_all(vector, web, "KG report", "")
        self.assertEqual(json.loads(normalized["vector_evidence"]), vector)
        self.assertEqual(json.loads(normalized["vector_web_evidence"]), web)

    def test_list_of_grouped_evidence_remains_structured(self):
        groups = [{
            "query": "pulmonary embolism",
            "articles": [{"title": "Clinical guideline", "url": "https://example.test"}],
        }]
        normalized = normalize_all("Vector result", groups, "Graph result", "")
        self.assertEqual(json.loads(normalized["vector_web_evidence"]), groups)


class TestConfidenceScoring(unittest.TestCase):
    def test_missing_web_evidence_scores_zero(self):
        self.assertEqual(score_web_evidence(""), 0.0)
        self.assertEqual(score_web_evidence("No evidence provided."), 0.0)

    def test_structured_vector_uses_actual_retrieval_score(self):
        vector = json.dumps({"evidence": [{"relevance_score": 0.5996}], "diagnostic_candidates": []})
        self.assertEqual(extract_similarity_score(vector), 0.5996)

    def test_shared_medication_does_not_create_diagnostic_agreement(self):
        vector = json.dumps({"diagnostic_candidates": [{"condition": "Pneumonia", "rank": 1}]})
        graph = "Known disease: myocardial infarction. Current medication: Sildenafil."
        self.assertEqual(check_inter_stream_agreement(vector, graph), (0.0, "Independent / Neutral"))

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

    def test_unmatched_evidence_hypotheses_do_not_validate_primary_candidate(self):
        vector = json.dumps({
            "diagnostic_candidates": [{"condition": "Myocardial Infarction", "rank": 1}],
            "evidence": [{"relevance_score": 0.7}],
        })
        web = json.dumps({
            "hypothesis_results": [{
                "hypothesis": "type 2 diabetes",
                "evidence": [{"tier_label": "Systematic Review / Meta-Analysis"}],
            }],
            "retrieval_warnings": [],
        })
        c1 = compute_confidence_score_1(vector, web)
        self.assertEqual(c1["web_evidence_score"], 0.0)
        self.assertEqual(c1["evidence_hypothesis_check"]["matched_hypotheses"], [])
        self.assertFalse(c1["passes_threshold"])

    def test_equivalent_agent_schema_matches_dynamic_diagnosis(self):
        vector = json.dumps({
            "candidates": [{"disease": "Pulmonary Embolism", "position": 1}],
            "evidence": [{"similarity_score": 0.82}],
        })
        web = json.dumps({
            "hypothesis": "acute pulmonary embolism",
            "evidence": [{"tier_label": "Clinical Guideline", "title": "Pulmonary embolism guideline"}],
        })
        c1 = compute_confidence_score_1(vector, web)
        self.assertEqual(c1["evidence_hypothesis_check"]["primary_candidate"], "Pulmonary Embolism")
        self.assertEqual(c1["evidence_hypothesis_check"]["matched_hypotheses"], ["acute pulmonary embolism"])
        self.assertGreater(c1["web_evidence_score"], 0.0)
        self.assertEqual(c1["similarity_score"], 0.82)

    def test_grouped_hypothesis_results_schema_matches_candidate(self):
        vector = json.dumps({
            "diagnostic_candidates": [{"condition": "Myocardial Infarction (Heart Attack)", "rank": 1}],
            "evidence": [{"relevance_score": 0.6}],
        })
        web = json.dumps({"hypothesis_results": [
            {"hypothesis": "acute myocardial infarction", "evidence": [{"tier_label": "Systematic Review"}]},
            {"hypothesis": "type 2 diabetes", "evidence": [{"tier_label": "Guideline"}]},
        ]})
        c1 = compute_confidence_score_1(vector, web)
        checks = c1["evidence_hypothesis_check"]
        self.assertEqual(checks["matched_hypotheses"], ["acute myocardial infarction"])
        self.assertEqual(checks["unmatched_hypotheses"], ["type 2 diabetes"])
        self.assertGreater(c1["web_evidence_score"], 0.0)

    def test_evidence_results_alias_supports_different_disease(self):
        vector = json.dumps({
            "predictions": [{"diagnosis": "Asthma", "rank": 1}],
            "evidence": [{"cosine_similarity": 0.76}],
        })
        web = json.dumps({"evidence_results": [
            {"query": "Asthma", "articles": [{"type": "Clinical guideline"}]}
        ]})
        c1 = compute_confidence_score_1(vector, web)
        self.assertEqual(c1["evidence_hypothesis_check"]["matched_hypotheses"], ["Asthma"])
        self.assertEqual(c1["similarity_score"], 0.76)

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

    def test_flat_kg_safety_messages_appear_verbatim_in_fusion_report(self):
        message = "Sildenafil is recorded as contraindicated in myocardial infarction"
        safety_messages = [{
            "type": "contraindication",
            "drug": "sildenafil",
            "target": "myocardial infarction",
            "severity": "major",
            "message": message,
        }]
        state = {
            "raw_vector_evidence": {"diagnostic_candidates": [{"condition": "Myocardial Infarction", "rank": 1}]},
            "raw_vector_web_evidence": "",
            "raw_graph_evidence": "Known disease: myocardial infarction",
            "raw_graph_web_evidence": "",
            "kg_safety_messages": safety_messages,
        }
        with patch("graph.run_fusion_agent", return_value="LLM synthesis"):
            result = fusion_node(state)

        self.assertIn(message, result["fusion_report"])
        self.assertIn('"severity": "major"', result["fusion_report"])
        self.assertIn('"type": "contraindication"', result["fusion_report"])


if __name__ == "__main__":
    unittest.main()
