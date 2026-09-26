import os
import sys
import unittest
import numpy as np

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(base_dir, "src", "models"))
sys.path.insert(0, os.path.join(base_dir, "src", "evaluation"))

from decision_layer import EntityDecisionLayer

class TestModelEvaluation(unittest.TestCase):
    def test_01_decision_layer_empty_gt_correct_empty_pred(self):
        gt_map = {"S1-1": set()}
        layer = EntityDecisionLayer(ground_truth_map=gt_map)
        res = layer.evaluate_predictions(
            s1_ids=["S1-1"],
            cand_ids=["S2-100"],
            cand_sources=["S2"],
            probs=np.array([0.20]),
            threshold=0.50
        )
        self.assertEqual(res["entity_macro_f05"], 1.0)
        self.assertEqual(res["entity_macro_precision"], 1.0)
        self.assertEqual(res["entity_macro_recall"], 1.0)

    def test_02_decision_layer_empty_gt_false_positive_pred(self):
        gt_map = {"S1-1": set()}
        layer = EntityDecisionLayer(ground_truth_map=gt_map)
        res = layer.evaluate_predictions(
            s1_ids=["S1-1"],
            cand_ids=["S2-100"],
            cand_sources=["S2"],
            probs=np.array([0.80]),
            threshold=0.50
        )
        self.assertEqual(res["entity_macro_f05"], 0.0)
        self.assertEqual(res["entity_macro_precision"], 0.0)
        self.assertEqual(res["entity_macro_recall"], 0.0)
        self.assertEqual(res["zero_gt_false_positive_count"], 1)

    def test_03_decision_layer_non_empty_gt_exact_match(self):
        gt_map = {"S1-1": {"S2-100"}}
        layer = EntityDecisionLayer(ground_truth_map=gt_map)
        res = layer.evaluate_predictions(
            s1_ids=["S1-1"],
            cand_ids=["S2-100"],
            cand_sources=["S2"],
            probs=np.array([0.90]),
            threshold=0.50
        )
        self.assertEqual(res["entity_macro_f05"], 1.0)
        self.assertEqual(res["entity_macro_precision"], 1.0)
        self.assertEqual(res["entity_macro_recall"], 1.0)

    def test_04_decision_layer_threshold_filtering(self):
        gt_map = {"S1-1": {"S2-100"}}
        layer = EntityDecisionLayer(ground_truth_map=gt_map)
        res = layer.evaluate_predictions(
            s1_ids=["S1-1"],
            cand_ids=["S2-100"],
            cand_sources=["S2"],
            probs=np.array([0.40]),
            threshold=0.50
        )
        self.assertEqual(res["entity_macro_f05"], 0.0)

    def test_05_decision_layer_strategy_b_margin(self):
        gt_map = {"S1-1": {"S2-100", "S2-101"}}
        layer = EntityDecisionLayer(ground_truth_map=gt_map)
        res = layer.evaluate_predictions(
            s1_ids=["S1-1", "S1-1"],
            cand_ids=["S2-100", "S2-101"],
            cand_sources=["S2", "S2"],
            probs=np.array([0.90, 0.88]),
            threshold=0.50,
            strategy="StrategyB_Margin"
        )
        self.assertTrue(res["entity_macro_f05"] > 0.0)

    def test_06_macro_f05_aggregation(self):
        gt_map = {"S1-1": {"S2-100"}, "S1-2": set()}
        layer = EntityDecisionLayer(ground_truth_map=gt_map)
        res = layer.evaluate_predictions(
            s1_ids=["S1-1", "S1-2"],
            cand_ids=["S2-100", "S2-200"],
            cand_sources=["S2", "S2"],
            probs=np.array([0.90, 0.10]),
            threshold=0.50
        )
        self.assertEqual(res["entity_macro_f05"], 1.0)

if __name__ == "__main__":
    unittest.main()
