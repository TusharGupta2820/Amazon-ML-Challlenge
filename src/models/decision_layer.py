import os
import sys
import numpy as np
from collections import defaultdict

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(base_dir, "src", "evaluation"))

from metrics import compute_entity_metrics

class EntityDecisionLayer:
    def __init__(self, ground_truth_map=None):
        """
        ground_truth_map: dict mapping s1_id -> set of true matched candidate IDs (S2/S3)
        """
        self.gt_map = ground_truth_map or {}

    def evaluate_predictions(self, s1_ids, cand_ids, cand_sources, probs, threshold=0.5, strategy="StrategyA_GlobalThresh", extra_features=None):
        """
        s1_ids: list/array of source1_entity_id strings
        cand_ids: list/array of candidate_entity_id strings
        cand_sources: list/array of candidate_source strings ("S2" or "S3")
        probs: array of predicted probabilities
        threshold: decision threshold (float or tuple for StrategyE)
        strategy: decision rule strategy
        extra_features: optional dict of additional feature vectors (e.g. contradiction scores)
        """
        # Group candidates by S1 ID
        s1_groups = defaultdict(list)
        for i in range(len(s1_ids)):
            s1_id = s1_ids[i]
            c_id = cand_ids[i]
            c_src = cand_sources[i]
            p = float(probs[i])
            contra_score = float(extra_features["name_address_contradiction_score"][i]) if (extra_features and "name_address_contradiction_score" in extra_features) else 0.0
            s1_groups[s1_id].append((p, c_id, c_src, contra_score))

        all_s1_ids = list(self.gt_map.keys())
        total_s1 = len(all_s1_ids)

        f05_sum = 0.0
        prec_sum = 0.0
        rec_sum = 0.0

        zero_gt_count = 0
        zero_gt_fp_count = 0
        false_positive_s1_count = 0
        total_pred_matches = 0

        for s1_id in all_s1_ids:
            gt_set = self.gt_map.get(s1_id, set())
            cand_list = s1_groups.get(s1_id, [])

            # Sort candidates by probability descending
            cand_list.sort(key=lambda x: x[0], reverse=True)

            pred_set = set()

            if strategy == "StrategyA_GlobalThresh":
                thresh = float(threshold)
                for p, c_id, c_src, _ in cand_list:
                    if p >= thresh:
                        pred_set.add(c_id)

            elif strategy == "StrategyB_Margin":
                thresh = float(threshold)
                min_margin = 0.05
                if cand_list:
                    top_p = cand_list[0][0]
                    sec_p = cand_list[1][0] if len(cand_list) > 1 else 0.0
                    for p, c_id, c_src, _ in cand_list:
                        if p >= thresh and (top_p - sec_p >= min_margin or p == top_p):
                            pred_set.add(c_id)

            elif strategy == "StrategyC_ContradictionSuppression":
                base_thresh = float(threshold)
                for p, c_id, c_src, contra_score in cand_list:
                    eff_thresh = base_thresh + 0.15 if contra_score >= 1.5 else base_thresh
                    if p >= eff_thresh:
                        pred_set.add(c_id)

            elif strategy == "StrategyD_MultiCandidateConsistency":
                thresh = float(threshold)
                if cand_list:
                    top_p = cand_list[0][0]
                    for p, c_id, c_src, _ in cand_list:
                        if p >= thresh and (top_p - p <= 0.10):
                            pred_set.add(c_id)

            elif strategy == "StrategyE_SeparateSourceCalib":
                if isinstance(threshold, (tuple, list)):
                    th_s2, th_s3 = threshold
                else:
                    th_s2, th_s3 = threshold, threshold
                for p, c_id, c_src, _ in cand_list:
                    th = th_s2 if c_src == "S2" else th_s3
                    if p >= th:
                        pred_set.add(c_id)

            else:
                thresh = float(threshold)
                for p, c_id, c_src, _ in cand_list:
                    if p >= thresh:
                        pred_set.add(c_id)

            # Compute exact entity metrics according to competition rules
            e_prec, e_rec, e_f05, _, _, _ = compute_entity_metrics(gt_set, pred_set)
            
            total_pred_matches += len(pred_set)

            if not gt_set and pred_set:
                zero_gt_fp_count += 1
                false_positive_s1_count += 1
            elif gt_set and (pred_set - gt_set):
                false_positive_s1_count += 1

            if not gt_set:
                zero_gt_count += 1

            f05_sum += e_f05
            prec_sum += e_prec
            rec_sum += e_rec

        macro_f05 = float(f05_sum / total_s1)
        macro_prec = float(prec_sum / total_s1)
        macro_rec = float(rec_sum / total_s1)
        avg_preds_per_s1 = float(total_pred_matches / total_s1)

        return {
            "entity_macro_f05": macro_f05,
            "entity_macro_precision": macro_prec,
            "entity_macro_recall": macro_rec,
            "total_validation_s1": total_s1,
            "zero_gt_s1_count": zero_gt_count,
            "zero_gt_false_positive_count": zero_gt_fp_count,
            "false_positive_s1_count": false_positive_s1_count,
            "mean_predictions_per_s1": avg_preds_per_s1,
            "threshold": threshold,
            "strategy": strategy
        }
