import numpy as np

def compute_entity_metrics(true_set, pred_set):
    """Compute entity-level Precision, Recall, and F0.5 score according to competition rules."""
    is_true_empty = (len(true_set) == 0)
    is_pred_empty = (len(pred_set) == 0)
    
    # Singleton / Zero-match ground truth case
    if is_true_empty:
        if is_pred_empty:
            return 1.0, 1.0, 1.0, 0, 0, 0  # Precision, Recall, F0.5, TP, FP, FN
        else:
            return 0.0, 0.0, 0.0, 0, len(pred_set), 0

    # Non-empty ground truth case
    if is_pred_empty:
        return 1.0, 0.0, 0.0, 0, 0, len(true_set)  # By convention Precision=1.0 on empty pred, Recall=0

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    if tp == 0:
        return 0.0, 0.0, 0.0, 0, fp, fn

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    denom = (0.25 * precision) + recall
    if denom == 0:
        f0_5 = 0.0
    else:
        f0_5 = (1.25 * precision * recall) / denom

    return precision, recall, f0_5, tp, fp, fn

def evaluate_predictions(predictions_dict, gt_loader, eval_s1_ids):
    """Evaluate predictions dictionary over a list of S1 entity IDs."""
    precisions = []
    recalls = []
    f0_5_scores = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred_links = 0
    total_true_links = 0

    # Diagnostics by cardinality
    card_metrics = {
        '0': {'count': 0, 'f0_5': [], 'prec': [], 'rec': [], 'pred_cnt': 0, 'true_cnt': 0},
        '1': {'count': 0, 'f0_5': [], 'prec': [], 'rec': [], 'pred_cnt': 0, 'true_cnt': 0},
        '2': {'count': 0, 'f0_5': [], 'prec': [], 'rec': [], 'pred_cnt': 0, 'true_cnt': 0},
        '3': {'count': 0, 'f0_5': [], 'prec': [], 'rec': [], 'pred_cnt': 0, 'true_cnt': 0},
        '4+': {'count': 0, 'f0_5': [], 'prec': [], 'rec': [], 'pred_cnt': 0, 'true_cnt': 0},
    }

    singleton_correct_empty = 0
    singleton_false_positives = 0
    total_singletons = 0

    for s1_id in eval_s1_ids:
        true_set = gt_loader.get_true_matches(s1_id)
        pred_set = predictions_dict.get(s1_id, set())

        p, r, f05, tp, fp, fn = compute_entity_metrics(true_set, pred_set)

        precisions.append(p)
        recalls.append(r)
        f0_5_scores.append(f05)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_pred_links += len(pred_set)
        total_true_links += len(true_set)

        # Bucket diagnostics
        n_m = len(true_set)
        bucket = "0" if n_m == 0 else ("1" if n_m == 1 else ("2" if n_m == 2 else ("3" if n_m == 3 else "4+")))
        card_metrics[bucket]['count'] += 1
        card_metrics[bucket]['f0_5'].append(f05)
        card_metrics[bucket]['prec'].append(p)
        card_metrics[bucket]['rec'].append(r)
        card_metrics[bucket]['pred_cnt'] += len(pred_set)
        card_metrics[bucket]['true_cnt'] += len(true_set)

        if n_m == 0:
            total_singletons += 1
            if len(pred_set) == 0:
                singleton_correct_empty += 1
            else:
                singleton_false_positives += 1

    macro_f05 = float(np.mean(f0_5_scores)) if f0_5_scores else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    # Aggregate bucket stats
    cardinality_summary = {}
    for bucket, d in card_metrics.items():
        cnt = d['count']
        cardinality_summary[bucket] = {
            'count': cnt,
            'macro_f0_5': float(np.mean(d['f0_5'])) if cnt > 0 else 0.0,
            'macro_precision': float(np.mean(d['prec'])) if cnt > 0 else 0.0,
            'macro_recall': float(np.mean(d['rec'])) if cnt > 0 else 0.0,
            'avg_pred_matches': float(d['pred_cnt'] / cnt) if cnt > 0 else 0.0,
            'avg_true_matches': float(d['true_cnt'] / cnt) if cnt > 0 else 0.0,
        }

    results = {
        'eval_s1_count': len(eval_s1_ids),
        'macro_f0_5': macro_f05,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'micro_precision': float(micro_precision),
        'micro_recall': float(micro_recall),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_pred_links': total_pred_links,
        'total_true_links': total_true_links,
        'singleton_analysis': {
            'total_singletons': total_singletons,
            'correct_empty': singleton_correct_empty,
            'false_positives': singleton_false_positives,
            'correct_empty_rate': (singleton_correct_empty / total_singletons * 100) if total_singletons > 0 else 0.0,
            'false_positive_rate': (singleton_false_positives / total_singletons * 100) if total_singletons > 0 else 0.0,
        },
        'cardinality_breakdown': cardinality_summary
    }

    return results

def run_metric_sanity_tests():
    print("Running Metric Sanity Tests (Examples A-F)...", flush=True)

    # Example A: True [S2-1], Pred [S2-1] -> P=1, R=1, F0.5=1
    p, r, f05, _, _, _ = compute_entity_metrics({'S2-1'}, {'S2-1'})
    assert p == 1.0 and r == 1.0 and f05 == 1.0, f"Example A Failed: P={p}, R={r}, F0.5={f05}"

    # Example B: True [S2-1], Pred [] -> P=1, R=0, F0.5=0
    p, r, f05, _, _, _ = compute_entity_metrics({'S2-1'}, set())
    assert r == 0.0 and f05 == 0.0, f"Example B Failed: P={p}, R={r}, F0.5={f05}"

    # Example C: True [], Pred [] -> F0.5=1
    p, r, f05, _, _, _ = compute_entity_metrics(set(), set())
    assert f05 == 1.0, f"Example C Failed: F0.5={f05}"

    # Example D: True [], Pred [S2-1] -> F0.5=0
    p, r, f05, _, _, _ = compute_entity_metrics(set(), {'S2-1'})
    assert f05 == 0.0, f"Example D Failed: F0.5={f05}"

    # Example E: True [S2-1, S2-2], Pred [S2-1] -> P=1, R=0.5, F0.5 = (1.25*1*0.5)/(0.25*1 + 0.5) = 0.625/0.75 = 0.833333...
    p, r, f05, _, _, _ = compute_entity_metrics({'S2-1', 'S2-2'}, {'S2-1'})
    expected_f05_e = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5)
    assert p == 1.0 and r == 0.5 and abs(f05 - expected_f05_e) < 1e-5, f"Example E Failed: P={p}, R={r}, F0.5={f05}"

    # Example F: True [S2-1], Pred [S2-1, S2-2] -> P=0.5, R=1.0, F0.5 = (1.25*0.5*1.0)/(0.25*0.5 + 1.0) = 0.625/1.125 = 0.555555...
    p, r, f05, _, _, _ = compute_entity_metrics({'S2-1'}, {'S2-1', 'S2-2'})
    expected_f05_f = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0)
    assert p == 0.5 and r == 1.0 and abs(f05 - expected_f05_f) < 1e-5, f"Example F Failed: P={p}, R={r}, F0.5={f05}"

    print("ALL METRIC SANITY TESTS PASSED SUCCESSFULLY! (100% Verified)", flush=True)

if __name__ == "__main__":
    run_metric_sanity_tests()
