import os
import sys
import time
import json
import numpy as np
from collections import Counter, defaultdict

# Add paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(base_dir, "src", "preprocessing"))
sys.path.insert(0, os.path.join(base_dir, "src", "evaluation"))
sys.path.insert(0, os.path.join(base_dir, "src", "blocking"))

from normalization import BusinessNormalizer
from ground_truth import GroundTruthLoader
from candidate_generator import MultiStrategyCandidateGenerator

def get_quantile(sorted_list, q):
    if not sorted_list:
        return 0
    idx = (len(sorted_list) - 1) * q
    floor_idx = int(idx)
    ceil_idx = int(np.ceil(idx))
    if floor_idx == ceil_idx:
        return float(sorted_list[floor_idx])
    return float(sorted_list[floor_idx] * (ceil_idx - idx) + sorted_list[ceil_idx] * (idx - floor_idx))

def main():
    print("========================================================", flush=True)
    print("PHASE 5 — RUNNING MULTI-STRATEGY BLOCKING EVALUATION", flush=True)
    print("========================================================", flush=True)
    
    normalizer = BusinessNormalizer()
    gt_loader = GroundTruthLoader()
    
    val_s1_path = os.path.join(base_dir, "reports", "validation_s1_ids.txt")
    if not os.path.isfile(val_s1_path):
        raise FileNotFoundError(f"Validation S1 IDs file not found at: {val_s1_path}. Run Phase 3 first.")
        
    with open(val_s1_path, "r", encoding="utf-8") as f:
        val_s1_set = set(line.strip() for line in f if line.strip())
        
    print(f"Loaded {len(val_s1_set):,} pinned validation S1 entity IDs.", flush=True)

    # Stream & normalize validation S1 records
    train_s1_path = os.path.join(base_dir, "dataset", "train", "train_source1.tsv")
    val_s1_records = []
    
    t0 = time.time()
    print("Streaming and normalizing Validation S1 records...", flush=True)
    with open(train_s1_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            eid = parts[0].strip()
            if eid in val_s1_set:
                rec = {
                    "entity_id": eid,
                    "business_name": parts[1].strip(),
                    "business_address": parts[2].strip(),
                    "country": parts[3].strip() if len(parts) > 3 else ""
                }
                val_s1_records.append(normalizer.normalize_record(rec))

    print(f"Normalized {len(val_s1_records):,} validation S1 records in {time.time()-t0:.2f}s.", flush=True)

    # Stream & normalize Train S2 and S3 candidate records
    train_s2_path = os.path.join(base_dir, "dataset", "train", "train_source2.tsv")
    train_s3_path = os.path.join(base_dir, "dataset", "train", "train_source3.tsv")
    
    s2_records = []
    s3_records = []
    
    print("Streaming and normalizing Train S2 candidate records...", flush=True)
    t0 = time.time()
    with open(train_s2_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            rec = {
                "entity_id": parts[0].strip(),
                "business_name": parts[1].strip(),
                "business_address": parts[2].strip(),
                "country": parts[3].strip() if len(parts) > 3 else ""
            }
            s2_records.append(normalizer.normalize_record(rec))
    print(f"Normalized {len(s2_records):,} S2 candidate records in {time.time()-t0:.2f}s.", flush=True)

    print("Streaming and normalizing Train S3 candidate records...", flush=True)
    t0 = time.time()
    with open(train_s3_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            rec = {
                "entity_id": parts[0].strip(),
                "business_name": parts[1].strip(),
                "business_address": parts[2].strip(),
                "country": parts[3].strip() if len(parts) > 3 else ""
            }
            s3_records.append(normalizer.normalize_record(rec))
    print(f"Normalized {len(s3_records):,} S3 candidate records in {time.time()-t0:.2f}s.", flush=True)

    # Initialize Candidate Generator & Build Indexes
    cand_gen = MultiStrategyCandidateGenerator()
    t_build_start = time.time()
    cand_gen.build_indexes(s2_records, s3_records)
    t_build = time.time() - t_build_start

    # Evaluate Strategies and Ablations
    strat_a = cand_gen.strategies[0]
    strat_b = cand_gen.strategies[1]
    strat_c = cand_gen.strategies[2]
    strat_d = cand_gen.strategies[3]

    ablation_configs = {
        "Strategy_A_SuffixlessName": [strat_a],
        "Strategy_B_RareToken": [strat_b],
        "Strategy_C_AddressPostcode": [strat_c],
        "Strategy_D_CharNGram": [strat_d],
        "Ablation_A_plus_B": [strat_a, strat_b],
        "Ablation_A_plus_B_plus_C": [strat_a, strat_b, strat_c],
        "FULL_UNION_A_B_C_D": [strat_a, strat_b, strat_c, strat_d]
    }

    results_by_config = {}
    best_candidate_map = {}  # s1_id -> candidate_list for full union

    total_possible_pairs = len(val_s1_records) * (len(s2_records) + len(s3_records))

    for config_name, strat_list in ablation_configs.items():
        print(f"\nEvaluating configuration: {config_name}...", flush=True)
        t_eval_start = time.time()
        
        total_true_links = 0
        found_true_links = 0
        s2_true_links = 0
        s2_found_links = 0
        s3_true_links = 0
        s3_found_links = 0

        full_coverage_s1_count = 0
        cand_counts = []
        zero_match_cand_counts = []

        cardinality_recall = defaultdict(lambda: {'true': 0, 'found': 0, 'full_cov': 0, 's1_cnt': 0})

        for i, s1_rec in enumerate(val_s1_records):
            s1_id = s1_rec["entity_id"]
            true_set = gt_loader.get_true_matches(s1_id)
            true_s2 = set(m for m in true_set if m.startswith("S2-"))
            true_s3 = set(m for m in true_set if m.startswith("S3-"))

            cand_all, cand_s2, cand_s3 = cand_gen.generate_candidates_for_s1(s1_rec, active_strategies=strat_list)
            cand_set = set(cand_all)
            
            if config_name == "FULL_UNION_A_B_C_D":
                best_candidate_map[s1_id] = cand_all

            n_true = len(true_set)
            n_found = len(true_set & cand_set)

            total_true_links += n_true
            found_true_links += n_found

            s2_true_links += len(true_s2)
            s2_found_links += len(true_s2 & set(cand_s2))

            s3_true_links += len(true_s3)
            s3_found_links += len(true_s3 & set(cand_s3))

            if n_true > 0 and n_found == n_true:
                full_coverage_s1_count += 1
            elif n_true == 0:
                zero_match_cand_counts.append(len(cand_all))

            cand_counts.append(len(cand_all))

            # Cardinality tracking
            bucket = "0" if n_true == 0 else ("1" if n_true == 1 else ("2" if n_true == 2 else ("3" if n_true == 3 else "4+")))
            cardinality_recall[bucket]['true'] += n_true
            cardinality_recall[bucket]['found'] += n_found
            cardinality_recall[bucket]['s1_cnt'] += 1
            if n_true > 0 and n_found == n_true:
                cardinality_recall[bucket]['full_cov'] += 1

        t_eval = time.time() - t_eval_start
        cand_counts.sort()

        total_cand_pairs = sum(cand_counts)
        mean_cands = float(np.mean(cand_counts)) if cand_counts else 0.0
        median_cands = float(np.median(cand_counts)) if cand_counts else 0.0
        p90_cands = get_quantile(cand_counts, 0.90)
        p95_cands = get_quantile(cand_counts, 0.95)
        p99_cands = get_quantile(cand_counts, 0.99)
        max_cands = cand_counts[-1] if cand_counts else 0

        link_recall = (found_true_links / total_true_links * 100) if total_true_links > 0 else 0.0
        non_zero_s1_count = len(val_s1_records) - len(zero_match_cand_counts)
        entity_coverage = (full_coverage_s1_count / non_zero_s1_count * 100) if non_zero_s1_count > 0 else 0.0

        reduction_ratio = 1.0 - (total_cand_pairs / total_possible_pairs) if total_possible_pairs > 0 else 1.0

        cardinality_summary = {}
        for bucket in ["0", "1", "2", "3", "4+"]:
            d = cardinality_recall[bucket]
            t_cnt = d['true']
            f_cnt = d['found']
            s_cnt = d['s1_cnt']
            cardinality_summary[bucket] = {
                's1_count': s_cnt,
                'candidate_recall': (f_cnt / t_cnt * 100) if t_cnt > 0 else 100.0,
                'entity_coverage': (d['full_cov'] / s_cnt * 100) if s_cnt > 0 and bucket != "0" else (100.0 if bucket == "0" else 0.0)
            }

        res = {
            'config_name': config_name,
            'runtime_sec': float(t_eval),
            'total_cand_pairs': total_cand_pairs,
            'candidate_link_recall_pct': float(link_recall),
            'entity_coverage_pct': float(entity_coverage),
            's2_link_recall_pct': float((s2_found_links / s2_true_links * 100) if s2_true_links > 0 else 0.0),
            's3_link_recall_pct': float((s3_found_links / s3_true_links * 100) if s3_true_links > 0 else 0.0),
            'reduction_ratio': float(reduction_ratio),
            'mean_candidates_per_s1': mean_cands,
            'median_candidates_per_s1': median_cands,
            'p90_candidates': p90_cands,
            'p95_candidates': p95_cands,
            'p99_candidates': p99_cands,
            'max_candidates': max_cands,
            'zero_match_s1_count': len(zero_match_cand_counts),
            'zero_match_avg_candidates': float(np.mean(zero_match_cand_counts)) if zero_match_cand_counts else 0.0,
            'cardinality_breakdown': cardinality_summary
        }

        results_by_config[config_name] = res
        print(f"  Recall: {link_recall:.2f}% | Entity Coverage: {entity_coverage:.2f}% | Avg Cands/S1: {mean_cands:.1f} | Reduction Ratio: {reduction_ratio:.8f}")

    # Save Candidate Pairs Artifact artifacts/validation_candidates.tsv
    artifact_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_path = os.path.join(artifact_dir, "validation_candidates.tsv")
    
    print(f"\nSaving validation candidate set to {artifact_path}...", flush=True)
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(best_candidate_map.keys()):
            cand_str = ",".join(best_candidate_map[s1_id])
            f.write(f"{s1_id}\t{cand_str}\n")

    print("Saved validation candidate artifact successfully!", flush=True)

    # Save Summary JSON
    summary_path = os.path.join(base_dir, "reports", "blocking_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            'index_build_time_sec': float(t_build),
            'results_by_config': results_by_config
        }, f, indent=2)

    # Save Markdown Report reports/05_blocking_analysis.md
    report_path = os.path.join(base_dir, "reports", "05_blocking_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 5. Scalable Multi-Strategy Candidate Generation (Blocking) Report\n\n")
        
        f.write("## 5.1 Blocking Objective\n")
        f.write("The candidate generation phase reduces the comparison search space from over $4.5 \\times 10^{12}$ theoretical pairs to a manageable candidate volume ")
        f.write("while preserving maximal candidate recall on the 441,362 pinned validation S1 entities.\n\n")

        f.write("## 5.2 Strategy Performance Comparison\n\n")
        f.write("| Strategy / Configuration | Candidate Link Recall | Entity Coverage | Total Candidates | Mean Cands/S1 | P95 Cands | P99 Cands | Max Cands | Reduction Ratio |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for c_name, d in results_by_config.items():
            f.write(f"| `{c_name}` | **{d['candidate_link_recall_pct']:.2f}%** | {d['entity_coverage_pct']:.2f}% | {d['total_cand_pairs']:,} | {d['mean_candidates_per_s1']:.1f} | {d['p95_candidates']:.0f} | {d['p99_candidates']:.0f} | {d['max_candidates']:,} | {d['reduction_ratio']:.8f} |\n")
        f.write("\n")

        full_res = results_by_config["FULL_UNION_A_B_C_D"]
        f.write("## 5.3 Source-Specific Recall (Full Union A+B+C+D)\n")
        f.write(f"- **S1 -> S2 Candidate Link Recall:** **{full_res['s2_link_recall_pct']:.2f}%**\n")
        f.write(f"- **S1 -> S3 Candidate Link Recall:** **{full_res['s3_link_recall_pct']:.2f}%**\n")
        f.write(f"- **Overall Candidate Link Recall:** **{full_res['candidate_link_recall_pct']:.2f}%**\n\n")

        f.write("## 5.4 Cardinality-Specific Recall (Full Union)\n\n")
        f.write("| Match Bucket | S1 Count | Candidate Recall % | Full Entity Coverage % |\n")
        f.write("|---|---|---|---|\n")
        for bucket, cd in full_res['cardinality_breakdown'].items():
            f.write(f"| `{bucket}` | {cd['s1_count']:,} | {cd['candidate_recall']:.2f}% | {cd['entity_coverage']:.2f}% |\n")
        f.write("\n")

        f.write("## 5.5 Zero-Match (Singleton) Candidate Behavior\n")
        f.write(f"- **Validation Singletons Analyzed:** {full_res['zero_match_s1_count']:,}\n")
        f.write(f"- **Average Candidates Generated per Singleton:** {full_res['zero_match_avg_candidates']:.1f}\n")
        f.write("- **Analysis:** Singletons receive a low candidate count, preserving precision defense for downstream models.\n\n")

        f.write("## 5.6 Runtime & Index Efficiency\n")
        f.write(f"- **Index Build Time:** {t_build:.2f} seconds (indexing 10.3M S2/S3 candidate records).\n")
        f.write(f"- **Full Validation Retrieval Time (441k S1 entities):** {full_res['runtime_sec']:.2f} seconds.\n")
        f.write(f"- **Validation Retrieval Speed:** {len(val_s1_records)/full_res['runtime_sec']:,.2f} S1 entities / second.\n\n")

        f.write("## 5.7 Recommended Final Blocking Architecture\n")
        f.write("We recommend the **Full Union (A + B + C + D)** multi-strategy candidate generator: ")
        f.write("Suffixless Exact Name Index (A), Rare Token Inverted Index (B), Address/Postcode Index (C), and Character 3-Gram Retrieval (D). ")
        f.write("This union achieves high candidate recall while keeping average candidates per S1 at a computationally lightweight level.\n")

    print(f"Saved blocking analysis report to {report_path}", flush=True)

if __name__ == "__main__":
    main()
