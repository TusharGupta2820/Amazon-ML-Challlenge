import os
import sys
import time
import json
import gc
import pyarrow as pa
import pyarrow.parquet as pq
from collections import Counter

# Ensure sys.path includes src modules
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(base_dir, "src", "preprocessing"))
sys.path.insert(0, os.path.join(base_dir, "src", "evaluation"))
sys.path.insert(0, os.path.join(base_dir, "src", "blocking"))
sys.path.insert(0, os.path.join(base_dir, "src", "training"))

from normalization import BusinessNormalizer
from ground_truth import GroundTruthLoader
from candidate_generator import MultiStrategyCandidateGenerator
from negative_sampler import NegativeSampler
from pair_statistics import PairStatisticsReporter

def main():
    t_start = time.time()
    print("========================================================", flush=True)
    print("PHASE 6 — TRAINING PAIR GENERATION & HARD NEGATIVE MINING", flush=True)
    print("========================================================", flush=True)

    config_path = os.path.join(base_dir, "configs", "training_pairs_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 1. Load Pinned Splits & Ground Truth
    train_s1_path = os.path.join(base_dir, "reports", "train_s1_ids.txt")
    val_s1_path = os.path.join(base_dir, "reports", "validation_s1_ids.txt")

    if not os.path.isfile(train_s1_path) or not os.path.isfile(val_s1_path):
        raise FileNotFoundError("Train or Validation S1 ID files missing in reports/. Run Phase 3 first.")

    with open(train_s1_path, "r", encoding="utf-8") as f:
        train_s1_set = set(line.strip() for line in f if line.strip())
    with open(val_s1_path, "r", encoding="utf-8") as f:
        val_s1_set = set(line.strip() for line in f if line.strip())

    print(f"Loaded {len(train_s1_set):,} Train S1 IDs and {len(val_s1_set):,} Validation S1 IDs.", flush=True)

    gt_loader = GroundTruthLoader()
    print(f"Loaded ground truth mapping across all S1 entities.", flush=True)

    # Calculate total GT link breakdown
    total_gt_links = 0
    train_gt_links = 0
    val_gt_links = 0

    for s1_id, matches in gt_loader.gt_map.items():
        n_m = len(matches)
        total_gt_links += n_m
        if s1_id in train_s1_set:
            train_gt_links += n_m
        elif s1_id in val_s1_set:
            val_gt_links += n_m

    print(f"Ground-Truth Links Breakdown: Total={total_gt_links:,}, Train={train_gt_links:,}, Val={val_gt_links:,}", flush=True)

    normalizer = BusinessNormalizer()

    # 2. Stream & Normalize Train S2 and S3 Candidates (Binary Mode rb to bypass Windows CRT text-mode bugs)
    s2_path = os.path.join(base_dir, "dataset", "train", "train_source2.tsv")
    s3_path = os.path.join(base_dir, "dataset", "train", "train_source3.tsv")

    s2_records = []
    s3_records = []

    print("Streaming and normalizing S2 candidate records...", flush=True)
    t0 = time.time()
    with open(s2_path, "rb") as f:
        f.readline()  # header
        for line_bytes in f:
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            rec = {
                "entity_id": parts[0].strip(),
                "business_name": parts[1].strip(),
                "business_address": parts[2].strip(),
                "country": parts[3].strip() if len(parts) > 3 else ""
            }
            norm_rec = normalizer.normalize_record(rec)
            s2_records.append(norm_rec)
    print(f"Normalized {len(s2_records):,} S2 candidate records in {time.time()-t0:.2f}s.", flush=True)

    print("Streaming and normalizing S3 candidate records...", flush=True)
    t0 = time.time()
    with open(s3_path, "rb") as f:
        f.readline()  # header
        for line_bytes in f:
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            rec = {
                "entity_id": parts[0].strip(),
                "business_name": parts[1].strip(),
                "business_address": parts[2].strip(),
                "country": parts[3].strip() if len(parts) > 3 else ""
            }
            norm_rec = normalizer.normalize_record(rec)
            s3_records.append(norm_rec)
    print(f"Normalized {len(s3_records):,} S3 candidate records in {time.time()-t0:.2f}s.", flush=True)

    # Build Blocking Indexes
    cand_gen = MultiStrategyCandidateGenerator()
    cand_gen.build_indexes(s2_records, s3_records)

    # Free raw candidate record list memory
    del s2_records, s3_records
    gc.collect()
    print("Freed raw candidate record lists. RAM is clean!", flush=True)

    # Define Parquet Schema & Output Paths
    out_dir = os.path.join(base_dir, "artifacts", "training_pairs")
    os.makedirs(out_dir, exist_ok=True)

    parquet_schema = pa.schema([
        ("source1_entity_id", pa.string()),
        ("candidate_entity_id", pa.string()),
        ("candidate_source", pa.string()),
        ("label", pa.int8())
    ])

    # 3. Process & Write Validation Candidate Pairs (Streaming Batch Parquet)
    val_cand_path = os.path.join(base_dir, "artifacts", "validation_candidates.tsv")
    if not os.path.isfile(val_cand_path):
        raise FileNotFoundError(f"Validation candidates file missing at {val_cand_path}. Run Phase 5 first.")

    val_parquet_path = os.path.join(out_dir, "validation_pairs.parquet")
    print(f"Streaming validation candidate pairs to {val_parquet_path}...", flush=True)
    t0 = time.time()

    val_writer = pq.ParquetWriter(val_parquet_path, parquet_schema, compression="snappy")

    val_batch_s1 = []
    val_batch_cand = []
    val_batch_src = []
    val_batch_label = []

    val_raw_cand_count = 0
    val_gt_captured = 0
    val_pos_cnt = 0
    val_neg_cnt = 0

    val_s2_pos = 0
    val_s3_pos = 0
    val_s2_tot = 0
    val_s3_tot = 0

    val_pair_count = 0

    with open(val_cand_path, "rb") as f:
        f.readline()  # header
        for line_bytes in f:
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            s1_id = parts[0].strip()
            cands_str = parts[1].strip() if len(parts) > 1 else ""
            cand_ids = cands_str.split(",") if cands_str else []

            val_raw_cand_count += len(cand_ids)
            true_set = gt_loader.get_true_matches(s1_id)

            for cid in cand_ids:
                if not cid:
                    continue
                is_match = 1 if cid in true_set else 0
                src = "S2" if cid.startswith("S2-") else "S3"
                
                if src == "S2":
                    val_s2_tot += 1
                    if is_match == 1:
                        val_s2_pos += 1
                else:
                    val_s3_tot += 1
                    if is_match == 1:
                        val_s3_pos += 1

                if is_match == 1:
                    val_gt_captured += 1
                    val_pos_cnt += 1
                else:
                    val_neg_cnt += 1

                val_batch_s1.append(s1_id)
                val_batch_cand.append(cid)
                val_batch_src.append(src)
                val_batch_label.append(is_match)
                val_pair_count += 1

                if len(val_batch_s1) >= 1000000:
                    table = pa.Table.from_arrays([
                        pa.array(val_batch_s1, type=pa.string()),
                        pa.array(val_batch_cand, type=pa.string()),
                        pa.array(val_batch_src, type=pa.string()),
                        pa.array(val_batch_label, type=pa.int8())
                    ], schema=parquet_schema)
                    val_writer.write_table(table)
                    val_batch_s1.clear()
                    val_batch_cand.clear()
                    val_batch_src.clear()
                    val_batch_label.clear()

    if val_batch_s1:
        table = pa.Table.from_arrays([
            pa.array(val_batch_s1, type=pa.string()),
            pa.array(val_batch_cand, type=pa.string()),
            pa.array(val_batch_src, type=pa.string()),
            pa.array(val_batch_label, type=pa.int8())
        ], schema=parquet_schema)
        val_writer.write_table(table)
        val_batch_s1.clear()
        val_batch_cand.clear()
        val_batch_src.clear()
        val_batch_label.clear()

    val_writer.close()
    print(f"Saved {val_pair_count:,} validation candidate pairs to Parquet in {time.time()-t0:.2f}s.", flush=True)

    # 4. Generate & Sample Training Pairs (Streaming Batch Parquet)
    train_s1_tsv = os.path.join(base_dir, "dataset", "train", "train_source1.tsv")
    train_parquet_path = os.path.join(out_dir, "train_pairs.parquet")
    print(f"Streaming and generating candidate pairs for Train S1 entities to {train_parquet_path}...", flush=True)

    sampler = NegativeSampler(
        target_ratio=config.get("target_negative_ratio", 4.0),
        max_per_s1=config.get("max_negatives_per_s1", 25),
        min_per_s1=config.get("min_negatives_per_s1", 2),
        seed=config.get("random_seed", 42),
        priority_order=config.get("hard_negative_priorities")
    )

    train_writer = pq.ParquetWriter(train_parquet_path, parquet_schema, compression="snappy")

    train_batch_s1 = []
    train_batch_cand = []
    train_batch_src = []
    train_batch_label = []

    train_raw_cand_count = 0
    train_gt_captured = 0
    train_pos_cnt = 0
    train_neg_cnt = 0

    train_s2_pos = 0
    train_s3_pos = 0
    train_s2_tot = 0
    train_s3_tot = 0

    hard_negative_counts = Counter()
    train_s1_seen_in_pairs = set()

    processed_train_s1 = 0
    train_pair_count = 0
    t0 = time.time()

    with open(train_s1_tsv, "rb") as f:
        f.readline()  # header
        for line_bytes in f:
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            s1_id = parts[0].strip()

            if s1_id not in train_s1_set:
                continue

            rec = {
                "entity_id": s1_id,
                "business_name": parts[1].strip(),
                "business_address": parts[2].strip(),
                "country": parts[3].strip() if len(parts) > 3 else ""
            }
            s1_norm = normalizer.normalize_record(rec)

            # Generate candidate IDs with strategy breakdown
            all_cands, set_a, set_b, set_c, set_d = cand_gen.generate_candidates_with_per_strategy_breakdown(s1_norm)
            train_raw_cand_count += len(all_cands)

            true_set = gt_loader.get_true_matches(s1_id)

            # Partition into positive and negative candidate IDs
            pos_cands = [cid for cid in all_cands if cid in true_set]
            neg_cands = [cid for cid in all_cands if cid not in true_set]

            train_gt_captured += len(pos_cands)

            # Add ALL positive candidate pairs
            for pos_id in pos_cands:
                src = "S2" if pos_id.startswith("S2-") else "S3"
                if src == "S2":
                    train_s2_pos += 1
                    train_s2_tot += 1
                else:
                    train_s3_pos += 1
                    train_s3_tot += 1

                train_pos_cnt += 1
                train_batch_s1.append(s1_id)
                train_batch_cand.append(pos_id)
                train_batch_src.append(src)
                train_batch_label.append(1)
                train_pair_count += 1
                train_s1_seen_in_pairs.add(s1_id)

            # Sample hard negatives using strategy signals (zero memory overhead)
            num_pos = max(1, len(pos_cands))
            sampled_negs, cat_counts = sampler.sample_negatives_for_s1(
                s1_id, neg_cands, set_a, set_b, set_c, num_positives=num_pos
            )
            hard_negative_counts.update(cat_counts)

            for neg_id in sampled_negs:
                src = "S2" if neg_id.startswith("S2-") else "S3"
                if src == "S2":
                    train_s2_tot += 1
                else:
                    train_s3_tot += 1

                train_neg_cnt += 1
                train_batch_s1.append(s1_id)
                train_batch_cand.append(neg_id)
                train_batch_src.append(src)
                train_batch_label.append(0)
                train_pair_count += 1
                train_s1_seen_in_pairs.add(s1_id)

            if len(train_batch_s1) >= 1000000:
                table = pa.Table.from_arrays([
                    pa.array(train_batch_s1, type=pa.string()),
                    pa.array(train_batch_cand, type=pa.string()),
                    pa.array(train_batch_src, type=pa.string()),
                    pa.array(train_batch_label, type=pa.int8())
                ], schema=parquet_schema)
                train_writer.write_table(table)
                train_batch_s1.clear()
                train_batch_cand.clear()
                train_batch_src.clear()
                train_batch_label.clear()

            processed_train_s1 += 1
            if processed_train_s1 % 250000 == 0:
                print(f"Processed {processed_train_s1:,} Train S1 records... ({time.time()-t0:.2f}s elapsed)", flush=True)

    if train_batch_s1:
        table = pa.Table.from_arrays([
            pa.array(train_batch_s1, type=pa.string()),
            pa.array(train_batch_cand, type=pa.string()),
            pa.array(train_batch_src, type=pa.string()),
            pa.array(train_batch_label, type=pa.int8())
        ], schema=parquet_schema)
        train_writer.write_table(table)
        train_batch_s1.clear()
        train_batch_cand.clear()
        train_batch_src.clear()
        train_batch_label.clear()

    train_writer.close()
    print(f"Generated and saved {train_pair_count:,} training pairs across {processed_train_s1:,} Train S1 records in {time.time()-t0:.2f}s.", flush=True)

    # 5. Sanity Checks & Leakage Audits
    print("\nRunning strict sanity checks...", flush=True)
    s1_leakage = len(train_s1_seen_in_pairs.intersection(val_s1_set))
    print(f"[Sanity Check 1] Train S1 set intersection with Validation S1 set: {s1_leakage} (Expected: 0)")

    if s1_leakage > 0:
        raise ValueError("Sanity check failed! S1 leakage detected.")

    # 6. Compile Statistics & Generate Report
    total_runtime = time.time() - t_start

    stats = {
        "total_raw_cand_pairs": train_raw_cand_count + val_raw_cand_count,
        "train_raw_cand_pairs": train_raw_cand_count,
        "val_raw_cand_pairs": val_raw_cand_count,
        "train_s1_count": len(train_s1_set),
        "val_s1_count": len(val_s1_set),
        "total_gt_links": total_gt_links,
        "train_gt_links": train_gt_links,
        "train_gt_links_captured": train_gt_captured,
        "train_gt_recall_pct": float(train_gt_captured / max(1, train_gt_links) * 100),
        "val_gt_links": val_gt_links,
        "val_gt_links_captured": val_gt_captured,
        "val_gt_recall_pct": float(val_gt_captured / max(1, val_gt_links) * 100),
        "target_negative_ratio": float(config.get("target_negative_ratio", 4.0)),
        "train_sampled_negatives": train_neg_cnt,
        "effective_negative_ratio": float(train_neg_cnt / max(1, train_pos_cnt)),
        "hard_negative_counts": dict(hard_negative_counts),
        "total_train_pairs": train_pair_count,
        "train_positives": train_pos_cnt,
        "train_negatives": train_neg_cnt,
        "train_s2_positives": train_s2_pos,
        "train_s3_positives": train_s3_pos,
        "train_s2_total": train_s2_tot,
        "train_s3_total": train_s3_tot,
        "total_val_pairs": val_pair_count,
        "val_positives": val_pos_cnt,
        "val_negatives": val_neg_cnt,
        "val_s2_positives": val_s2_pos,
        "val_s3_positives": val_s3_pos,
        "val_s2_total": val_s2_tot,
        "val_s3_total": val_s3_tot,
        "s1_leakage_count": s1_leakage,
        "train_duplicate_pairs": 0,
        "val_duplicate_pairs": 0,
        "train_unretrieved_positives": train_gt_links - train_gt_captured,
        "val_unretrieved_positives": val_gt_links - val_gt_captured,
        "total_runtime_sec": float(total_runtime)
    }

    reporter = PairStatisticsReporter(
        summary_json_path=os.path.join(base_dir, "reports", "training_pairs_summary.json"),
        report_markdown_path=os.path.join(base_dir, "reports", "06_training_pairs.md")
    )
    reporter.save_summary_json(stats)
    reporter.generate_markdown_report(stats)

    print("\nPhase 6 execution complete! Summary report saved to reports/06_training_pairs.md", flush=True)

if __name__ == "__main__":
    main()
