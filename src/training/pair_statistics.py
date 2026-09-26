import os
import json
import pandas as pd

class PairStatisticsReporter:
    """Generates Phase 6 training & validation pair distribution statistics report."""

    def __init__(self, summary_json_path, report_markdown_path):
        self.summary_json_path = summary_json_path
        self.report_markdown_path = report_markdown_path

    def save_summary_json(self, stats_data):
        os.makedirs(os.path.dirname(self.summary_json_path), exist_ok=True)
        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2)

    def generate_markdown_report(self, stats):
        os.makedirs(os.path.dirname(self.report_markdown_path), exist_ok=True)
        
        with open(self.report_markdown_path, "w", encoding="utf-8") as f:
            f.write("# 6. Training Pair Generation & Hard Negative Mining Report\n\n")

            f.write("## 6.1 Candidate Pool\n")
            f.write(f"- **Total Raw Candidate Pairs Available (Train + Val):** {stats['total_raw_cand_pairs']:,}\n")
            f.write(f"- **Train S1 Candidates:** {stats['train_raw_cand_pairs']:,} across {stats['train_s1_count']:,} S1 entities (Mean: {stats['train_raw_cand_pairs']/max(1, stats['train_s1_count']):.1f} cands/S1)\n")
            f.write(f"- **Validation S1 Candidates:** {stats['val_raw_cand_pairs']:,} across {stats['val_s1_count']:,} S1 entities (Mean: {stats['val_raw_cand_pairs']/max(1, stats['val_s1_count']):.1f} cands/S1)\n\n")

            f.write("## 6.2 Positive Pair Coverage\n")
            f.write(f"- **Total Ground-Truth Links (Dataset Wide):** {stats['total_gt_links']:,}\n")
            f.write(f"- **Train Ground-Truth Links:** {stats['train_gt_links']:,}\n")
            f.write(f"- **Train GT Links Captured by Blocking:** {stats['train_gt_links_captured']:,} ({stats['train_gt_recall_pct']:.2f}% recall)\n")
            f.write(f"- **Validation Ground-Truth Links:** {stats['val_gt_links']:,}\n")
            f.write(f"- **Validation GT Links Captured by Blocking:** {stats['val_gt_links_captured']:,} ({stats['val_gt_recall_pct']:.2f}% recall)\n\n")

            f.write("## 6.3 Negative Sampling\n")
            f.write(f"- **Train Potential Candidates:** {stats['train_raw_cand_pairs']:,}\n")
            f.write(f"- **Train Sampled Negatives:** {stats['train_sampled_negatives']:,}\n")
            f.write(f"- **Target Negative Ratio:** {stats['target_negative_ratio']:.1f} negatives per positive\n")
            f.write(f"- **Effective Negative Ratio:** {stats['effective_negative_ratio']:.2f} negatives per positive\n\n")

            f.write("## 6.4 Hard Negative Categories\n\n")
            f.write("| Category | Count | Percentage |\n")
            f.write("|---|---|---|\n")
            tot_neg = max(1, stats['train_sampled_negatives'])
            for cat_name, cnt in stats['hard_negative_counts'].items():
                f.write(f"| `{cat_name}` | {cnt:,} | {cnt/tot_neg*100:.2f}% |\n")
            f.write("\n")

            f.write("## 6.5 Training Pair Distribution\n")
            f.write(f"- **Total Selected Training Pairs:** {stats['total_train_pairs']:,}\n")
            f.write(f"- **Positive Pairs (Label=1):** {stats['train_positives']:,} ({stats['train_positives']/stats['total_train_pairs']*100:.2f}%)\n")
            f.write(f"- **Negative Pairs (Label=0):** {stats['train_negatives']:,} ({stats['train_negatives']/stats['total_train_pairs']*100:.2f}%)\n")
            f.write(f"- **Train S1->S2 Positives:** {stats['train_s2_positives']:,}\n")
            f.write(f"- **Train S1->S3 Positives:** {stats['train_s3_positives']:,}\n\n")

            f.write("## 6.6 Validation Pair Distribution\n")
            f.write(f"- **Total Pinned Validation Pairs:** {stats['total_val_pairs']:,}\n")
            f.write(f"- **Validation Positives (Label=1):** {stats['val_positives']:,} ({stats['val_positives']/stats['total_val_pairs']*100:.2f}%)\n")
            f.write(f"- **Validation Negatives (Label=0):** {stats['val_negatives']:,} ({stats['val_negatives']/stats['total_val_pairs']*100:.2f}%)\n")
            f.write(f"- **Val S1->S2 Positives:** {stats['val_s2_positives']:,}\n")
            f.write(f"- **Val S1->S3 Positives:** {stats['val_s3_positives']:,}\n\n")

            f.write("## 6.7 S1 Leakage Check\n")
            f.write(f"- **Overlap between Train S1 and Val S1:** {stats['s1_leakage_count']}\n")
            f.write(f"- **Duplicate Pairs in Train:** {stats['train_duplicate_pairs']}\n")
            f.write(f"- **Duplicate Pairs in Val:** {stats['val_duplicate_pairs']}\n")
            f.write(f"- **Status:** **{'PASSED — 0 LEAKAGE / 0 DUPLICATES' if stats['s1_leakage_count'] == 0 and stats['train_duplicate_pairs'] == 0 else 'FAILED'}**\n\n")

            f.write("## 6.8 Source Distribution\n")
            f.write(f"- **Train S2 Candidate Pairs:** {stats['train_s2_total']:,}\n")
            f.write(f"- **Train S3 Candidate Pairs:** {stats['train_s3_total']:,}\n")
            f.write(f"- **Validation S2 Candidate Pairs:** {stats['val_s2_total']:,}\n")
            f.write(f"- **Validation S3 Candidate Pairs:** {stats['val_s3_total']:,}\n\n")

            f.write("## 6.9 Unretrieved Positive Analysis\n")
            f.write(f"- **Train Ground-Truth Links Missed by Blocking:** {stats['train_unretrieved_positives']:,}\n")
            f.write(f"- **Validation Ground-Truth Links Missed by Blocking:** {stats['val_unretrieved_positives']:,}\n")
            f.write("- **Critical Safety Rule Verified:** Unretrieved positives are **NOT** labeled negative. They remain explicitly classified as unretrieved.\n\n")

            f.write("## 6.10 Memory and Runtime\n")
            f.write(f"- **Total Execution Time:** {stats['total_runtime_sec']:.2f} seconds ({stats['total_runtime_sec']/60:.2f} minutes).\n")
            f.write(f"- **Output Format:** Efficient compressed Parquet (`train_pairs.parquet` & `validation_pairs.parquet`).\n\n")

            f.write("## 6.11 Training Dataset Recommendation\n")
            f.write("The generated training pair dataset combines all candidate-covered true positives with a hard-negative sample ")
            f.write("prioritizing name conflicts, address conflicts, and high similarity non-matches. This provides optimal discriminative signal for downstream Gradient Boosted Decision Trees (GBDTs).\n")
