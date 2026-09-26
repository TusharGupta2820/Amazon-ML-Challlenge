import os
import sys
import json
import random
from collections import Counter, defaultdict
from ground_truth import GroundTruthLoader

def get_cardinality_bucket(n):
    if n == 0:
        return "0"
    elif n == 1:
        return "1"
    elif n == 2:
        return "2"
    elif n == 3:
        return "3"
    else:
        return "4+"

def create_grouped_validation_split(gt_path="dataset/train/train_ground_truth.tsv", config_path="configs/validation_config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    seed = config.get("seed", 42)
    val_frac = config.get("validation_fraction", 0.20)
    
    random.seed(seed)
    gt_loader = GroundTruthLoader(gt_path=gt_path)
    
    # Bucket S1 IDs by match cardinality for stratified split
    buckets = defaultdict(list)
    for s1_id in gt_loader.get_all_s1_ids():
        n_m = gt_loader.get_match_count(s1_id)
        bucket = get_cardinality_bucket(n_m)
        buckets[bucket].append(s1_id)

    train_s1_set = set()
    val_s1_set = set()

    for bucket_name, id_list in sorted(buckets.items()):
        # Shuffle reproducibly
        random.shuffle(id_list)
        val_count = int(len(id_list) * val_frac)
        val_ids = id_list[:val_count]
        train_ids = id_list[val_count:]
        
        val_s1_set.update(val_ids)
        train_s1_set.update(train_ids)

    # Verify zero leakage
    overlap = train_s1_set & val_s1_set
    assert len(overlap) == 0, f"Data leakage detected! {len(overlap)} S1 IDs overlap between train and val."

    os.makedirs("reports", exist_ok=True)
    val_file_path = os.path.join("reports", "validation_s1_ids.txt")
    train_file_path = os.path.join("reports", "train_s1_ids.txt")

    with open(val_file_path, "w", encoding="utf-8") as f:
        for s1 in sorted(val_s1_set):
            f.write(f"{s1}\n")

    with open(train_file_path, "w", encoding="utf-8") as f:
        for s1 in sorted(train_s1_set):
            f.write(f"{s1}\n")

    print(f"Validation split created successfully!")
    print(f"  Train S1 Entities: {len(train_s1_set):,} ({len(train_s1_set)/(len(train_s1_set)+len(val_s1_set))*100:.2f}%)")
    print(f"  Val S1 Entities:   {len(val_s1_set):,} ({len(val_s1_set)/(len(train_s1_set)+len(val_s1_set))*100:.2f}%)")

    # Calculate detailed distribution stats
    def compute_distribution_stats(s1_set):
        card_counter = Counter()
        s2_links = 0
        s3_links = 0
        s1_only_s2 = 0
        s1_only_s3 = 0
        s1_both = 0
        s1_neither = 0
        
        for s1 in s1_set:
            matches = gt_loader.get_true_matches(s1)
            n_m = len(matches)
            card_counter[get_cardinality_bucket(n_m)] += 1
            
            s2_m = [m for m in matches if m.startswith("S2-")]
            s3_m = [m for m in matches if m.startswith("S3-")]
            
            s2_links += len(s2_m)
            s3_links += len(s3_m)
            
            if len(s2_m) > 0 and len(s3_m) > 0:
                s1_both += 1
            elif len(s2_m) > 0:
                s1_only_s2 += 1
            elif len(s3_m) > 0:
                s1_only_s3 += 1
            else:
                s1_neither += 1
                
        return {
            'total_entities': len(s1_set),
            'cardinality_dist': dict(sorted(card_counter.items())),
            's2_links': s2_links,
            's3_links': s3_links,
            'total_links': s2_links + s3_links,
            's1_only_s2': s1_only_s2,
            's1_only_s3': s1_only_s3,
            's1_both': s1_both,
            's1_neither': s1_neither
        }

    train_stats = compute_distribution_stats(train_s1_set)
    val_stats = compute_distribution_stats(val_s1_set)

    split_summary = {
        'seed': seed,
        'train_stats': train_stats,
        'val_stats': val_stats
    }

    summary_file_path = os.path.join("reports", "validation_split_summary.json")
    with open(summary_file_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    return split_summary

if __name__ == "__main__":
    create_grouped_validation_split()
