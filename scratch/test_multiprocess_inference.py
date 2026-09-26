import os
import sys
import time
import json
import pyarrow.parquet as pq
import numpy as np
import xgboost as xgb
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(base_dir, "src", "preprocessing"))
sys.path.insert(0, os.path.join(base_dir, "src", "blocking"))
sys.path.insert(0, os.path.join(base_dir, "src", "features"))
sys.path.insert(0, os.path.join(base_dir, "src", "models"))
sys.path.insert(0, os.path.join(base_dir, "src", "evaluation"))

from normalization import BusinessNormalizer
from feature_engineering import PairFeatureEngine, load_tsv_records_binary
from name_features import NameFeatureExtractor
from frequency_features import FrequencyFeatureExtractor

FINAL_THRESHOLD = 0.96

def process_rg_range(rg_indices, test_s1_records, cand_records, rare_token_set, expected_feature_cols, model_path, test_pairs_parquet):
    # Worker process function
    model = xgb.Booster()
    model.load_model(model_path)
    pf = pq.ParquetFile(test_pairs_parquet)

    pred_matches = defaultdict(list)
    total_preds = 0
    s2_preds = 0
    s3_preds = 0
    processed = 0

    for rg in rg_indices:
        tbl = pf.read_row_group(rg)
        b_len = len(tbl)
        b_s1 = tbl.column("source1_entity_id").to_pylist()
        b_cand = tbl.column("candidate_entity_id").to_pylist()
        b_src = tbl.column("candidate_source").to_pylist()

        X_b = np.zeros((b_len, len(expected_feature_cols)), dtype=np.float32)
        valid_indices = []

        for i in range(b_len):
            s1_id = b_s1[i]
            c_id = b_cand[i]
            c_src = b_src[i]
            rec1 = test_s1_records.get(s1_id)
            rec2 = cand_records.get(c_id)
            if not rec1 or not rec2:
                continue

            n_sfx1 = rec1.get("business_name_suffixless", "")
            n_sfx2 = rec2.get("business_name_suffixless", "")
            hit_a = 1 if (n_sfx1 and n_sfx1 == n_sfx2) else 0

            a_norm1 = rec1.get("business_address_norm", "")
            a_norm2 = rec2.get("business_address_norm", "")
            p1 = rec1.get("postal_code", "")
            p2 = rec2.get("postal_code", "")
            hit_c = 1 if (a_norm1 and a_norm1 == a_norm2 and p1 and p1 == p2) else 0

            t1 = set(rec1.get("name_tokens", []))
            t2 = set(rec2.get("name_tokens", []))
            hit_b = 1 if (t1 & t2 & rare_token_set) else 0

            at1 = set(rec1.get("address_tokens", [])) if "address_tokens" in rec1 else set(a_norm1.split())
            at2 = set(rec2.get("address_tokens", [])) if "address_tokens" in rec2 else set(a_norm2.split())
            has_name_tok = bool(t1 & t2)
            has_addr_tok = bool(at1 & at2)

            if not hit_a and not hit_c and not hit_b and not has_name_tok and not has_addr_tok:
                continue

            valid_indices.append(i)

        processed += b_len

    return len(valid_indices), processed

if __name__ == "__main__":
    test_pairs_parquet = os.path.join(base_dir, "artifacts", "test_pairs", "test_candidate_pairs.parquet")
    pf = pq.ParquetFile(test_pairs_parquet)
    print(f"Total row groups: {pf.num_row_groups}, total rows: {pf.metadata.num_rows:,}")
