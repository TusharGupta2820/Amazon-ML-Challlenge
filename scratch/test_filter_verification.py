import pyarrow.parquet as pq
import numpy as np

train_parquet = "artifacts/features/train_features.parquet"
pf = pq.ParquetFile(train_parquet)

total_pos = 0
filtered_pos = 0

print("Verifying zero-leakage early filter on training dataset positives...")

for batch in pf.iter_batches(batch_size=500000):
    labels = batch.column("label").to_numpy()
    pos_mask = (labels == 1)
    if not np.any(pos_mask):
        continue

    tot = np.sum(pos_mask)
    total_pos += tot

    name_jaccard = batch.column("name_token_jaccard").to_numpy()[pos_mask]
    addr_jaccard = batch.column("address_token_jaccard").to_numpy()[pos_mask]
    exact_sfx = batch.column("name_exact_suffixless").to_numpy()[pos_mask]
    exact_raw = batch.column("name_exact_raw").to_numpy()[pos_mask]
    rare_tok = batch.column("name_shared_rare_token_count").to_numpy()[pos_mask]

    zero_overlap = (name_jaccard == 0) & (addr_jaccard == 0) & (exact_sfx == 0) & (exact_raw == 0) & (rare_tok == 0)
    num_zero = np.sum(zero_overlap)
    filtered_pos += num_zero

print(f"Total True Positive Pairs Evaluated: {total_pos:,}")
print(f"True Positives Filtered Out by Zero-Overlap Rule: {filtered_pos} ({filtered_pos/max(1, total_pos)*100:.4f}%)")
