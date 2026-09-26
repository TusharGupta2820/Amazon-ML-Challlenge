import os
import sys
import time

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(base_dir, "src", "features"))

from feature_engineering import PairFeatureEngine

print("Testing PairFeatureEngine sample execution...", flush=True)
engine = PairFeatureEngine()
engine.load_and_index_entities()

sample_s1_id = list(engine.s1_records.keys())[0]
sample_rec1 = engine.s1_records[sample_s1_id]

all_cands, sa, sb, sc, sd = engine.cand_gen.generate_candidates_with_per_strategy_breakdown(sample_rec1)
print(f"Sample S1 ID: {sample_s1_id} -> Generated {len(all_cands)} candidates.", flush=True)

if all_cands:
    cand_id = all_cands[0]
    cand_src = "S2" if cand_id.startswith("S2-") else "S3"
    feats = engine.extract_pair(
        sample_s1_id, cand_id, cand_src, rank=1,
        total_cands_for_s1=len(all_cands), cand_src_group_size=len(all_cands),
        strategy_sets=(sa, sb, sc, sd)
    )
    print(f"Extracted {len(feats)} total pairwise features for pair ({sample_s1_id}, {cand_id}):")
    for k, v in list(feats.items())[:15]:
        print(f"  {k}: {v} ({type(v)})")

t0 = time.time()
n_samples = 10000
for i in range(n_samples):
    feats = engine.extract_pair(
        sample_s1_id, cand_id, cand_src, rank=i+1,
        total_cands_for_s1=len(all_cands), cand_src_group_size=len(all_cands),
        strategy_sets=(sa, sb, sc, sd)
    )
dt = time.time() - t0
print(f"\nBenchmark: Extracted features for {n_samples:,} pairs in {dt:.3f}s ({n_samples/dt:,.0f} pairs/sec).", flush=True)
