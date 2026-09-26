import os
import sys

class GroundTruthLoader:
    def __init__(self, gt_path="dataset/train/train_ground_truth.tsv"):
        self.gt_path = gt_path
        self.gt_map = {}  # s1_id -> set of matched_entity_ids
        self._load()

    def _load(self):
        if not os.path.isfile(self.gt_path):
            raise FileNotFoundError(f"Ground truth file not found at: {self.gt_path}")
            
        print(f"Loading Ground Truth from {self.gt_path}...", flush=True)
        with open(self.gt_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\r\n").split("\t")
                s1_id = parts[0].strip()
                matched_str = parts[1].strip() if len(parts) > 1 else ""
                
                if not matched_str:
                    matched_ids = set()
                else:
                    matched_ids = set(m.strip() for m in matched_str.split(',') if m.strip())
                    
                self.gt_map[s1_id] = matched_ids
                
        print(f"Loaded {len(self.gt_map):,} S1 ground-truth entries.", flush=True)

    def get_true_matches(self, s1_id):
        return self.gt_map.get(s1_id, set())

    def get_all_s1_ids(self):
        return list(self.gt_map.keys())

    def get_match_count(self, s1_id):
        return len(self.gt_map.get(s1_id, set()))

if __name__ == "__main__":
    gt = GroundTruthLoader()
    s1_ids = gt.get_all_s1_ids()
    print(f"Sample S1 ID: {s1_ids[0]}, matches: {gt.get_true_matches(s1_ids[0])}")
