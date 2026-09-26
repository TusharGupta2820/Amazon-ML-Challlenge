import os
import sys
from collections import defaultdict
from base_index import BaseBlockingIndex

class SuffixlessExactNameIndex(BaseBlockingIndex):
    def __init__(self, max_group_size=50):
        super().__init__(name="StrategyA_SuffixlessName")
        self.max_group_size = max_group_size
        self.s2_name_index = defaultdict(list)
        self.s3_name_index = defaultdict(list)

    def build(self, s2_records, s3_records):
        print(f"[{self.name}] Building suffixless name inverted index...", flush=True)
        for rec in s2_records:
            sn = rec.get("business_name_suffixless")
            if sn:
                self.s2_name_index[sn].append(rec["entity_id"])

        for rec in s3_records:
            sn = rec.get("business_name_suffixless")
            if sn:
                self.s3_name_index[sn].append(rec["entity_id"])

        print(f"[{self.name}] S2 unique suffixless keys: {len(self.s2_name_index):,}, S3: {len(self.s3_name_index):,}", flush=True)

    def get_candidates(self, s1_record):
        sn = s1_record.get("business_name_suffixless")
        if not sn:
            return set(), set()

        s2_matches = self.s2_name_index.get(sn, [])
        s3_matches = self.s3_name_index.get(sn, [])

        # Apply group size cap to prevent generic name explosion
        s2_cands = set(s2_matches[:self.max_group_size]) if len(s2_matches) <= self.max_group_size else set(s2_matches[:self.max_group_size])
        s3_cands = set(s3_matches[:self.max_group_size]) if len(s3_matches) <= self.max_group_size else set(s3_matches[:self.max_group_size])

        return s2_cands, s3_cands
