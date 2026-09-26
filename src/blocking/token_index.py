import os
import sys
from collections import Counter, defaultdict
from base_index import BaseBlockingIndex

class RareTokenIndex(BaseBlockingIndex):
    def __init__(self, max_token_doc_freq=5000, rare_tokens_per_name=2, max_cands_per_token=50):
        super().__init__(name="StrategyB_RareToken")
        self.max_token_doc_freq = max_token_doc_freq
        self.rare_tokens_per_name = rare_tokens_per_name
        self.max_cands_per_token = max_cands_per_token
        
        self.token_df = Counter()
        self.s2_token_index = defaultdict(list)
        self.s3_token_index = defaultdict(list)

    def build(self, s2_records, s3_records):
        print(f"[{self.name}] Building rare token index...", flush=True)
        
        # Step 1: Count Document Frequencies
        for rec in s2_records:
            tokens = set(rec.get("business_name_tokens", []))
            for tok in tokens:
                self.token_df[tok] += 1

        for rec in s3_records:
            tokens = set(rec.get("business_name_tokens", []))
            for tok in tokens:
                self.token_df[tok] += 1

        # Step 2: Build inverted index for non-common tokens
        for rec in s2_records:
            eid = rec["entity_id"]
            tokens = set(rec.get("business_name_tokens", []))
            for tok in tokens:
                if self.token_df[tok] <= self.max_token_doc_freq:
                    self.s2_token_index[tok].append(eid)

        for rec in s3_records:
            eid = rec["entity_id"]
            tokens = set(rec.get("business_name_tokens", []))
            for tok in tokens:
                if self.token_df[tok] <= self.max_token_doc_freq:
                    self.s3_token_index[tok].append(eid)

        print(f"[{self.name}] Indexed {len(self.token_df):,} total unique tokens.", flush=True)

    def get_candidates(self, s1_record):
        tokens = set(s1_record.get("business_name_tokens", []))
        if not tokens:
            return set(), set()

        # Sort tokens by ascending document frequency (rarest first)
        valid_tokens = [t for t in tokens if t in self.token_df and self.token_df[t] <= self.max_token_doc_freq]
        if not valid_tokens:
            return set(), set()

        sorted_tokens = sorted(valid_tokens, key=lambda t: self.token_df[t])
        selected_tokens = sorted_tokens[:self.rare_tokens_per_name]

        s2_cands = set()
        s3_cands = set()

        for tok in selected_tokens:
            s2_matches = self.s2_token_index.get(tok, [])
            s3_matches = self.s3_token_index.get(tok, [])
            
            s2_cands.update(s2_matches[:self.max_cands_per_token])
            s3_cands.update(s3_matches[:self.max_cands_per_token])

        return s2_cands, s3_cands
