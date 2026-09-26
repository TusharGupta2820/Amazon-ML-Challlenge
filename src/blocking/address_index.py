import os
import sys
from collections import defaultdict
from base_index import BaseBlockingIndex

class AddressPostcodeIndex(BaseBlockingIndex):
    def __init__(self, max_cands_per_key=50):
        super().__init__(name="StrategyC_AddressPostcode")
        self.max_cands_per_key = max_cands_per_key
        
        self.s2_addr_index = defaultdict(list)
        self.s3_addr_index = defaultdict(list)

    def _extract_keys(self, rec):
        if rec.get("address_missing", True):
            return []
            
        postcode = rec.get("postal_code_candidate", "")
        house_num = rec.get("house_number_candidate", "")
        tokens = [t for t in rec.get("business_address_tokens", []) if len(t) >= 4 and not t.isdigit()]
        
        keys = []
        if postcode and house_num:
            keys.append(f"ph:{postcode}_{house_num}")
            
        if postcode and tokens:
            # Pick first significant address token
            keys.append(f"pt:{postcode}_{tokens[0]}")
            
        if house_num and tokens:
            keys.append(f"ht:{house_num}_{tokens[0]}")
            
        return keys

    def build(self, s2_records, s3_records):
        print(f"[{self.name}] Building address & postcode index...", flush=True)
        for rec in s2_records:
            eid = rec["entity_id"]
            keys = self._extract_keys(rec)
            for k in keys:
                self.s2_addr_index[k].append(eid)

        for rec in s3_records:
            eid = rec["entity_id"]
            keys = self._extract_keys(rec)
            for k in keys:
                self.s3_addr_index[k].append(eid)

        print(f"[{self.name}] Indexed {len(self.s2_addr_index):,} S2 address keys, {len(self.s3_addr_index):,} S3 keys.", flush=True)

    def get_candidates(self, s1_record):
        keys = self._extract_keys(s1_record)
        if not keys:
            return set(), set()

        s2_cands = set()
        s3_cands = set()

        for k in keys:
            s2_matches = self.s2_addr_index.get(k, [])
            s3_matches = self.s3_addr_index.get(k, [])
            
            s2_cands.update(s2_matches[:self.max_cands_per_key])
            s3_cands.update(s3_matches[:self.max_cands_per_key])

        return s2_cands, s3_cands
