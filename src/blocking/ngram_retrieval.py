import os
import sys
from collections import Counter, defaultdict
from base_index import BaseBlockingIndex

def generate_char_ngrams(text, n=3):
    if not text or len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]

class CharacterNGramIndex(BaseBlockingIndex):
    def __init__(self, ngram_size=3, top_k=20, max_ngram_df=2000, max_ngrams_per_entity=4):
        super().__init__(name="StrategyD_CharNGram")
        self.ngram_size = ngram_size
        self.top_k = top_k
        self.max_ngram_df = max_ngram_df
        self.max_ngrams_per_entity = max_ngrams_per_entity
        
        self.ngram_df = Counter()
        self.s2_ngram_index = defaultdict(list)
        self.s3_ngram_index = defaultdict(list)

    def build(self, s2_records, s3_records):
        print(f"[{self.name}] Building character {self.ngram_size}-gram index...", flush=True)
        
        # Step 1: Count Document Frequencies
        for rec in s2_records:
            ngrams = set(generate_char_ngrams(rec.get("business_name_norm", ""), n=self.ngram_size))
            for ng in ngrams:
                self.ngram_df[ng] += 1

        for rec in s3_records:
            ngrams = set(generate_char_ngrams(rec.get("business_name_norm", ""), n=self.ngram_size))
            for ng in ngrams:
                self.ngram_df[ng] += 1

        # Step 2: Build inverted index for non-common n-grams
        for rec in s2_records:
            eid = rec["entity_id"]
            ngrams = set(generate_char_ngrams(rec.get("business_name_norm", ""), n=self.ngram_size))
            for ng in ngrams:
                if self.ngram_df[ng] <= self.max_ngram_df:
                    self.s2_ngram_index[ng].append(eid)

        for rec in s3_records:
            eid = rec["entity_id"]
            ngrams = set(generate_char_ngrams(rec.get("business_name_norm", ""), n=self.ngram_size))
            for ng in ngrams:
                if self.ngram_df[ng] <= self.max_ngram_df:
                    self.s3_ngram_index[ng].append(eid)

        print(f"[{self.name}] Indexed {len(self.ngram_df):,} unique n-grams.", flush=True)

    def get_candidates(self, s1_record):
        ngrams = set(generate_char_ngrams(s1_record.get("business_name_norm", ""), n=self.ngram_size))
        if not ngrams:
            return set(), set()

        valid_ngrams = [ng for ng in ngrams if ng in self.ngram_df and self.ngram_df[ng] <= self.max_ngram_df]
        if not valid_ngrams:
            return set(), set()

        # Sort n-grams by ascending document frequency (rarest first) and pick top rarest
        sorted_ngrams = sorted(valid_ngrams, key=lambda ng: self.ngram_df[ng])[:self.max_ngrams_per_entity]

        s2_overlap = Counter()
        s3_overlap = Counter()

        for ng in sorted_ngrams:
            for s2_id in self.s2_ngram_index.get(ng, []):
                s2_overlap[s2_id] += 1
            for s3_id in self.s3_ngram_index.get(ng, []):
                s3_overlap[s3_id] += 1

        top_s2 = set(e[0] for e in s2_overlap.most_common(self.top_k))
        top_s3 = set(e[0] for e in s3_overlap.most_common(self.top_k))

        return top_s2, top_s3
