import math
import numpy as np
import rapidfuzz

class NameFeatureExtractor:
    def __init__(self, rare_token_set=None):
        self.rare_token_set = rare_token_set or set()

    def extract_pair_features(self, rec1, rec2):
        """
        rec1: dict for S1 entity (raw, norm, compact, suffixless, tokens, etc.)
        rec2: dict for candidate entity
        Returns dict of Feature Family A values.
        """
        raw1 = rec1.get("business_name_raw", "")
        raw2 = rec2.get("business_name_raw", "")
        norm1 = rec1.get("business_name_norm", "")
        norm2 = rec2.get("business_name_norm", "")
        sfx1 = rec1.get("business_name_suffixless", "")
        sfx2 = rec2.get("business_name_suffixless", "")
        cmp1 = rec1.get("business_name_compact", "")
        cmp2 = rec2.get("business_name_compact", "")

        toks1 = rec1.get("name_tokens") or rec1.get("business_name_tokens", [])
        toks2 = rec2.get("name_tokens") or rec2.get("business_name_tokens", [])

        # Exact matches
        name_exact_raw = 1 if (raw1 and raw1 == raw2) else 0
        name_exact_normalized = 1 if (norm1 and norm1 == norm2) else 0
        name_exact_suffixless = 1 if (sfx1 and sfx1 == sfx2) else 0
        name_compact_exact = 1 if (cmp1 and cmp1 == cmp2) else 0

        # Rapidfuzz String similarities
        if norm1 and norm2:
            lev_sim = float(rapidfuzz.distance.Levenshtein.normalized_similarity(norm1, norm2))
            jw_sim = float(rapidfuzz.distance.JaroWinkler.similarity(norm1, norm2))
        else:
            lev_sim = 0.0
            jw_sim = 0.0

        # Token operations
        set1 = set(toks1)
        set2 = set(toks2)
        len1 = len(set1)
        len2 = len(set2)

        intersection = set1.intersection(set2)
        union_len = len(set1.union(set2))
        shared_count = len(intersection)

        name_token_jaccard = float(shared_count / union_len) if union_len > 0 else 0.0
        overlap_s1_cand = float(shared_count / len1) if len1 > 0 else 0.0
        overlap_cand_s1 = float(shared_count / len2) if len2 > 0 else 0.0

        # Char 3-gram Jaccard & Cosine
        ng1 = rec1.get("name_3grams", set())
        ng2 = rec2.get("name_3grams", set())
        ng_int = len(ng1.intersection(ng2))
        ng_union = len(ng1.union(ng2))
        char_3gram_jaccard = float(ng_int / ng_union) if ng_union > 0 else 0.0
        denom_cos = math.sqrt(len(ng1) * len(ng2))
        char_3gram_cosine = float(ng_int / denom_cos) if denom_cos > 0 else 0.0

        # Length & Count differences
        l1, l2 = len(norm1), len(norm2)
        len_abs_diff = abs(l1 - l2)
        max_l = max(l1, l2)
        len_ratio = float(min(l1, l2) / max_l) if max_l > 0 else 1.0

        tok_abs_diff = abs(len(toks1) - len(toks2))

        # Positional token matches
        first_match = 1 if (toks1 and toks2 and toks1[0] == toks2[0]) else 0
        last_match = 1 if (toks1 and toks2 and toks1[-1] == toks2[-1]) else 0

        # Numeric token overlap in name
        num1 = [t for t in toks1 if t.isdigit()]
        num2 = [t for t in toks2 if t.isdigit()]
        if num1 and num2:
            num_int = len(set(num1).intersection(set(num2)))
            num_un = len(set(num1).union(set(num2)))
            num_overlap = float(num_int / num_un) if num_un > 0 else 0.0
        else:
            num_overlap = 1.0 if (not num1 and not num2) else 0.0

        # Rare token overlap
        rare_shared = len(intersection.intersection(self.rare_token_set)) if self.rare_token_set else 0

        # Approximate TF-IDF similarity via char/word n-gram overlap weighting
        name_tfidf_cosine = float(0.6 * char_3gram_cosine + 0.4 * name_token_jaccard)

        return {
            "name_exact_raw": np.int8(name_exact_raw),
            "name_exact_normalized": np.int8(name_exact_normalized),
            "name_exact_suffixless": np.int8(name_exact_suffixless),
            "name_levenshtein_similarity": np.float32(lev_sim),
            "name_jaro_winkler_similarity": np.float32(jw_sim),
            "name_token_jaccard": np.float32(name_token_jaccard),
            "name_token_overlap_s1_to_candidate": np.float32(overlap_s1_cand),
            "name_token_overlap_candidate_to_s1": np.float32(overlap_cand_s1),
            "name_char_3gram_jaccard": np.float32(char_3gram_jaccard),
            "name_char_3gram_cosine": np.float32(char_3gram_cosine),
            "name_tfidf_cosine": np.float32(name_tfidf_cosine),
            "name_length_abs_diff": np.int16(len_abs_diff),
            "name_length_ratio": np.float32(len_ratio),
            "name_token_count_abs_diff": np.int16(tok_abs_diff),
            "name_first_token_match": np.int8(first_match),
            "name_last_token_match": np.int8(last_match),
            "name_numeric_token_overlap": np.float32(num_overlap),
            "name_shared_token_count": np.int16(shared_count),
            "name_shared_rare_token_count": np.int16(rare_shared),
            "name_compact_exact": np.int8(name_compact_exact)
        }
