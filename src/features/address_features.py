import math
import numpy as np
import rapidfuzz

class AddressFeatureExtractor:
    def __init__(self):
        pass

    def extract_pair_features(self, rec1, rec2):
        """
        rec1: dict for S1 entity
        rec2: dict for candidate entity
        Returns dict of Feature Family B values.
        """
        raw1 = rec1.get("business_address_raw", "")
        raw2 = rec2.get("business_address_raw", "")
        norm1 = rec1.get("business_address_norm", "")
        norm2 = rec2.get("business_address_norm", "")

        toks1 = rec1.get("address_tokens") or rec1.get("business_address_tokens", [])
        toks2 = rec2.get("address_tokens") or rec2.get("business_address_tokens", [])

        # Missingness flags
        missing1 = 1 if not norm1 else 0
        missing2 = 1 if not norm2 else 0
        missing_either = 1 if (missing1 or missing2) else 0
        both_present = 1 if (not missing1 and not missing2) else 0

        # Exact matches
        addr_exact_raw = 1 if (raw1 and raw1 == raw2) else 0
        addr_exact_norm = 1 if (norm1 and norm1 == norm2) else 0

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

        token_jaccard = float(shared_count / union_len) if union_len > 0 else 0.0
        overlap_s1_cand = float(shared_count / len1) if len1 > 0 else 0.0
        overlap_cand_s1 = float(shared_count / len2) if len2 > 0 else 0.0

        # Char 3-grams
        ng1 = set(rec1.get("address_3grams", [])) if "address_3grams" in rec1 else set([norm1[i:i+3] for i in range(len(norm1)-2)])
        ng2 = set(rec2.get("address_3grams", [])) if "address_3grams" in rec2 else set([norm2[i:i+3] for i in range(len(norm2)-2)])
        ng_int = len(ng1.intersection(ng2))
        ng_union = len(ng1.union(ng2))
        char_3gram_jaccard = float(ng_int / ng_union) if ng_union > 0 else 0.0
        denom_cos = math.sqrt(len(ng1) * len(ng2))
        char_3gram_cosine = float(ng_int / denom_cos) if denom_cos > 0 else 0.0

        # Length & Count differences
        l1, l2 = len(norm1), len(norm2)
        len_abs_diff = abs(l1 - l2)
        max_l = max(l1, l2)
        len_ratio = float(min(l1, l2) / max_l) if max_l > 0 else (1.0 if not missing_either else 0.0)

        tok_abs_diff = abs(len(toks1) - len(toks2))

        # Numeric token overlap in address
        num1 = [t for t in toks1 if t.isdigit()]
        num2 = [t for t in toks2 if t.isdigit()]
        num_diff = abs(len(num1) - len(num2))
        if num1 and num2:
            num_int = len(set(num1).intersection(set(num2)))
            num_un = len(set(num1).union(set(num2)))
            num_overlap = float(num_int / num_un) if num_un > 0 else 0.0
        else:
            num_overlap = 1.0 if (not num1 and not num2) else 0.0

        # House numbers
        hn1 = rec1.get("house_number") or rec1.get("house_number_candidate", "")
        hn2 = rec2.get("house_number") or rec2.get("house_number_candidate", "")
        if hn1 and hn2:
            hn_exact = 1 if hn1 == hn2 else 0
            hn_conflict = 1 if hn1 != hn2 else 0
        else:
            hn_exact = 0
            hn_conflict = 0

        # Postal codes
        p1 = rec1.get("postal_code") or rec1.get("postal_code_candidate", "")
        p2 = rec2.get("postal_code") or rec2.get("postal_code_candidate", "")
        if p1 and p2:
            postal_exact = 1 if p1 == p2 else 0
            postal_conflict = 1 if p1 != p2 else 0
            prefix_match = 1 if (p1[:3] == p2[:3] and len(p1) >= 3 and len(p2) >= 3) else 0
        else:
            postal_exact = 0
            postal_conflict = 0
            prefix_match = 0

        # Locality (City / Region tokens)
        loc1 = rec1.get("locality_tokens", set())
        loc2 = rec2.get("locality_tokens", set())
        if loc1 and loc2:
            loc_int = len(loc1.intersection(loc2))
            loc_un = len(loc1.union(loc2))
            locality_overlap = float(loc_int / loc_un) if loc_un > 0 else 0.0
            locality_conflict = 1 if loc_int == 0 else 0
        else:
            locality_overlap = 0.0
            locality_conflict = 0

        # TF-IDF Cosine approximation
        addr_tfidf_cosine = float(0.6 * char_3gram_cosine + 0.4 * token_jaccard)

        return {
            "address_exact_raw": np.int8(addr_exact_raw),
            "address_exact_normalized": np.int8(addr_exact_norm),
            "address_token_jaccard": np.float32(token_jaccard),
            "address_token_overlap_s1_to_candidate": np.float32(overlap_s1_cand),
            "address_token_overlap_candidate_to_s1": np.float32(overlap_cand_s1),
            "address_levenshtein_similarity": np.float32(lev_sim),
            "address_jaro_winkler_similarity": np.float32(jw_sim),
            "address_char_3gram_jaccard": np.float32(char_3gram_jaccard),
            "address_char_3gram_cosine": np.float32(char_3gram_cosine),
            "address_tfidf_cosine": np.float32(addr_tfidf_cosine),
            "address_length_abs_diff": np.int16(len_abs_diff),
            "address_length_ratio": np.float32(len_ratio),
            "address_token_count_abs_diff": np.int16(tok_abs_diff),
            "address_shared_token_count": np.int16(shared_count),
            "address_numeric_token_overlap": np.float32(num_overlap),
            "address_numeric_token_count_diff": np.int16(num_diff),
            "house_number_exact": np.int8(hn_exact),
            "house_number_conflict": np.int8(hn_conflict),
            "postal_exact": np.int8(postal_exact),
            "postal_conflict": np.int8(postal_conflict),
            "postal_prefix_match": np.int8(prefix_match),
            "locality_token_overlap": np.float32(locality_overlap),
            "locality_conflict": np.int8(locality_conflict),
            "address_missing_s1": np.int8(missing1),
            "address_missing_candidate": np.int8(missing2),
            "address_missing_either": np.int8(missing_either),
            "address_both_present": np.int8(both_present)
        }
