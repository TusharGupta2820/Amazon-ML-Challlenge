import numpy as np

class BlockingAndSourceFeatureExtractor:
    def __init__(self):
        pass

    def extract_pair_features(self, set_a, set_b, set_c, set_d, cand_id, cand_src, rec2, rank=1):
        """
        set_a, set_b, set_c, set_d: sets of candidate IDs for Strategy A, B, C, D
        cand_id: candidate entity ID string (e.g. S2-12345 or S3-67890)
        cand_src: "S2" or "S3"
        rec2: candidate record dictionary
        rank: position index of candidate in list
        """
        hit_a = 1 if cand_id in set_a else 0
        hit_b = 1 if cand_id in set_b else 0
        hit_c = 1 if cand_id in set_c else 0
        hit_d = 1 if cand_id in set_d else 0

        strategy_count = hit_a + hit_b + hit_c + hit_d
        name_ev = 1 if (hit_a or hit_b or hit_d) else 0
        addr_ev = 1 if hit_c else 0
        multi_strat = 1 if strategy_count > 1 else 0

        # Source flags
        is_s2 = 1 if cand_src == "S2" else 0
        is_s3 = 1 if cand_src == "S3" else 0

        # Candidate missingness
        c_name_norm = rec2.get("business_name_norm", "")
        c_addr_norm = rec2.get("business_address_norm", "")
        c_country = rec2.get("country", "")

        c_name_missing = 1 if not c_name_norm else 0
        c_addr_missing = 1 if not c_addr_norm else 0
        c_country_missing = 1 if not c_country else 0

        return {
            "blocked_by_A_suffixless_name": np.int8(hit_a),
            "blocked_by_B_rare_token": np.int8(hit_b),
            "blocked_by_C_address_postal": np.int8(hit_c),
            "blocked_by_D_char_3gram": np.int8(hit_d),
            "blocking_strategy_count": np.int8(strategy_count),
            "blocking_name_evidence": np.int8(name_ev),
            "blocking_address_evidence": np.int8(addr_ev),
            "blocking_multi_strategy": np.int8(multi_strat),
            "blocking_candidate_rank_if_available": np.int32(rank),
            "candidate_is_s2": np.int8(is_s2),
            "candidate_is_s3": np.int8(is_s3),
            "s2_source_flag": np.int8(is_s2),
            "s3_source_flag": np.int8(is_s3),
            "candidate_address_missing": np.int8(c_addr_missing),
            "candidate_name_missing": np.int8(c_name_missing),
            "candidate_country_missing": np.int8(c_country_missing),
            "source_specific_name_length": np.int16(len(c_name_norm)),
            "source_specific_address_length": np.int16(len(c_addr_norm))
        }
