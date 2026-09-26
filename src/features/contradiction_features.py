import numpy as np

class ContradictionFeatureExtractor:
    def __init__(self, thresholds=None):
        t = thresholds or {}
        self.strong_name_thresh = float(t.get("strong_name_threshold", 0.85))
        self.strong_addr_thresh = float(t.get("strong_address_threshold", 0.80))
        self.low_name_thresh = float(t.get("low_name_similarity", 0.35))

    def extract_pair_features(self, name_feats, addr_feats):
        """
        name_feats: dict of Family A values
        addr_feats: dict of Family B values
        Returns dict of Feature Family H values.
        """
        postal_conflict = int(addr_feats["postal_conflict"])
        house_conflict = int(addr_feats["house_number_conflict"])
        locality_conflict = int(addr_feats["locality_conflict"])

        num_overlap = float(addr_feats["address_numeric_token_overlap"])
        num_diff = int(addr_feats["address_numeric_token_count_diff"])
        num_address_conflict = 1 if (num_overlap < 0.5 and num_diff > 0) else 0

        name_jw = float(name_feats["name_jaro_winkler_similarity"])
        name_jac = float(name_feats["name_token_jaccard"])
        name_sim = max(name_jw, name_jac)

        addr_jw = float(addr_feats["address_jaro_winkler_similarity"])
        addr_jac = float(addr_feats["address_token_jaccard"])
        addr_sim = max(addr_jw, addr_jac)

        strong_name = 1 if name_sim >= self.strong_name_thresh else 0
        strong_addr = 1 if addr_sim >= self.strong_addr_thresh else 0

        sn_postal_conflict = 1 if (strong_name == 1 and postal_conflict == 1) else 0
        sn_house_conflict = 1 if (strong_name == 1 and house_conflict == 1) else 0
        sn_loc_conflict = 1 if (strong_name == 1 and locality_conflict == 1) else 0
        sa_name_conflict = 1 if (strong_addr == 1 and name_sim <= self.low_name_thresh) else 0

        overall_count = (
            postal_conflict + house_conflict + locality_conflict +
            num_address_conflict + sn_postal_conflict + sn_house_conflict +
            sn_loc_conflict + sa_name_conflict
        )

        contradiction_score = float(
            1.5 * postal_conflict + 1.5 * house_conflict + 1.0 * locality_conflict +
            1.0 * num_address_conflict + 2.0 * sn_postal_conflict + 2.0 * sn_house_conflict
        )

        return {
            "postal_conflict": np.int8(postal_conflict),
            "house_number_conflict": np.int8(house_conflict),
            "locality_conflict": np.int8(locality_conflict),
            "numeric_address_conflict": np.int8(num_address_conflict),
            "strong_name_postal_conflict": np.int8(sn_postal_conflict),
            "strong_name_house_conflict": np.int8(sn_house_conflict),
            "strong_name_locality_conflict": np.int8(sn_loc_conflict),
            "strong_address_name_conflict": np.int8(sa_name_conflict),
            "name_address_contradiction_score": np.float32(contradiction_score),
            "overall_contradiction_count": np.int8(overall_count)
        }
