import numpy as np

class InteractionFeatureExtractor:
    def __init__(self, thresholds=None):
        t = thresholds or {}
        self.high_name_thresh = float(t.get("high_name_similarity", 0.80))
        self.low_name_thresh = float(t.get("low_name_similarity", 0.35))
        self.high_addr_thresh = float(t.get("high_address_similarity", 0.75))
        self.low_addr_thresh = float(t.get("low_address_similarity", 0.30))
        self.strong_name_thresh = float(t.get("strong_name_threshold", 0.85))
        self.strong_addr_thresh = float(t.get("strong_address_threshold", 0.80))

    def extract_pair_features(self, name_feats, addr_feats):
        """
        name_feats: dict of Family A values
        addr_feats: dict of Family B values
        Returns dict of Feature Family C values.
        """
        name_jac = float(name_feats["name_token_jaccard"])
        name_jw = float(name_feats["name_jaro_winkler_similarity"])
        name_exact_norm = int(name_feats["name_exact_normalized"])
        name_exact_sfx = int(name_feats["name_exact_suffixless"])

        addr_jac = float(addr_feats["address_token_jaccard"])
        addr_jw = float(addr_feats["address_jaro_winkler_similarity"])

        postal_exact = int(addr_feats["postal_exact"])
        postal_conflict = int(addr_feats["postal_conflict"])
        house_exact = int(addr_feats["house_number_exact"])
        house_conflict = int(addr_feats["house_number_conflict"])
        locality_overlap = float(addr_feats["locality_token_overlap"])

        name_tfidf = float(name_feats["name_tfidf_cosine"])
        addr_tfidf = float(addr_feats["address_tfidf_cosine"])

        # Interaction products
        name_x_address_jaccard = float(name_jac * addr_jac)
        name_x_address_tfidf = float(name_tfidf * addr_tfidf)
        name_x_postal_match = float(name_jac * postal_exact)
        name_x_house_match = float(name_jac * house_exact)
        name_x_locality_match = float(name_jac * locality_overlap)

        # High/Low Contradictions
        name_sim = max(name_jw, name_jac)
        addr_sim = max(addr_jw, addr_jac)

        name_high_address_low = 1 if (name_sim >= self.high_name_thresh and addr_sim <= self.low_addr_thresh) else 0
        address_high_name_low = 1 if (addr_sim >= self.high_addr_thresh and name_sim <= self.low_name_thresh) else 0
        strong_name_strong_address = 1 if (name_sim >= self.strong_name_thresh and addr_sim >= self.strong_addr_thresh) else 0

        has_address_conflict = 1 if (postal_conflict == 1 or house_conflict == 1) else 0

        strong_name_address_conflict = 1 if (name_sim >= self.strong_name_thresh and has_address_conflict == 1) else 0
        strong_address_name_conflict = 1 if (addr_sim >= self.strong_addr_thresh and name_sim <= self.low_name_thresh) else 0

        name_exact_and_address_conflict = 1 if (name_exact_norm == 1 and has_address_conflict == 1) else 0
        suffixless_exact_and_address_conflict = 1 if (name_exact_sfx == 1 and has_address_conflict == 1) else 0

        return {
            "name_x_address_jaccard": np.float32(name_x_address_jaccard),
            "name_x_address_tfidf": np.float32(name_x_address_tfidf),
            "name_x_postal_match": np.float32(name_x_postal_match),
            "name_x_house_match": np.float32(name_x_house_match),
            "name_x_locality_match": np.float32(name_x_locality_match),
            "name_high_address_low": np.int8(name_high_address_low),
            "address_high_name_low": np.int8(address_high_name_low),
            "strong_name_strong_address": np.int8(strong_name_strong_address),
            "strong_name_address_conflict": np.int8(strong_name_address_conflict),
            "strong_address_name_conflict": np.int8(strong_address_name_conflict),
            "name_exact_and_address_conflict": np.int8(name_exact_and_address_conflict),
            "suffixless_exact_and_address_conflict": np.int8(suffixless_exact_and_address_conflict)
        }
