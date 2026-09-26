import numpy as np

class CountryFeatureExtractor:
    def __init__(self):
        pass

    def extract_pair_features(self, rec1, rec2):
        """
        rec1: dict for S1 entity
        rec2: dict for candidate entity
        Returns dict of Feature Family D values.
        """
        c1 = (rec1.get("country") or rec1.get("country_raw") or rec1.get("country_norm", "")).upper().strip()
        c2 = (rec2.get("country") or rec2.get("country_raw") or rec2.get("country_norm", "")).upper().strip()

        missing1 = 1 if not c1 else 0
        missing2 = 1 if not c2 else 0
        missing_either = 1 if (missing1 or missing2) else 0
        both_present = 1 if (not missing1 and not missing2) else 0

        if both_present == 1:
            exact_match = 1 if c1 == c2 else 0
            mismatch = 1 if c1 != c2 else 0
        else:
            exact_match = 0
            mismatch = 0

        return {
            "country_exact_match": np.int8(exact_match),
            "country_mismatch": np.int8(mismatch),
            "country_missing_s1": np.int8(missing1),
            "country_missing_candidate": np.int8(missing2),
            "country_missing_either": np.int8(missing_either),
            "country_both_present": np.int8(both_present)
        }
