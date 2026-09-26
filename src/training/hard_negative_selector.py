import os

def compute_jaccard(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

class HardNegativeSelector:
    """Categorizes non-ground-truth candidate pairs into hard negative categories

    Categories:
      1. high_name_and_address_similarity (name Jaccard >= 0.4 AND address Jaccard >= 0.3)
      2. name_conflict (same normalized_name or suffixless_name, but label = 0)
      3. address_conflict (same house_number + postal_code, non-empty, but label = 0)
      4. high_name_similarity (name Jaccard >= 0.5)
      5. high_address_similarity (address Jaccard >= 0.4)
      6. easy_negative (all other non-matches)
    """

    def __init__(self):
        pass

    def categorize_pair(self, s1_norm_rec, cand_norm_rec):
        """Categorize a non-match pair into its primary hard negative category."""
        s1_name_norm = s1_norm_rec.get("normalized_name", "")
        cand_name_norm = cand_norm_rec.get("normalized_name", "")
        
        s1_suffixless = s1_norm_rec.get("suffixless_name", "")
        cand_suffixless = cand_norm_rec.get("suffixless_name", "")

        s1_name_tokens = set(s1_norm_rec.get("name_tokens", []))
        cand_name_tokens = set(cand_norm_rec.get("name_tokens", []))

        s1_addr_tokens = set(s1_norm_rec.get("address_tokens", []))
        cand_addr_tokens = set(cand_norm_rec.get("address_tokens", []))

        s1_hn = s1_norm_rec.get("house_number", "")
        cand_hn = cand_norm_rec.get("house_number", "")
        s1_zip = s1_norm_rec.get("postal_code", "")
        cand_zip = cand_norm_rec.get("postal_code", "")

        name_jaccard = compute_jaccard(s1_name_tokens, cand_name_tokens)
        addr_jaccard = compute_jaccard(s1_addr_tokens, cand_addr_tokens)

        # Check Category 1: High Name + Address Similarity
        if name_jaccard >= 0.4 and addr_jaccard >= 0.3:
            return "high_name_and_address_similarity"

        # Check Category 2: Name Conflict (exact normalized or suffixless match)
        if (s1_name_norm and s1_name_norm == cand_name_norm) or \
           (s1_suffixless and s1_suffixless == cand_suffixless):
            return "name_conflict"

        # Check Category 3: Address Conflict (same non-empty house number + zip code)
        if s1_hn and s1_zip and s1_hn == cand_hn and s1_zip == cand_zip:
            return "address_conflict"

        # Check Category 4: High Name Similarity
        if name_jaccard >= 0.5:
            return "high_name_similarity"

        # Check Category 5: High Address Similarity
        if addr_jaccard >= 0.4:
            return "high_address_similarity"

        # Fallback: Easy Negative
        return "easy_negative"
