import math
import numpy as np

class FrequencyFeatureExtractor:
    def __init__(self, name_freqs=None, sfx_freqs=None, addr_freqs=None, postal_freqs=None, token_freqs=None, total_entities=1):
        self.name_freqs = name_freqs or {}
        self.sfx_freqs = sfx_freqs or {}
        self.addr_freqs = addr_freqs or {}
        self.postal_freqs = postal_freqs or {}
        self.token_freqs = token_freqs or {}
        self.total_entities = max(1, total_entities)

    def extract_pair_features(self, rec1, rec2, total_cands_for_s1=1, cand_source_group_size=1):
        """
        rec1: dict for S1 entity
        rec2: dict for candidate entity
        total_cands_for_s1: total candidates generated for this S1
        cand_source_group_size: total candidates generated for this S1 from candidate_source (S2 or S3)
        """
        norm1 = rec1.get("business_name_norm", "")
        norm2 = rec2.get("business_name_norm", "")
        sfx1 = rec1.get("business_name_suffixless", "")
        sfx2 = rec2.get("business_name_suffixless", "")
        addr1 = rec1.get("business_address_norm", "")
        addr2 = rec2.get("business_address_norm", "")
        p1 = rec1.get("postal_code") or rec1.get("postal_code_candidate", "")
        p2 = rec2.get("postal_code") or rec2.get("postal_code_candidate", "")

        toks1 = set(rec1.get("name_tokens") or rec1.get("business_name_tokens", []))
        toks2 = set(rec2.get("name_tokens") or rec2.get("business_name_tokens", []))
        shared_toks = toks1.intersection(toks2)

        # Raw Frequencies
        f_n1 = self.name_freqs.get(norm1, 1) if norm1 else 1
        f_n2 = self.name_freqs.get(norm2, 1) if norm2 else 1
        f_s1 = self.sfx_freqs.get(sfx1, 1) if sfx1 else 1
        f_s2 = self.sfx_freqs.get(sfx2, 1) if sfx2 else 1
        f_a1 = self.addr_freqs.get(addr1, 1) if addr1 else 1
        f_a2 = self.addr_freqs.get(addr2, 1) if addr2 else 1
        f_p1 = self.postal_freqs.get(p1, 1) if p1 else 1
        f_p2 = self.postal_freqs.get(p2, 1) if p2 else 1

        # Rarity (IDF = log(N / freq))
        r_n1 = float(math.log(self.total_entities / f_n1))
        r_n2 = float(math.log(self.total_entities / f_n2))
        r_s1 = float(math.log(self.total_entities / f_s1))
        r_s2 = float(math.log(self.total_entities / f_s2))

        # Shared Token Rarity
        if shared_toks:
            tok_idfs = [math.log(self.total_entities / self.token_freqs.get(t, 1)) for t in shared_toks]
            avg_shared_rarity = float(sum(tok_idfs) / len(tok_idfs))
            max_shared_rarity = float(max(tok_idfs))
        else:
            avg_shared_rarity = 0.0
            max_shared_rarity = 0.0

        return {
            "s1_name_frequency": np.int32(f_n1),
            "candidate_name_frequency": np.int32(f_n2),
            "s1_suffixless_name_frequency": np.int32(f_s1),
            "candidate_suffixless_name_frequency": np.int32(f_s2),
            "s1_address_frequency": np.int32(f_a1),
            "candidate_address_frequency": np.int32(f_a2),
            "s1_postal_frequency": np.int32(f_p1),
            "candidate_postal_frequency": np.int32(f_p2),
            "s1_name_rarity": np.float32(r_n1),
            "candidate_name_rarity": np.float32(r_n2),
            "s1_suffixless_rarity": np.float32(r_s1),
            "candidate_suffixless_rarity": np.float32(r_s2),
            "candidate_group_size": np.int16(total_cands_for_s1),
            "candidate_source_group_size": np.int16(cand_source_group_size),
            "shared_token_rarity": np.float32(avg_shared_rarity),
            "rarest_shared_token_rarity": np.float32(max_shared_rarity)
        }
