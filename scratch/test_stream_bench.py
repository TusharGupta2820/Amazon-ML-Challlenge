import time
import numpy as np

def fast_blocking_hits(rec1, rec2, rare_token_set):
    n_sfx1 = rec1.get("business_name_suffixless", "")
    n_sfx2 = rec2.get("business_name_suffixless", "")
    hit_a = 1 if (n_sfx1 and n_sfx1 == n_sfx2) else 0

    a_norm1 = rec1.get("business_address_norm", "")
    a_norm2 = rec2.get("business_address_norm", "")
    p1 = rec1.get("postal_code", "")
    p2 = rec2.get("postal_code", "")
    hit_c = 1 if (a_norm1 and a_norm1 == a_norm2 and p1 and p1 == p2) else 0

    t1 = set(rec1.get("name_tokens", []))
    t2 = set(rec2.get("name_tokens", []))
    hit_b = 1 if (t1 & t2 & rare_token_set) else 0

    hit_d = 1
    return hit_a, hit_b, hit_c, hit_d

t0 = time.time()
r1 = {"business_name_suffixless": "acme supply", "business_address_norm": "123 main st", "postal_code": "10001", "name_tokens": ["acme", "supply"]}
r2 = {"business_name_suffixless": "acme supply", "business_address_norm": "123 main st", "postal_code": "10001", "name_tokens": ["acme", "supply"]}
rare_toks = {"acme"}

for _ in range(100000):
    fast_blocking_hits(r1, r2, rare_toks)

elapsed = time.time() - t0
print(f"100,000 pair blocking hits computed in {elapsed:.4f}s ({100000/elapsed:,.0f} pairs/sec).")
