import time
import math
from collections import Counter

def levenshtein_sim(s1, s2):
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    if abs(len1 - len2) > max(len1, len2) * 0.8 and max(len1, len2) > 10:
        return 0.0  # Fast prune
    
    # Fast DP table
    dp = list(range(len2 + 1))
    for i, c1 in enumerate(s1):
        new_dp = [i + 1] * (len2 + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            new_dp[j + 1] = min(dp[j + 1] + 1, new_dp[j] + 1, dp[j] + cost)
        dp = new_dp
    dist = dp[len2]
    max_len = max(len1, len2)
    return 1.0 - dist / max_len

def jaro_winkler_sim(s1, s2):
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = (max(len1, len2) // 2) - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions //= 2
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0

    if jaro < 0.7:
        return jaro

    # Winkler prefix scale (max 4 chars)
    prefix = 0
    for c1, c2 in zip(s1[:4], s2[:4]):
        if c1 == c2:
            prefix += 1
        else:
            break

    return jaro + (prefix * 0.1 * (1.0 - jaro))

def token_jaccard(tokens1, tokens2):
    if not tokens1 or not tokens2:
        return 0.0
    set1 = set(tokens1) if isinstance(tokens1, list) else tokens1
    set2 = set(tokens2) if isinstance(tokens2, list) else tokens2
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union) if union > 0 else 0.0

def char_3gram_jaccard(ngrams1, ngrams2):
    if not ngrams1 or not ngrams2:
        return 0.0
    set1 = set(ngrams1) if isinstance(ngrams1, list) else ngrams1
    set2 = set(ngrams2) if isinstance(ngrams2, list) else ngrams2
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union) if union > 0 else 0.0

print("Benchmarking 100,000 string comparisons...", flush=True)
t0 = time.time()
n_pairs = 100000
for i in range(n_pairs):
    s1 = "urgent care & physical therapy center"
    s2 = "urgent care physical therapy clinic inc"
    lev = levenshtein_sim(s1, s2)
    jw = jaro_winkler_sim(s1, s2)
    jac = token_jaccard(s1.split(), s2.split())
dt = time.time() - t0
print(f"Computed {n_pairs:,} comparisons in {dt:.2f}s ({n_pairs/dt:,.0f} pairs/sec).", flush=True)
