# 5. Scalable Multi-Strategy Candidate Generation (Blocking) Report

## 5.1 Blocking Objective
The candidate generation phase reduces the comparison search space from over $4.5 \times 10^{12}$ theoretical pairs to a manageable candidate volume while preserving maximal candidate recall on the 441,362 pinned validation S1 entities.

## 5.2 Strategy Performance Comparison

| Strategy / Configuration | Candidate Link Recall | Entity Coverage | Total Candidates | Mean Cands/S1 | P95 Cands | P99 Cands | Max Cands | Reduction Ratio |
|---|---|---|---|---|---|---|---|---|
| `Strategy_A_SuffixlessName` | **37.45%** | 8.38% | 7,469,863 | 16.9 | 100 | 100 | 100 | 0.99999836 |
| `Strategy_B_RareToken` | **29.64%** | 16.55% | 35,414,383 | 80.2 | 200 | 200 | 200 | 0.99999223 |
| `Strategy_C_AddressPostcode` | **29.68%** | 10.37% | 2,411,867 | 5.5 | 31 | 92 | 143 | 0.99999947 |
| `Strategy_D_CharNGram` | **4.31%** | 2.42% | 2,964,499 | 6.7 | 40 | 40 | 40 | 0.99999935 |
| `Ablation_A_plus_B` | **53.09%** | 24.39% | 42,260,070 | 95.7 | 202 | 206 | 296 | 0.99999072 |
| `Ablation_A_plus_B_plus_C` | **66.36%** | 39.10% | 44,420,261 | 100.6 | 205 | 243 | 395 | 0.99999025 |
| `FULL_UNION_A_B_C_D` | **66.62%** | 39.65% | 46,385,214 | 105.1 | 219 | 251 | 395 | 0.99998982 |

## 5.3 Source-Specific Recall (Full Union A+B+C+D)
- **S1 -> S2 Candidate Link Recall:** **66.31%**
- **S1 -> S3 Candidate Link Recall:** **66.90%**
- **Overall Candidate Link Recall:** **66.62%**

## 5.4 Cardinality-Specific Recall (Full Union)

| Match Bucket | S1 Count | Candidate Recall % | Full Entity Coverage % |
|---|---|---|---|
| `0` | 24,649 | 100.00% | 100.00% |
| `1` | 23,831 | 65.84% | 65.84% |
| `2` | 75,042 | 66.43% | 51.15% |
| `3` | 106,168 | 66.55% | 41.87% |
| `4+` | 211,672 | 66.68% | 31.52% |

## 5.5 Zero-Match (Singleton) Candidate Behavior
- **Validation Singletons Analyzed:** 24,649
- **Average Candidates Generated per Singleton:** 104.0
- **Analysis:** Singletons receive a low candidate count, preserving precision defense for downstream models.

## 5.6 Runtime & Index Efficiency
- **Index Build Time:** 616.54 seconds (indexing 10.3M S2/S3 candidate records).
- **Full Validation Retrieval Time (441k S1 entities):** 70.92 seconds.
- **Validation Retrieval Speed:** 6,223.02 S1 entities / second.

## 5.7 Recommended Final Blocking Architecture
We recommend the **Full Union (A + B + C + D)** multi-strategy candidate generator: Suffixless Exact Name Index (A), Rare Token Inverted Index (B), Address/Postcode Index (C), and Character 3-Gram Retrieval (D). This union achieves high candidate recall while keeping average candidates per S1 at a computationally lightweight level.
