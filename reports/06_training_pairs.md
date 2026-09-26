# 6. Training Pair Generation & Hard Negative Mining Report

## 6.1 Candidate Pool
- **Total Raw Candidate Pairs Available (Train + Val):** 231,768,654
- **Train S1 Candidates:** 185,383,440 across 1,765,459 S1 entities (Mean: 105.0 cands/S1)
- **Validation S1 Candidates:** 46,385,214 across 441,362 S1 entities (Mean: 105.1 cands/S1)

## 6.2 Positive Pair Coverage
- **Total Ground-Truth Links (Dataset Wide):** 7,638,365
- **Train Ground-Truth Links:** 6,110,412
- **Train GT Links Captured by Blocking:** 4,068,991 (66.59% recall)
- **Validation Ground-Truth Links:** 1,527,953
- **Validation GT Links Captured by Blocking:** 1,017,869 (66.62% recall)

## 6.3 Negative Sampling
- **Train Potential Candidates:** 185,383,440
- **Train Sampled Negatives:** 15,950,690
- **Target Negative Ratio:** 4.0 negatives per positive
- **Effective Negative Ratio:** 3.92 negatives per positive

## 6.4 Hard Negative Categories

| Category | Count | Percentage |
|---|---|---|
| `high_name_and_address_similarity` | 1,204 | 0.01% |
| `name_conflict` | 5,031,594 | 31.54% |
| `address_conflict` | 1,917,198 | 12.02% |
| `high_name_similarity` | 8,876,976 | 55.65% |
| `high_address_similarity` | 0 | 0.00% |
| `easy_negative` | 123,718 | 0.78% |

## 6.5 Training Pair Distribution
- **Total Selected Training Pairs:** 20,019,681
- **Positive Pairs (Label=1):** 4,068,991 (20.32%)
- **Negative Pairs (Label=0):** 15,950,690 (79.68%)
- **Train S1->S2 Positives:** 1,958,246
- **Train S1->S3 Positives:** 2,110,745

## 6.6 Validation Pair Distribution
- **Total Pinned Validation Pairs:** 46,385,214
- **Validation Positives (Label=1):** 1,017,869 (2.19%)
- **Validation Negatives (Label=0):** 45,367,345 (97.81%)
- **Val S1->S2 Positives:** 489,564
- **Val S1->S3 Positives:** 528,305

## 6.7 S1 Leakage Check
- **Overlap between Train S1 and Val S1:** 0
- **Duplicate Pairs in Train:** 0
- **Duplicate Pairs in Val:** 0
- **Status:** **PASSED — 0 LEAKAGE / 0 DUPLICATES**

## 6.8 Source Distribution
- **Train S2 Candidate Pairs:** 9,693,628
- **Train S3 Candidate Pairs:** 10,326,053
- **Validation S2 Candidate Pairs:** 22,753,221
- **Validation S3 Candidate Pairs:** 23,631,993

## 6.9 Unretrieved Positive Analysis
- **Train Ground-Truth Links Missed by Blocking:** 2,041,421
- **Validation Ground-Truth Links Missed by Blocking:** 510,084
- **Critical Safety Rule Verified:** Unretrieved positives are **NOT** labeled negative. They remain explicitly classified as unretrieved.

## 6.10 Memory and Runtime
- **Total Execution Time:** 1737.44 seconds (28.96 minutes).
- **Output Format:** Efficient compressed Parquet (`train_pairs.parquet` & `validation_pairs.parquet`).

## 6.11 Training Dataset Recommendation
The generated training pair dataset combines all candidate-covered true positives with a hard-negative sample prioritizing name conflicts, address conflicts, and high similarity non-matches. This provides optimal discriminative signal for downstream Gradient Boosted Decision Trees (GBDTs).
