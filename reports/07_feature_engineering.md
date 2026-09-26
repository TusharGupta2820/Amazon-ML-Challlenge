# 7. Production-Grade Feature Engineering Report

## 7.1 Objective
Build a production-grade, zero-leakage pairwise feature-engineering pipeline across 9 feature families for both training candidate pairs (~20.02M) and validation candidate pairs (~46.39M).

## 7.2 Input Artifacts
- `artifacts/training_pairs/train_pairs.parquet` (20,019,681 rows)
- `artifacts/training_pairs/validation_pairs.parquet` (46,385,214 rows)
- `dataset/train/train_source1.tsv` (2,206,821 S1 entities)
- `dataset/train/train_source2.tsv` (5,034,616 S2 entities)
- `dataset/train/train_source3.tsv` (5,285,603 S3 entities)

## 7.3 Feature Architecture & Families (56 Total Features)
1. **Name Similarity (A1-A20):** Exact raw/norm/suffixless/compact, Levenshtein, Jaro-Winkler, Token Jaccard, Token overlap S1<->cand, Char 3-gram Jaccard/Cosine, TF-IDF cosine approximation, length diff/ratio, token count diff, first/last token match, numeric token overlap, shared rare tokens.
2. **Address Similarity (B1-B27):** Exact raw/norm, Token Jaccard, Token overlap S1<->cand, Levenshtein, Jaro-Winkler, Char 3-gram Jaccard/Cosine, TF-IDF cosine, length diff/ratio, house number exact/conflict, postal exact/conflict/prefix match, locality overlap/conflict, missingness flags.
3. **Cross-Field Interactions (C1-C12):** Name x Address Jaccard/TF-IDF, Name x Postal/House/Locality match, High-Name/Low-Address, Address-High/Name-Low, Strong Name & Strong Address, Strong Name + Address conflict, Suffixless exact + Address conflict.
4. **Open-Set Country (D1-D6):** Country exact match, country mismatch, missing S1, missing candidate, missing either, both present (fully supports France and unseen test countries).
5. **Frequency & Rarity (E1-E16):** Unsupervised S1/candidate name, suffixless name, address, postal code frequencies; entity rarity (IDF); candidate group sizes; shared token & rarest token rarity.
6. **Blocking Evidence (F1-F9):** Binary hits for Strategy A (suffixless name), B (rare token), C (address/postcode), D (char 3-gram), strategy hit count, name evidence, address evidence, multi-strategy hit indicator, candidate rank.
7. **Source-Aware Features (G1-G9):** Candidate source indicators (`candidate_is_s2`, `candidate_is_s3`), missingness flags, source-specific string lengths.
8. **Contradiction Features (H1-H10):** Postal conflict, house number conflict, locality conflict, numeric address conflict, strong name + postal/house/locality conflict, contradiction score, overall contradiction count.
9. **Explicit Missingness Indicators:** Name, address, country, postal, house number, locality missingness.

## 7.4 Feature Matrix Summary
- **Train Feature Rows:** 20,019,681
- **Validation Feature Rows:** 46,385,214
- **Total Feature Columns:** 105
- **Train Output File:** `artifacts/features/train_features.parquet`
- **Validation Output File:** `artifacts/features/validation_features.parquet`
- **Schema Match (Train vs Validation):** **True**

## 7.5 Label Leakage Audit
- **Status:** **PASSED — 0 LEAKAGE DETECTED**
- **Ground Truth Reads for Features:** 0
- **Validation Label Access for Features:** 0
- **External Data/APIs Used:** 0
- **Report Location:** `reports/feature_leakage_audit.json`

## 7.6 Runtime & Performance Benchmark
- **Train Pair Computation Time:** 6438.30 seconds (3,109 pairs/sec)
- **Validation Pair Computation Time:** 8423.83 seconds (5,506 pairs/sec)
- **Total Pipeline Execution Time:** 14862.13 seconds (247.70 minutes)
- **Overall Throughput:** 4,468 pairs/sec
- **Peak Memory Usage:** < 1.2 GB RAM (chunked PyArrow Parquet streaming)

## 7.7 Unit Tests
- **Test File:** `tests/test_features.py`
- **Total Unit Tests Executed:** 16
- **Status:** **16/16 PASSED (100% Success)**

## 7.8 Next Steps Recommendation
Phase 7 is complete and verified. The feature matrix is ready for Phase 8 (Model Selection & GBDT Matcher Training).
