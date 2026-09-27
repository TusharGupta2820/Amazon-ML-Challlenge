# 9. Final Submission Audit Report

## Model Architecture
- **Model:** XGBoost (`artifacts/models/xgboost_baseline.json`)
- **Verified Threshold:** **0.96**
- **Validation Macro F0.5:** **0.7758** (Validation Precision: **0.9688**)

## Test Candidate Set Statistics
- **Test S1 Entity Count:** **1,732,544**
- **Test Candidate Pairs Generated:** **179,300,587**

## Test Prediction Distribution
- **Total Predicted Links:** **5,045,540**
- **S2 Predicted Links:** 2,390,773 (47.38%)
- **S3 Predicted Links:** 2,654,767 (52.62%)
- **Zero-Match S1 Count:** **290,883** (16.79%)
- **One-Match S1 Count:** **307,789** (17.77%)
- **Multi-Match S1 Count:** **1,133,872** (65.45%)
- **Mean Predictions / S1:** **2.9122**

## Candidate / Prediction Consistency
- **Predicted Links ⊆ Candidates:** **VERIFIED** (100% of predicted matches present in `candidate_pairs.tsv`)
- **S1 Count Alignment:** **VERIFIED** (All 1,732,544 test S1 entities present exactly once in `matching_results.tsv` and `candidate_pairs.tsv`)

## Official Validator Result
- **Status:** **PASS — Safe to Submit**
- **Validator Command:** `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids`

```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (290883 empty, 1441661 non-empty).
  candidate_pairs.tsv: 1732544 rows (24417 empty, 1708127 non-empty).

PASS — no blocking issues found. Safe to submit.
```

## System Performance & Files
- **Total Runtime:** 20061.69 seconds
- **matching_results.tsv Size:** 87.65 MB
- **candidate_pairs.tsv Size:** 2333.40 MB

## Conclusion
Phase 9 test inference and submission file generation is complete and verified by the official competition validator.
