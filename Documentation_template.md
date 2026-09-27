# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Forbidden
**Team Members:** Bhumeshwari Bisen, Tushar Gupta, Roodraksh Bisen, Geetanjali Varma
**Submission Date:** 27/09/2026

---

## 1. Executive Summary
*Multi-strategy blocking + gradient-boosted pairwise classifier for cross-source entity resolution. Candidates generated via 4 blocking indexes, scored with an XGBoost model on 101 engineered features, thresholded per-entity to optimize macro F0.5. Validation Macro F0.5: 0.7758. Leaderboard score: 0.734641.*

---

## 2. Methodology

### 2.1 Problem Analysis
*Only 4.45% of true matches have identical raw names (36.50% after suffix-stripping) — main noise: abbreviations/trade names (33.8%), typos (15.9%), legal suffix diffs (15.1%).

Only 2.40% of true matches have identical raw addresses; ~3.3% of S2/S3 addresses are missing.
30.25% of S1 names are non-unique (generic/franchise names) and 10.49% of S2 addresses are duplicated (multi-tenant buildings) → name alone or address alone is unreliable; both + contradiction checks needed.

Country agreement in true matches = 100%, but test set adds France → country kept open-set, never hard-coded.

5.58% of S1 entities are true singletons (zero matches); 89.02% have 2+ matches → decision layer must support 0, 1, or many matches per entity.*

### 2.2 Solution Strategy
**Approach Type:** Blocking + Pairwise Classifier (Candidate Generation → Feature Engineering → GBDT Scoring → Entity-Level Thresholding) 

**Core Innovation:** 4-strategy union blocking (66.62% candidate recall, ~105 candidates/S1) + contradiction-aware 101-feature XGBoost model + fine-grained threshold verification (found true optimum at 0.96, not the coarse-grid 0.95).*


---

## 3. Candidate Generation (Blocking)
**Blocking keys used:**
A — Suffixless Exact Name Index (exact match on normalized, suffix-stripped name)
B — Rare Token Inverted Index (IDF-weighted name tokens)
C — Address/Postcode Index (postal code + house number)
D — Character 3-Gram Retrieval (fallback for typos/abbreviations)

**Candidate pairs generated:** 46.39M (validation), 231.77M raw pairs total (train+val); mean 105.1 candidates/S1.
Recall by strategy: A=37.45%, B=29.64%, C=29.68%, D=4.31% → Union A+B+C+D = 66.62% (selected).

**Ensuring true matches not lost:** benchmarked each strategy + ablations on pinned validation set (0 leakage); unretrieved positives tracked explicitly (never mislabeled as negative); singletons get low, bounded candidate counts (~104 avg) to protect precision.

---

## 4. Matching Model

**Features used (101 total, 9 families):**

**Name features:** exact/normalized/suffixless match, Levenshtein, Jaro-Winkler, token Jaccard, char 3-gram similarity, TF-IDF cosine, length/token diffs.
**Address features:** exact/normalized match, token overlap, Levenshtein, house-number/postal exact & conflict, locality overlap, missingness flags.
**Other:** cross-field interactions (name×address), open-set country match/mismatch, frequency/rarity features, blocking-evidence flags (which strategy hit), contradiction scores (postal/house/locality conflicts).

**Model type:** XGBoost (selected over LightGBM — 0.7754 vs 0.7735 macro F0.5); hist tree method, max_depth=8, lr=0.05, trained on 20.02M pairs (20.3% positive, 4:1 hard-negative sampling).

**Threshold selection method:** Macro F0.5 grid search on validation set — coarse grid picked 0.95, extended fine-grained grid verified true optimum at 0.96 (F0.5=0.7758, Precision=0.9688, Recall=0.6537).

---

## 5. Results & Error Analysis

**F_0.5 Score (macro):** 0.7758 (validation) / 0.734641 (leaderboard)
**Feature ablation:** removing name features → −0.0298 F0.5; removing address → −0.0201; auxiliary features (blocking/contradiction) → <0.001 impact each — name+address dominate.
**By cardinality:** singletons score highest (0.8971, well-protected by 0.96 threshold); single-match entities lowest (0.6102) — bound by blocking recall, not model quality.
**Common false positives:** high-confidence (>0.97) near-duplicate candidates picked over the actual correct match for entities with 3+ true matches.
**Common false negatives:** mostly due to blocking recall ceiling (66.62%) — heavily reformatted name+address pairs never reach the model. Within the retrieved pool, classifier recovers 96.36% of positives.

---

## 6. Conclusion
*Blocking recall (66.62%), not classifier quality (96.36% in-pool recall), is the main bottleneck. Future gains should target better/embedding-based blocking rather than further model tuning.*

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`src/eda/` — dataset audit & quality checks
`src/preprocessing/normalization.py` — name/address/country normalization
`src/evaluation/`— leakage-free split + macro-F0.5 metric
`src/blocking/` — 4 blocking indexes + union candidate generator
`src/training/` — training-pair generation + hard-negative sampling
`src/features/` — 9 feature-family modules
`src/models/` LightGBM/XGBoost training, model selection, threshold verification
`src/pipeline/run_test_inference.py` — end-to-end test inference →
`output/matching_results.tsv`
`output/candidate_pairs.tsv`
`utils/validate_submission.py` — official validator
`artifacts/models/xgboost_baseline.json` — final selected model*

### B. Additional Results
*Test predictions: 5.05M links (47.4% S2, 52.6% S3); 16.79% zero-match, 17.77% one-match, 65.45% multi-match entities.
Validator: **PASS — safe to submit** (100% predicted links verified as subset of candidates).*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
