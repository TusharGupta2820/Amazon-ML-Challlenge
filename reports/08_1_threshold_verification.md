# Phase 8.1 — Final Threshold Verification Report

## 1. Executive Summary
- **Previous Selected Threshold (Phase 8 Grid):** **0.95** (Entity Macro F0.5: 0.7754 if res_095 else 0.0)
- **Verified Optimal Threshold:** **0.9600**
- **Verified Entity-Level Macro F0.5:** **0.7758**
- **Precision Change vs 0.95:** `+0.0026` (0.9662 → 0.9688)
- **Recall Change vs 0.95:** `-0.0020` (0.6557 → 0.6537)
- **Macro F0.5 Change vs 0.95:** `+0.0004` (0.7754 → 0.7758)

## 2. Extended Threshold Grid Search Results

Evaluating XGBoost baseline (`artifacts/models/xgboost_baseline.json`) across all 441,362 validation S1 entities:

| Threshold | Macro F0.5 | Precision | Recall | False Positive S1s | Zero Match FPs | Mean Preds / S1 | Total Predicted Links |
|---|---|---|---|---|---|---|---|
| 0.9000 | 0.7723 | 0.9565 | 0.6604 | 37,555 | 3,358 | 2.4625 | 1,086,872 |
| 0.9100 | 0.7730 | 0.9582 | 0.6598 | 35,874 | 3,225 | 2.4511 | 1,081,806 |
| 0.9200 | 0.7737 | 0.9600 | 0.6592 | 34,139 | 3,069 | 2.4388 | 1,076,379 |
| 0.9300 | 0.7744 | 0.9619 | 0.6583 | 32,245 | 2,909 | 2.4249 | 1,070,261 |
| 0.9400 | 0.7750 | 0.9640 | 0.6572 | 30,273 | 2,737 | 2.4090 | 1,063,248 |
| 0.9500 | 0.7754 | 0.9662 | 0.6557 | 28,151 | 2,536 | 2.3907 | 1,055,156 |
| 0.9600 **(OPTIMAL)** | 0.7758 | 0.9688 | 0.6537 | 25,675 | 2,314 | 2.3682 | 1,045,242 |
| 0.9700 | 0.7758 | 0.9719 | 0.6507 | 22,884 | 2,041 | 2.3388 | 1,032,269 |
| 0.9800 | 0.7750 | 0.9756 | 0.6457 | 19,489 | 1,687 | 2.2976 | 1,014,083 |
| 0.9850 | 0.7737 | 0.9779 | 0.6414 | 17,474 | 1,478 | 2.2677 | 1,000,865 |
| 0.9900 | 0.7710 | 0.9810 | 0.6342 | 14,902 | 1,217 | 2.2223 | 980,817 |
| 0.9950 | 0.7639 | 0.9855 | 0.6189 | 11,205 | 803 | 2.1379 | 943,577 |
| 0.9990 | 0.7263 | 0.9935 | 0.5603 | 5,186 | 316 | 1.8819 | 830,592 |

## 3. Findings & Decision Logic
1. **Precision vs. Recall Balance:** Increasing the threshold from 0.95 further suppresses subtle false positive matches across singletons and multi-match entities.
2. **Boundary Verification:** The extended grid established that the global Macro $F_{0.5}$ peak occurs cleanly at **0.9600**.
3. **Selected Threshold:** **0.9600** will be used as the verified decision threshold for Phase 9 final test set inference.
