# 3. Leakage-Free Validation Strategy Report

## 3.1 Why Pair-Level Splitting Is Unsafe
Business Entity Resolution evaluates performance per **Source 1 Entity** across all its potential matches. Randomly splitting candidate pairs at the pair level allows candidate pairs belonging to the *same* S1 entity to appear in both training and validation sets. This creates severe data leakage, inflating validation scores and distorting threshold selection. Therefore, our validation framework enforces strict **Grouping by `source1_entity_id`**, ensuring that any given S1 entity and all its candidate pairs belong entirely to either TRAIN or VALIDATION, never both.

## 3.2 Grouped S1 Split
- **Seed:** 42
- **Train Fraction:** 80.0%
- **Validation Fraction:** 20.0%
- **Grouping Column:** `source1_entity_id`
- **Stratification:** Match cardinality buckets (`0`, `1`, `2`, `3`, `4+` matches)
- **Data Leakage Check:** 0 overlapping S1 IDs (100% clean split).

## 3.3 Train / Validation Sizes
- **Train S1 Entities:** 1,765,459 (80.00%)
- **Validation S1 Entities:** 441,362 (20.00%)
- **Total S1 Entities:** 2,206,821

## 3.4 Match Cardinality Distribution

| Cardinality Bucket | Train S1 Count | Train % | Val S1 Count | Val % |
|---|---|---|---|---|
| `0` | 98,598 | 5.58% | 24,649 | 5.58% |
| `1` | 95,326 | 5.40% | 23,831 | 5.40% |
| `2` | 300,170 | 17.00% | 75,042 | 17.00% |
| `3` | 424,673 | 24.05% | 106,168 | 24.05% |
| `4+` | 846,692 | 47.96% | 211,672 | 47.96% |

## 3.5 S2 / S3 Match Distribution

| Source Category | Train Ground-Truth Links | Val Ground-Truth Links |
|---|---|---|
| Total S2 Links | 2,955,351 | 738,268 |
| Total S3 Links | 3,155,061 | 789,685 |
| Total Ground-Truth Links | 6,110,412 | 1,527,953 |
| S1 Matching ONLY S2 | 114,424 | 28,605 |
| S1 Matching ONLY S3 | 131,557 | 32,941 |
| S1 Matching BOTH S2 and S3 | 1,420,880 | 355,167 |
| S1 Zero Matches (Singletons) | 98,598 | 24,649 |

## 3.6 Official Metric Definition (Macro F0.5)
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

Computed per S1 entity and averaged macro across ALL S1 entities in the evaluation set.

## 3.7 Singleton Handling
- For a true zero-match entity ($T = \emptyset$):
  - Predicted $P = \emptyset \implies F_{0.5} = 1.0$
  - Predicted $P \neq \emptyset \implies F_{0.5} = 0.0$

## 3.8 Metric Sanity Test Results
- Example A (True [S2-1], Pred [S2-1]): P=1.0, R=1.0, F0.5=1.0 **[PASSED]**
- Example B (True [S2-1], Pred []): R=0.0, F0.5=0.0 **[PASSED]**
- Example C (True [], Pred []): F0.5=1.0 **[PASSED]**
- Example D (True [], Pred [S2-1]): F0.5=0.0 **[PASSED]**
- Example E (True [S2-1, S2-2], Pred [S2-1]): P=1.0, R=0.5, F0.5=0.8333 **[PASSED]**
- Example F (True [S2-1], Pred [S2-1, S2-2]): P=0.5, R=1.0, F0.5=0.5556 **[PASSED]**

## 3.9 Reproducibility & Persistence
- `configs/validation_config.json`: Seed 42, 80/20 split configuration.
- `reports/validation_s1_ids.txt`: Pinned list of 441,364 S1 validation IDs.
- `reports/train_s1_ids.txt`: Pinned list of 1,765,457 S1 training IDs.

