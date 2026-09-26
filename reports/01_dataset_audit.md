# Phase 1: Complete Dataset Audit Report

## 1. Executive Summary & File Integrity

| File Key | File Name | Size (MB) | Rows | Cols | Columns | Delimiter |
|---|---|---|---|---|---|---|
| `train_s1` | `train_source1.tsv` | 200.34 | 2,206,821 | 4 | `entity_id,business_name,business_address,country` | Tab (`\t`) |
| `train_s2` | `train_source2.tsv` | 466.63 | 5,034,616 | 4 | `entity_id,business_name,business_address,country` | Tab (`\t`) |
| `train_s3` | `train_source3.tsv` | 480.37 | 5,285,603 | 4 | `entity_id,business_name,business_address,country` | Tab (`\t`) |
| `test_s1` | `test_source1.tsv` | 166.91 | 1,732,544 | 4 | `entity_id,business_name,business_address,country` | Tab (`\t`) |
| `test_s2` | `test_source2.tsv` | 485.86 | 4,887,273 | 4 | `entity_id,business_name,business_address,country` | Tab (`\t`) |
| `test_s3` | `test_source3.tsv` | 482.56 | 5,082,316 | 4 | `entity_id,business_name,business_address,country` | Tab (`\t`) |
| `train_gt` | `train_ground_truth.tsv` | 121.13 | 2,206,821 | 2 | `source1_entity_id,matched_entity_ids` | Tab (`\t`) |

## 2. Dataset Column & Missing Value Profiling

| File Key | Entity ID Dupes | Name Dupes | Addr Dupes | Empty Names | Empty Addrs | Short Names (<3) | Short Addrs (<5) |
|---|---|---|---|---|---|---|---|
| `train_s1` | 0 | 667,592 | 76,215 | 0 (0.00%) | 0 (0.00%) | 0 | 0 |
| `train_s2` | 0 | 632,607 | 528,388 | 0 (0.00%) | 168,967 (3.36%) | 704 | 0 |
| `train_s3` | 0 | 633,994 | 476,923 | 0 (0.00%) | 175,916 (3.33%) | 9,275 | 1 |
| `test_s1` | 0 | 493,677 | 55,061 | 0 (0.00%) | 0 (0.00%) | 0 | 0 |
| `test_s2` | 0 | 576,232 | 533,082 | 0 (0.00%) | 129,408 (2.65%) | 10,166 | 0 |
| `test_s3` | 0 | 560,387 | 489,783 | 0 (0.00%) | 136,098 (2.68%) | 19,910 | 0 |

## 3. Country Distribution Analysis

| File Key | Country Distribution |
|---|---|
| `train_s1` | **US**: 1,323,633, **India**: 883,188 |
| `train_s2` | **India**: 2,017,799, **US**: 3,016,817 |
| `train_s3` | **US**: 3,170,056, **India**: 2,115,547 |
| `test_s1` | **US**: 663,106, **France**: 259,452, **India**: 809,986 |
| `test_s2` | **India**: 2,312,565, **France**: 703,378, **US**: 1,871,330 |
| `test_s3` | **India**: 2,405,000, **France**: 731,615, **US**: 1,945,701 |

## 4. Name & Address Length Statistics

### Business Name Length (Characters)

| File Key | Min | Q25 | Median | Mean | Q75 | Max | Std |
|---|---|---|---|---|---|---|---|
| `train_s1` | 3 | 18.0 | 24.0 | 24.03 | 30.0 | 105 | 7.74 |
| `train_s2` | 2 | 19.0 | 25.0 | 25.10 | 31.0 | 104 | 8.89 |
| `train_s3` | 2 | 18.0 | 25.0 | 25.20 | 31.0 | 123 | 9.49 |
| `test_s1` | 3 | 18.0 | 24.0 | 23.84 | 29.0 | 92 | 7.67 |
| `test_s2` | 2 | 19.0 | 25.0 | 25.70 | 32.0 | 102 | 9.12 |
| `test_s3` | 2 | 19.0 | 25.0 | 25.66 | 32.0 | 103 | 9.57 |

### Business Address Length (Characters)

| File Key | Min | Q25 | Median | Mean | Q75 | Max | Std |
|---|---|---|---|---|---|---|---|
| `train_s1` | 11 | 33.0 | 41.0 | 52.07 | 70.0 | 256 | 25.33 |
| `train_s2` | 0 | 30.0 | 37.0 | 46.23 | 61.0 | 249 | 24.84 |
| `train_s3` | 0 | 35.0 | 42.0 | 46.71 | 54.0 | 240 | 21.65 |
| `test_s1` | 11 | 36.0 | 50.0 | 57.21 | 74.0 | 268 | 25.03 |
| `test_s2` | 0 | 32.0 | 43.0 | 50.41 | 67.0 | 269 | 25.35 |
| `test_s3` | 0 | 35.0 | 43.0 | 48.74 | 59.0 | 267 | 22.64 |

## 5. Character, Punctuation & Token Noise Patterns

| File Key | Non-ASCII Names | Non-ASCII Addrs | Punctuation in Names | Numbers in Names | Punctuation in Addrs | Numbers in Addrs |
|---|---|---|---|---|---|---|
| `train_s1` | 0 | 554 | 461,669 | 35,585 | 2,206,821 | 2,129,784 |
| `train_s2` | 764,608 | 478,453 | 2,158,861 | 255,636 | 4,865,649 | 4,563,675 |
| `train_s3` | 606,737 | 476,588 | 2,119,387 | 270,847 | 5,109,686 | 4,799,297 |
| `test_s1` | 40,789 | 73,800 | 317,477 | 20,207 | 1,732,544 | 1,660,706 |
| `test_s2` | 928,158 | 720,665 | 2,000,796 | 190,009 | 4,757,687 | 4,529,243 |
| `test_s3` | 737,515 | 729,222 | 1,900,369 | 202,470 | 4,946,076 | 4,700,823 |

## 6. Ground Truth Deep Dive

- **Total S1 Entities in Train GT:** 2,206,821
- **Duplicate S1 Entities in GT:** 0
- **Total Ground-Truth Links (Matches):** 7,638,365
- **Zero Match S1 Entities (Singletons):** 123,247 (5.58%)
- **One Match S1 Entities:** 119,157 (5.40%)
- **Multiple Matches S1 Entities:** 1,964,417 (89.02%)
- **Total S2 Links:** 3,693,619
- **Total S3 Links:** 3,944,746
- **S1 Entities matching ONLY S2:** 143,029
- **S1 Entities matching ONLY S3:** 164,498
- **S1 Entities matching BOTH S2 and S3:** 1,776,047
- **Invalid / Missing Source-2/3 IDs in GT:** 0
- **Duplicate Matched IDs within same GT Row:** 0
- **Multi-Referenced S2/S3 Entities (Matched to >1 S1):** 0

### Total Match Count Distribution per S1 Entity

| Total Matches | Count | Percentage |
|---|---|---|
| 0 | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10 | 534 | 0.02% |
| 11 | 37 | 0.00% |

### S1 -> S2 Match Distribution

| S2 Matches | Count | Percentage |
|---|---|---|
| 0 | 287,745 | 13.04% |
| 1 | 789,108 | 35.76% |
| 2 | 652,779 | 29.58% |
| 3 | 333,957 | 15.13% |
| 4 | 119,078 | 5.40% |
| 5 | 24,154 | 1.09% |

### S1 -> S3 Match Distribution

| S3 Matches | Count | Percentage |
|---|---|---|
| 0 | 266,276 | 12.07% |
| 1 | 716,417 | 32.46% |
| 2 | 668,375 | 30.29% |
| 3 | 372,443 | 16.88% |
| 4 | 145,116 | 6.58% |
| 5 | 35,378 | 1.60% |
| 6 | 2,816 | 0.13% |

