# 4. Production-Grade Normalization Report

## 4.1 Normalization Design
To prevent information loss, the normalization system preserves raw attributes (`business_name_raw`, `business_address_raw`, `country_raw`) and generates multi-view derived representations for blocking, token matching, edit distance, and tabular ML features.

## 4.2 Name Normalization
- **Unicode Normalization:** NFKD decomposition + diacritical mark removal (e.g., `Café` $\rightarrow$ `cafe`).
- **Symbol Handling:** Ampersand expansion (`&` $\rightarrow$ `and`).
- **Punctuation & Whitespace:** Punctuation stripped, multiple spaces collapsed to single spaces.
- **Compact Name:** Whitespace-stripped representation for exact dictionary lookups.

## 4.3 Legal Suffix Handling
- **Suffix Removal:** Conservative removal of trailing legal company identifiers (`inc`, `corp`, `corporation`, `llc`, `ltd`, `limited`, `pvt ltd`, `private limited`, `gmbh`, `sarl`, `bv`).
- **Safety Rule:** Stripping applies strictly to trailing phrase suffixes, preserving internal brand words.

## 4.4 Address Normalization
- **Case & Punctuation:** Case folding to lowercase and punctuation removal.
- **Abbreviation Expansion:** Conservative mapping (`st` $\rightarrow$ `street`, `rd` $\rightarrow$ `road`, `ave` $\rightarrow$ `avenue`, `dr` $\rightarrow$ `drive`, `blvd` $\rightarrow$ `boulevard`, `ste` $\rightarrow$ `suite`, `apt` $\rightarrow$ `apartment`).

## 4.5 Numeric Extraction
- **Numeric Tokens:** Extracted list of digit tokens present in address string.
- **House Number Candidate:** Leading street/building number regex extraction.
- **Postal Code Candidate:** Country-aware & generic 5-digit / 6-digit PIN code regex extraction.

## 4.6 Country Normalization
- **Open-Set Support:** Trimmed, lowercased, unicode-normalized string. **US**, **India**, and **France** (and any new country) are processed identically without hard-coded domain restrictions.

## 4.7 Missingness Handling
- Explicit boolean missing flags (`name_missing`, `address_missing`, `country_missing`). Empty addresses (~3.3% in S2/S3) are preserved as empty strings rather than dummy text.

## 4.8 Collision Analysis
- **Total Sampled Records Analyzed:** 120,000
- **Unique Raw Names:** 116,827 (97.36% unique)
- **Unique Normalized Names:** 115,186 (95.99% unique)
- **Unique Suffixless Names:** 110,787 (92.32% unique)

### Top Collisions in Suffixless Names:

| Suffixless Name | Collision Count (Occurrences in Sample) |
|---|---|
| `physical therapy` | 20 |
| `internal medicine` | 18 |
| `juniper` | 17 |
| `redwood` | 16 |
| `ear nose and throat health` | 15 |

## 4.9 Performance Benchmark
- **Processed Records:** 120,000
- **Elapsed Time:** 3.731 seconds
- **Throughput Rate:** **32,164.53 records / second**
- **Estimated Full Dataset Normalization Time (24.2M records):** ~12.55 minutes.

## 4.10 Example Transformations

### Example 1 (`S1` - `S1-925783039`)
- **Raw Name:** `Orelee's Barbershop`
  - **Name Norm:** `orelee s barbershop`
  - **Name Compact:** `oreleesbarbershop`
  - **Name Suffixless:** `orelee s barbershop`
- **Raw Address:** `1795 Westchester Drive, High Point, NC`
  - **Address Norm:** `1795 westchester drive high point nc`
  - **Numeric Tokens:** `['1795']`
  - **House Number:** `1795`
  - **Postal Code:** ``
- **Country Norm:** `us`

### Example 2 (`S1` - `S1-377745466`)
- **Raw Name:** `B+ Retail Inc`
  - **Name Norm:** `b retail inc`
  - **Name Compact:** `bretailinc`
  - **Name Suffixless:** `b retail`
- **Raw Address:** `1712 Montebello Avenue, Phoenix, AZ`
  - **Address Norm:** `1712 montebello avenue phoenix az`
  - **Numeric Tokens:** `['1712']`
  - **House Number:** `1712`
  - **Postal Code:** ``
- **Country Norm:** `us`

### Example 3 (`S1` - `S1-309349399`)
- **Raw Name:** `Callicoat & Dailey Inc`
  - **Name Norm:** `callicoat and dailey inc`
  - **Name Compact:** `callicoatanddaileyinc`
  - **Name Suffixless:** `callicoat and dailey`
- **Raw Address:** `Charlotte, NC, 833 Reliance Street`
  - **Address Norm:** `charlotte nc 833 reliance street`
  - **Numeric Tokens:** `['833']`
  - **House Number:** ``
  - **Postal Code:** ``
- **Country Norm:** `us`

### Example 4 (`S1` - `S1-925783039`)
- **Raw Name:** `Orelee's Barbershop`
  - **Name Norm:** `orelee s barbershop`
  - **Name Compact:** `oreleesbarbershop`
  - **Name Suffixless:** `orelee s barbershop`
- **Raw Address:** `1795 Westchester Drive, High Point, NC`
  - **Address Norm:** `1795 westchester drive high point nc`
  - **Numeric Tokens:** `['1795']`
  - **House Number:** `1795`
  - **Postal Code:** ``
- **Country Norm:** `us`

### Example 5 (`S1` - `S1-773889195`)
- **Raw Name:** `Prime Money`
  - **Name Norm:** `prime money`
  - **Name Compact:** `primemoney`
  - **Name Suffixless:** `prime money`
- **Raw Address:** `17560 Ellis Road, Tahlequah, OK`
  - **Address Norm:** `17560 ellis road tahlequah ok`
  - **Numeric Tokens:** `['17560']`
  - **House Number:** `17560`
  - **Postal Code:** `17560`
- **Country Norm:** `us`

### Example 6 (`S1` - `S1-377745466`)
- **Raw Name:** `B+ Retail Inc`
  - **Name Norm:** `b retail inc`
  - **Name Compact:** `bretailinc`
  - **Name Suffixless:** `b retail`
- **Raw Address:** `1712 Montebello Avenue, Phoenix, AZ`
  - **Address Norm:** `1712 montebello avenue phoenix az`
  - **Numeric Tokens:** `['1712']`
  - **House Number:** `1712`
  - **Postal Code:** ``
- **Country Norm:** `us`

## 4.11 Risks and Safeguards
1. **Over-normalization Risk:** Suffixless equality creates large collision groups for generic franchise names (`urgent care`, `physical therapy`). Safeguard: Suffixless matching must be combined with address/postal verification and used primarily for candidate generation or feature engineering, never as sole match criteria.
2. **Address Missingness Risk:** 3.3% of S2/S3 records lack addresses. Safeguard: `address_missing` boolean flag informs ML models to rely on name similarity features when addresses are absent.
