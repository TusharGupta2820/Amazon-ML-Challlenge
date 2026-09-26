import os
import sys
import time
import json
import unittest
from collections import Counter

# Add paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(base_dir, "src", "preprocessing"))

from normalization import BusinessNormalizer

def run_tests():
    print("Running normalization unit tests...", flush=True)
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(base_dir, "tests"), pattern="test_normalization.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Unit tests failed!")
    print("All unit tests passed successfully!", flush=True)

def analyze_normalization():
    print("Starting Normalization Analysis & Benchmarking...", flush=True)
    normalizer = BusinessNormalizer()
    
    source_files = {
        "train_s1": os.path.join(base_dir, "dataset", "train", "train_source1.tsv"),
        "train_s2": os.path.join(base_dir, "dataset", "train", "train_source2.tsv"),
        "train_s3": os.path.join(base_dir, "dataset", "train", "train_source3.tsv"),
        "test_s1": os.path.join(base_dir, "dataset", "test", "test_source1.tsv"),
        "test_s2": os.path.join(base_dir, "dataset", "test", "test_source2.tsv"),
        "test_s3": os.path.join(base_dir, "dataset", "test", "test_source3.tsv"),
    }
    
    sample_records = []
    # Take 20,000 records from each file -> 120,000 records total sample
    for key, path in source_files.items():
        if not os.path.isfile(path):
            continue
        count = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\r\n").split("\t")
                while len(parts) < 4:
                    parts.append("")
                sample_records.append({
                    "entity_id": parts[0].strip(),
                    "business_name": parts[1].strip(),
                    "business_address": parts[2].strip(),
                    "country": parts[3].strip(),
                    "file_key": key
                })
                count += 1
                if count >= 20000:
                    break

    print(f"Collected sample of {len(sample_records):,} records across 6 files.", flush=True)
    
    # Throughput Benchmark
    t0 = time.time()
    enriched_records = []
    for rec in sample_records:
        enriched_records.append(normalizer.normalize_record(rec))
    t1 = time.time()
    
    elapsed = t1 - t0
    throughput = len(sample_records) / elapsed if elapsed > 0 else 0
    print(f"Benchmark: Normalized {len(sample_records):,} records in {elapsed:.3f}s ({throughput:,.2f} records/sec).", flush=True)

    # Collision & Uniqueness Analysis
    raw_names = set()
    norm_names = set()
    suffixless_names = set()
    
    norm_name_counts = Counter()
    suffixless_counts = Counter()
    
    for rec in enriched_records:
        rn = rec["business_name_raw"]
        nn = rec["business_name_norm"]
        sn = rec["business_name_suffixless"]
        
        if rn:
            raw_names.add(rn)
        if nn:
            norm_names.add(nn)
            norm_name_counts[nn] += 1
        if sn:
            suffixless_names.add(sn)
            suffixless_counts[sn] += 1

    total_names = len(enriched_records)
    n_raw_unique = len(raw_names)
    n_norm_unique = len(norm_names)
    n_suffixless_unique = len(suffixless_names)

    top_norm_collisions = norm_name_counts.most_common(5)
    top_suffixless_collisions = suffixless_counts.most_common(5)

    # Select representative transformation examples
    examples = []
    seen_types = set()
    for rec in sample_records[:500]:
        rn = rec["business_name"]
        ra = rec["business_address"]
        cty = rec["country"]
        
        enriched = normalizer.normalize_record(rec)
        
        ex_type = None
        if "café" in rn.lower() or "é" in rn or "è" in rn or "ç" in rn:
            ex_type = "unicode"
        elif "&" in rn:
            ex_type = "ampersand"
        elif any(s in rn.lower().split() for s in ["inc", "corp", "llc", "ltd"]):
            ex_type = "legal_suffix"
        elif "st." in ra.lower() or "rd." in ra.lower() or "ste" in ra.lower():
            ex_type = "abbreviation"
        elif not ra:
            ex_type = "missing_address"
            
        if ex_type and ex_type not in seen_types:
            seen_types.add(ex_type)
            examples.append(enriched)
            if len(examples) >= 6:
                break
                
    if len(examples) < 6:
        for rec in sample_records[:6]:
            if len(examples) >= 6:
                break
            examples.append(normalizer.normalize_record(rec))

    # Generate Markdown Report reports/04_normalization.md
    os.makedirs(os.path.join(base_dir, "reports"), exist_ok=True)
    report_path = os.path.join(base_dir, "reports", "04_normalization.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 4. Production-Grade Normalization Report\n\n")
        f.write("## 4.1 Normalization Design\n")
        f.write("To prevent information loss, the normalization system preserves raw attributes (`business_name_raw`, `business_address_raw`, `country_raw`) ")
        f.write("and generates multi-view derived representations for blocking, token matching, edit distance, and tabular ML features.\n\n")

        f.write("## 4.2 Name Normalization\n")
        f.write("- **Unicode Normalization:** NFKD decomposition + diacritical mark removal (e.g., `Café` $\\rightarrow$ `cafe`).\n")
        f.write("- **Symbol Handling:** Ampersand expansion (`&` $\\rightarrow$ `and`).\n")
        f.write("- **Punctuation & Whitespace:** Punctuation stripped, multiple spaces collapsed to single spaces.\n")
        f.write("- **Compact Name:** Whitespace-stripped representation for exact dictionary lookups.\n\n")

        f.write("## 4.3 Legal Suffix Handling\n")
        f.write("- **Suffix Removal:** Conservative removal of trailing legal company identifiers (`inc`, `corp`, `corporation`, `llc`, `ltd`, `limited`, `pvt ltd`, `private limited`, `gmbh`, `sarl`, `bv`).\n")
        f.write("- **Safety Rule:** Stripping applies strictly to trailing phrase suffixes, preserving internal brand words.\n\n")

        f.write("## 4.4 Address Normalization\n")
        f.write("- **Case & Punctuation:** Case folding to lowercase and punctuation removal.\n")
        f.write("- **Abbreviation Expansion:** Conservative mapping (`st` $\\rightarrow$ `street`, `rd` $\\rightarrow$ `road`, `ave` $\\rightarrow$ `avenue`, `dr` $\\rightarrow$ `drive`, `blvd` $\\rightarrow$ `boulevard`, `ste` $\\rightarrow$ `suite`, `apt` $\\rightarrow$ `apartment`).\n\n")

        f.write("## 4.5 Numeric Extraction\n")
        f.write("- **Numeric Tokens:** Extracted list of digit tokens present in address string.\n")
        f.write("- **House Number Candidate:** Leading street/building number regex extraction.\n")
        f.write("- **Postal Code Candidate:** Country-aware & generic 5-digit / 6-digit PIN code regex extraction.\n\n")

        f.write("## 4.6 Country Normalization\n")
        f.write("- **Open-Set Support:** Trimmed, lowercased, unicode-normalized string. **US**, **India**, and **France** (and any new country) are processed identically without hard-coded domain restrictions.\n\n")

        f.write("## 4.7 Missingness Handling\n")
        f.write("- Explicit boolean missing flags (`name_missing`, `address_missing`, `country_missing`). Empty addresses (~3.3% in S2/S3) are preserved as empty strings rather than dummy text.\n\n")

        f.write("## 4.8 Collision Analysis\n")
        f.write(f"- **Total Sampled Records Analyzed:** {total_names:,}\n")
        f.write(f"- **Unique Raw Names:** {n_raw_unique:,} ({n_raw_unique/total_names*100:.2f}% unique)\n")
        f.write(f"- **Unique Normalized Names:** {n_norm_unique:,} ({n_norm_unique/total_names*100:.2f}% unique)\n")
        f.write(f"- **Unique Suffixless Names:** {n_suffixless_unique:,} ({n_suffixless_unique/total_names*100:.2f}% unique)\n\n")
        f.write("### Top Collisions in Suffixless Names:\n\n")
        f.write("| Suffixless Name | Collision Count (Occurrences in Sample) |\n")
        f.write("|---|---|\n")
        for sname, cnt in top_suffixless_collisions:
            f.write(f"| `{sname}` | {cnt:,} |\n")
        f.write("\n")

        f.write("## 4.9 Performance Benchmark\n")
        f.write(f"- **Processed Records:** {len(sample_records):,}\n")
        f.write(f"- **Elapsed Time:** {elapsed:.3f} seconds\n")
        f.write(f"- **Throughput Rate:** **{throughput:,.2f} records / second**\n")
        f.write(f"- **Estimated Full Dataset Normalization Time (24.2M records):** ~{24229173 / throughput / 60:.2f} minutes.\n\n")

        f.write("## 4.10 Example Transformations\n\n")
        for i, ex in enumerate(examples, 1):
            f.write(f"### Example {i} (`{ex['source']}` - `{ex['entity_id']}`)\n")
            f.write(f"- **Raw Name:** `{ex['business_name_raw']}`\n")
            f.write(f"  - **Name Norm:** `{ex['business_name_norm']}`\n")
            f.write(f"  - **Name Compact:** `{ex['business_name_compact']}`\n")
            f.write(f"  - **Name Suffixless:** `{ex['business_name_suffixless']}`\n")
            f.write(f"- **Raw Address:** `{ex['business_address_raw']}`\n")
            f.write(f"  - **Address Norm:** `{ex['business_address_norm']}`\n")
            f.write(f"  - **Numeric Tokens:** `{ex['business_address_numeric_tokens']}`\n")
            f.write(f"  - **House Number:** `{ex['house_number_candidate']}`\n")
            f.write(f"  - **Postal Code:** `{ex['postal_code_candidate']}`\n")
            f.write(f"- **Country Norm:** `{ex['country_norm']}`\n\n")

        f.write("## 4.11 Risks and Safeguards\n")
        f.write("1. **Over-normalization Risk:** Suffixless equality creates large collision groups for generic franchise names (`urgent care`, `physical therapy`). ")
        f.write("Safeguard: Suffixless matching must be combined with address/postal verification and used primarily for candidate generation or feature engineering, never as sole match criteria.\n")
        f.write("2. **Address Missingness Risk:** 3.3% of S2/S3 records lack addresses. Safeguard: `address_missing` boolean flag informs ML models to rely on name similarity features when addresses are absent.\n")

    print(f"Saved normalization report to {report_path}", flush=True)

if __name__ == "__main__":
    run_tests()
    analyze_normalization()
