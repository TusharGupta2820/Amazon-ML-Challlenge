import os
import sys
import re
from collections import Counter
import math

LEGAL_TOKENS = {'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private', 'co', 'company', 'gmbh', 'sa', 'sarl', 'bv'}
ABBREV_TOKENS = {'st', 'rd', 'ave', 'dr', 'blvd', 'ste', 'apt', 'pvt', 'ltd', 'corp', 'inc', 'co', 'dept'}

def calc_stats_from_hist(hist_counter, total_count):
    if not hist_counter or total_count == 0:
        return {'min': 0, 'max': 0, 'mean': 0.0, 'median': 0.0, 'std': 0.0, 'q25': 0.0, 'q75': 0.0}
    
    sorted_lens = sorted(hist_counter.keys())
    min_val = sorted_lens[0]
    max_val = sorted_lens[-1]
    
    total_sum = sum(l * cnt for l, cnt in hist_counter.items())
    mean_val = total_sum / total_count
    
    var_val = sum(cnt * ((l - mean_val) ** 2) for l, cnt in hist_counter.items()) / total_count
    std_val = math.sqrt(var_val)
    
    def get_percentile_val(p):
        target_rank = total_count * p
        cum = 0
        for l in sorted_lens:
            cum += hist_counter[l]
            if cum >= target_rank:
                return float(l)
        return float(sorted_lens[-1])
        
    return {
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'median': get_percentile_val(0.5),
        'std': std_val,
        'q25': get_percentile_val(0.25),
        'q75': get_percentile_val(0.75)
    }

def analyze_source_file(file_path):
    print(f"Analyzing {file_path}...", flush=True)
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    seen_ids = set()
    dup_ids = 0
    seen_name_hashes = set()
    dup_names = 0
    seen_addr_hashes = set()
    dup_addrs = 0
    
    country_counts = Counter()
    missing_counts = {'entity_id': 0, 'business_name': 0, 'business_address': 0, 'country': 0}
    
    name_len_hist = Counter()
    addr_len_hist = Counter()
    
    empty_names = 0
    empty_addrs = 0
    short_names = 0  # < 3
    short_addrs = 0  # < 5
    
    non_ascii_names = 0
    non_ascii_addrs = 0
    punct_names = 0
    punct_addrs = 0
    num_names = 0
    num_addrs = 0
    
    suffix_counts = Counter()
    abbrev_counts = Counter()
    
    row_count = 0
    headers = []
    
    all_entity_ids = set()

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            return None, set()
        headers = [h.strip() for h in header_line.rstrip("\r\n").split("\t")]
        
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            while len(parts) < 4:
                parts.append("")
                
            eid, name, addr, cty = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
            row_count += 1
            
            all_entity_ids.add(eid)
            
            if eid in seen_ids:
                dup_ids += 1
            else:
                seen_ids.add(eid)
                
            if name:
                h_name = hash(name)
                if h_name in seen_name_hashes:
                    dup_names += 1
                else:
                    seen_name_hashes.add(h_name)
            else:
                empty_names += 1
                missing_counts['business_name'] += 1
                
            if addr:
                h_addr = hash(addr)
                if h_addr in seen_addr_hashes:
                    dup_addrs += 1
                else:
                    seen_addr_hashes.add(h_addr)
            else:
                empty_addrs += 1
                missing_counts['business_address'] += 1
                
            if not eid:
                missing_counts['entity_id'] += 1
            if not cty:
                missing_counts['country'] += 1
            else:
                country_counts[cty] += 1
                
            n_len = len(name)
            a_len = len(addr)
            name_len_hist[n_len] += 1
            addr_len_hist[a_len] += 1
            
            if 0 < n_len < 3:
                short_names += 1
            if 0 < a_len < 5:
                short_addrs += 1
                
            if any(ord(c) > 127 for c in name):
                non_ascii_names += 1
            if any(ord(c) > 127 for c in addr):
                non_ascii_addrs += 1
                
            if any(not c.isalnum() and not c.isspace() for c in name):
                punct_names += 1
            if any(not c.isalnum() and not c.isspace() for c in addr):
                punct_addrs += 1
                
            if any(c.isdigit() for c in name):
                num_names += 1
            if any(c.isdigit() for c in addr):
                num_addrs += 1
                
            name_words = name.lower().split()
            if name_words:
                last_w = name_words[-1].strip(".,;:'\"()")
                if last_w in LEGAL_TOKENS:
                    suffix_counts[last_w] += 1
                    
            addr_words = addr.lower().split()
            all_words = set([w.strip(".,;:'\"()") for w in name_words + addr_words])
            for tok in all_words:
                if tok in ABBREV_TOKENS:
                    abbrev_counts[tok] += 1

    missing_pct_dict = {k: (v / row_count * 100 if row_count > 0 else 0.0) for k, v in missing_counts.items()}

    stats = {
        'file_name': os.path.basename(file_path),
        'file_path': file_path,
        'file_size_mb': file_size_mb,
        'row_count': row_count,
        'col_count': len(headers),
        'col_names': headers,
        'missing_dict': missing_counts,
        'missing_pct_dict': missing_pct_dict,
        'dup_ids': dup_ids,
        'dup_names': dup_names,
        'dup_addrs': dup_addrs,
        'country_dist': dict(country_counts),
        'name_len_stats': calc_stats_from_hist(name_len_hist, row_count),
        'addr_len_stats': calc_stats_from_hist(addr_len_hist, row_count),
        'empty_names': empty_names,
        'empty_addrs': empty_addrs,
        'short_names': short_names,
        'short_addrs': short_addrs,
        'non_ascii_names': non_ascii_names,
        'non_ascii_addrs': non_ascii_addrs,
        'punct_names': punct_names,
        'punct_addrs': punct_addrs,
        'num_names': num_names,
        'num_addrs': num_addrs,
        'suffix_counts': dict(suffix_counts.most_common(10)),
        'abbrev_counts': dict(abbrev_counts.most_common(10))
    }
    print(f"Finished {file_path}: {row_count:,} rows processed.", flush=True)
    return stats, all_entity_ids

def analyze_ground_truth(gt_path, train_s1_ids, train_s2_ids, train_s3_ids):
    print(f"Analyzing {gt_path}...", flush=True)
    file_size_bytes = os.path.getsize(gt_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    n_s1_rows = 0
    seen_s1 = set()
    dup_s1_in_gt = 0
    
    total_links = 0
    zero_matches_count = 0
    one_match_count = 0
    multi_matches_count = 0
    
    s2_links_count = 0
    s3_links_count = 0
    
    s1_matching_both = 0
    s1_only_s2 = 0
    s1_only_s3 = 0
    
    match_count_dist = Counter()
    s2_match_count_dist = Counter()
    s3_match_count_dist = Counter()
    
    invalid_ids = set()
    unexpected_source_ids = []
    duplicate_gt_matched_ids_within_row = 0
    
    matched_ids_counter = Counter()

    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            matched_str = parts[1].strip() if len(parts) > 1 else ""
            
            n_s1_rows += 1
            if s1_id in seen_s1:
                dup_s1_in_gt += 1
            else:
                seen_s1.add(s1_id)
                
            if not matched_str:
                matched_ids = []
            else:
                matched_ids = [m.strip() for m in matched_str.split(',') if m.strip()]
                
            if len(matched_ids) != len(set(matched_ids)):
                duplicate_gt_matched_ids_within_row += 1
                
            n_m = len(matched_ids)
            match_count_dist[n_m] += 1
            total_links += n_m
            
            if n_m == 0:
                zero_matches_count += 1
            elif n_m == 1:
                one_match_count += 1
            else:
                multi_matches_count += 1
                
            s2_in_row = [m for m in matched_ids if m.startswith('S2-')]
            s3_in_row = [m for m in matched_ids if m.startswith('S3-')]
            other_in_row = [m for m in matched_ids if not (m.startswith('S2-') or m.startswith('S3-'))]
            
            s2_links_count += len(s2_in_row)
            s3_links_count += len(s3_in_row)
            
            s2_match_count_dist[len(s2_in_row)] += 1
            s3_match_count_dist[len(s3_in_row)] += 1
            
            if len(s2_in_row) > 0 and len(s3_in_row) > 0:
                s1_matching_both += 1
            elif len(s2_in_row) > 0:
                s1_only_s2 += 1
            elif len(s3_in_row) > 0:
                s1_only_s3 += 1
                
            if other_in_row:
                unexpected_source_ids.extend(other_in_row)
                
            for m in matched_ids:
                matched_ids_counter[m] += 1
                if m.startswith('S2-') and m not in train_s2_ids:
                    invalid_ids.add(m)
                elif m.startswith('S3-') and m not in train_s3_ids:
                    invalid_ids.add(m)

    multi_referenced_matches = {k: v for k, v in matched_ids_counter.items() if v > 1}

    gt_stats = {
        'file_name': os.path.basename(gt_path),
        'file_size_mb': file_size_mb,
        'n_s1_entities': n_s1_rows,
        'dup_s1_in_gt': dup_s1_in_gt,
        'total_links': total_links,
        'zero_matches_count': zero_matches_count,
        'singleton_pct': (zero_matches_count / n_s1_rows * 100) if n_s1_rows > 0 else 0.0,
        'one_match_count': one_match_count,
        'multi_matches_count': multi_matches_count,
        's2_links_count': s2_links_count,
        's3_links_count': s3_links_count,
        's1_matching_both': s1_matching_both,
        's1_only_s2': s1_only_s2,
        's1_only_s3': s1_only_s3,
        'match_count_dist': dict(sorted(match_count_dist.items())),
        's2_match_count_dist': dict(sorted(s2_match_count_dist.items())),
        's3_match_count_dist': dict(sorted(s3_match_count_dist.items())),
        'invalid_ids_count': len(invalid_ids),
        'invalid_ids_sample': list(invalid_ids)[:10],
        'duplicate_gt_matched_ids_within_row': duplicate_gt_matched_ids_within_row,
        'unexpected_source_ids': unexpected_source_ids[:10],
        'multi_referenced_matches_count': len(multi_referenced_matches)
    }
    print(f"Finished Ground Truth: {n_s1_rows:,} S1 entities processed.", flush=True)
    return gt_stats

def main():
    base_dir = "."
    train_dir = os.path.join(base_dir, "dataset", "train")
    test_dir = os.path.join(base_dir, "dataset", "test")
    
    source_files = {
        'train_s1': os.path.join(train_dir, "train_source1.tsv"),
        'train_s2': os.path.join(train_dir, "train_source2.tsv"),
        'train_s3': os.path.join(train_dir, "train_source3.tsv"),
        'test_s1': os.path.join(test_dir, "test_source1.tsv"),
        'test_s2': os.path.join(test_dir, "test_source2.tsv"),
        'test_s3': os.path.join(test_dir, "test_source3.tsv"),
    }
    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    
    file_stats = {}
    id_sets = {}
    
    for key, path in source_files.items():
        stats, e_ids = analyze_source_file(path)
        file_stats[key] = stats
        id_sets[key] = e_ids

    gt_stats = analyze_ground_truth(gt_file, id_sets['train_s1'], id_sets['train_s2'], id_sets['train_s3'])
    
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "01_dataset_audit.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 1: Complete Dataset Audit Report\n\n")
        f.write("## 1. Executive Summary & File Integrity\n\n")
        f.write("| File Key | File Name | Size (MB) | Rows | Cols | Columns | Delimiter |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for k, s in file_stats.items():
            f.write(f"| `{k}` | `{s['file_name']}` | {s['file_size_mb']:.2f} | {s['row_count']:,} | {s['col_count']} | `{','.join(s['col_names'])}` | Tab (`\\t`) |\n")
        f.write(f"| `train_gt` | `{gt_stats['file_name']}` | {gt_stats['file_size_mb']:.2f} | {gt_stats['n_s1_entities']:,} | 2 | `source1_entity_id,matched_entity_ids` | Tab (`\\t`) |\n\n")
        
        f.write("## 2. Dataset Column & Missing Value Profiling\n\n")
        f.write("| File Key | Entity ID Dupes | Name Dupes | Addr Dupes | Empty Names | Empty Addrs | Short Names (<3) | Short Addrs (<5) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for k, s in file_stats.items():
            f.write(f"| `{k}` | {s['dup_ids']:,} | {s['dup_names']:,} | {s['dup_addrs']:,} | {s['empty_names']:,} ({s['missing_pct_dict'].get('business_name',0):.2f}%) | {s['empty_addrs']:,} ({s['missing_pct_dict'].get('business_address',0):.2f}%) | {s['short_names']:,} | {s['short_addrs']:,} |\n")
        f.write("\n")
        
        f.write("## 3. Country Distribution Analysis\n\n")
        f.write("| File Key | Country Distribution |\n")
        f.write("|---|---|\n")
        for k, s in file_stats.items():
            dist_str = ", ".join([f"**{c}**: {cnt:,}" for c, cnt in s['country_dist'].items()])
            f.write(f"| `{k}` | {dist_str} |\n")
        f.write("\n")

        f.write("## 4. Name & Address Length Statistics\n\n")
        f.write("### Business Name Length (Characters)\n\n")
        f.write("| File Key | Min | Q25 | Median | Mean | Q75 | Max | Std |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for k, s in file_stats.items():
            ns = s['name_len_stats']
            f.write(f"| `{k}` | {ns['min']} | {ns['q25']:.1f} | {ns['median']:.1f} | {ns['mean']:.2f} | {ns['q75']:.1f} | {ns['max']} | {ns['std']:.2f} |\n")
        f.write("\n")

        f.write("### Business Address Length (Characters)\n\n")
        f.write("| File Key | Min | Q25 | Median | Mean | Q75 | Max | Std |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for k, s in file_stats.items():
            as_ = s['addr_len_stats']
            f.write(f"| `{k}` | {as_['min']} | {as_['q25']:.1f} | {as_['median']:.1f} | {as_['mean']:.2f} | {as_['q75']:.1f} | {as_['max']} | {as_['std']:.2f} |\n")
        f.write("\n")

        f.write("## 5. Character, Punctuation & Token Noise Patterns\n\n")
        f.write("| File Key | Non-ASCII Names | Non-ASCII Addrs | Punctuation in Names | Numbers in Names | Punctuation in Addrs | Numbers in Addrs |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for k, s in file_stats.items():
            f.write(f"| `{k}` | {s['non_ascii_names']:,} | {s['non_ascii_addrs']:,} | {s['punct_names']:,} | {s['num_names']:,} | {s['punct_addrs']:,} | {s['num_addrs']:,} |\n")
        f.write("\n")

        f.write("## 6. Ground Truth Deep Dive\n\n")
        f.write(f"- **Total S1 Entities in Train GT:** {gt_stats['n_s1_entities']:,}\n")
        f.write(f"- **Duplicate S1 Entities in GT:** {gt_stats['dup_s1_in_gt']}\n")
        f.write(f"- **Total Ground-Truth Links (Matches):** {gt_stats['total_links']:,}\n")
        f.write(f"- **Zero Match S1 Entities (Singletons):** {gt_stats['zero_matches_count']:,} ({gt_stats['singleton_pct']:.2f}%)\n")
        f.write(f"- **One Match S1 Entities:** {gt_stats['one_match_count']:,} ({gt_stats['one_match_count']/gt_stats['n_s1_entities']*100:.2f}%)\n")
        f.write(f"- **Multiple Matches S1 Entities:** {gt_stats['multi_matches_count']:,} ({gt_stats['multi_matches_count']/gt_stats['n_s1_entities']*100:.2f}%)\n")
        f.write(f"- **Total S2 Links:** {gt_stats['s2_links_count']:,}\n")
        f.write(f"- **Total S3 Links:** {gt_stats['s3_links_count']:,}\n")
        f.write(f"- **S1 Entities matching ONLY S2:** {gt_stats['s1_only_s2']:,}\n")
        f.write(f"- **S1 Entities matching ONLY S3:** {gt_stats['s1_only_s3']:,}\n")
        f.write(f"- **S1 Entities matching BOTH S2 and S3:** {gt_stats['s1_matching_both']:,}\n")
        f.write(f"- **Invalid / Missing Source-2/3 IDs in GT:** {gt_stats['invalid_ids_count']}\n")
        f.write(f"- **Duplicate Matched IDs within same GT Row:** {gt_stats['duplicate_gt_matched_ids_within_row']}\n")
        f.write(f"- **Multi-Referenced S2/S3 Entities (Matched to >1 S1):** {gt_stats['multi_referenced_matches_count']}\n\n")

        f.write("### Total Match Count Distribution per S1 Entity\n\n")
        f.write("| Total Matches | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for num_m, cnt in gt_stats['match_count_dist'].items():
            f.write(f"| {num_m} | {cnt:,} | {cnt/gt_stats['n_s1_entities']*100:.2f}% |\n")
        f.write("\n")

        f.write("### S1 -> S2 Match Distribution\n\n")
        f.write("| S2 Matches | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for num_m, cnt in gt_stats['s2_match_count_dist'].items():
            f.write(f"| {num_m} | {cnt:,} | {cnt/gt_stats['n_s1_entities']*100:.2f}% |\n")
        f.write("\n")

        f.write("### S1 -> S3 Match Distribution\n\n")
        f.write("| S3 Matches | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for num_m, cnt in gt_stats['s3_match_count_dist'].items():
            f.write(f"| {num_m} | {cnt:,} | {cnt/gt_stats['n_s1_entities']*100:.2f}% |\n")
        f.write("\n")

    print(f"Audit completed successfully! Saved report to {report_path}", flush=True)

if __name__ == "__main__":
    main()
