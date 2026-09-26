import os
import sys
import re
import json
import unicodedata

DEFAULT_CONFIG_PATH = os.path.join("configs", "normalization_config.json")

class BusinessNormalizer:
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        if os.path.isfile(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {
                "unicode_normalization_form": "NFKD",
                "legal_suffixes": [
                    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited", 
                    "pvt", "pvt ltd", "private limited", "co", "company", "gmbh", "sa", "sarl", "bv"
                ],
                "address_abbreviations": {
                    "st": "street", "rd": "road", "ave": "avenue", "dr": "drive",
                    "blvd": "boulevard", "ste": "suite", "apt": "apartment"
                },
                "postal_code_regex": {
                    "generic": r"\b\d{5,6}\b"
                },
                "house_number_regex": r"^\s*(\d+[-/a-zA-Z0-9]*)"
            }
            
        self.unicode_form = config.get("unicode_normalization_form", "NFKD")
        self.legal_suffixes = sorted(config.get("legal_suffixes", []), key=len, reverse=True)
        self.address_abbrev_map = config.get("address_abbreviations", {})
        
        self.postal_code_regexes = {
            k: re.compile(v) for k, v in config.get("postal_code_regex", {}).items()
        }
        self.house_number_regex = re.compile(config.get("house_number_regex", r"^\s*(\d+[-/a-zA-Z0-9]*)"))
        self.re_punct = re.compile(r"[^\w\s]")
        self.re_digits = re.compile(r"\b\d+\b")

    def normalize_unicode(self, text):
        if not text:
            return ""
        norm_text = unicodedata.normalize(self.unicode_form, text)
        return "".join(c for c in norm_text if not unicodedata.combining(c))

    def normalize_business_name(self, name):
        if not name:
            return ""
        name_clean = self.normalize_unicode(name).lower()
        name_clean = re.sub(r"\s*&\s*", " and ", name_clean)
        name_clean = self.re_punct.sub(" ", name_clean)
        return " ".join(name_clean.split())

    def compact_business_name(self, name):
        norm_name = self.normalize_business_name(name)
        return norm_name.replace(" ", "")

    def strip_legal_suffix(self, norm_name):
        if not norm_name:
            return ""
        words = norm_name.split()
        changed = True
        while words and changed:
            changed = False
            curr_str = " ".join(words)
            for suffix in self.legal_suffixes:
                if curr_str.endswith(" " + suffix) or curr_str == suffix:
                    words = words[:-len(suffix.split())]
                    changed = True
                    break
        return " ".join(words) if words else norm_name

    def tokenize_name(self, name):
        norm_name = self.normalize_business_name(name)
        return norm_name.split()

    def normalize_address(self, address):
        if not address:
            return ""
        addr_clean = self.normalize_unicode(address).lower()
        addr_clean = self.re_punct.sub(" ", addr_clean)
        tokens = addr_clean.split()
        expanded_tokens = [self.address_abbrev_map.get(t, t) for t in tokens]
        return " ".join(expanded_tokens)

    def compact_address(self, address):
        norm_addr = self.normalize_address(address)
        return norm_addr.replace(" ", "")

    def tokenize_address(self, address):
        norm_addr = self.normalize_address(address)
        return norm_addr.split()

    def extract_numeric_tokens(self, address):
        if not address:
            return []
        return self.re_digits.findall(address)

    def extract_postal_code(self, address, country=None):
        if not address:
            return ""
        if country and country in self.postal_code_regexes:
            match = self.postal_code_regexes[country].search(address)
            if match:
                return match.group(0)
        match = self.postal_code_regexes.get("generic", re.compile(r"\b\d{5,6}\b")).search(address)
        return match.group(0) if match else ""

    def extract_house_number(self, address):
        if not address:
            return ""
        match = self.house_number_regex.search(address)
        return match.group(1) if match else ""

    def normalize_country(self, country):
        if not country:
            return ""
        return self.normalize_unicode(country).strip().lower()

    def get_source_prefix(self, entity_id):
        if not entity_id:
            return "UNKNOWN"
        if entity_id.startswith("S1-"):
            return "S1"
        elif entity_id.startswith("S2-"):
            return "S2"
        elif entity_id.startswith("S3-"):
            return "S3"
        return "UNKNOWN"

    def normalize_record(self, record):
        eid = record.get("entity_id", "").strip()
        raw_name = record.get("business_name", "")
        raw_addr = record.get("business_address", "")
        raw_cty = record.get("country", "")

        name_norm = self.normalize_business_name(raw_name)
        name_compact = self.compact_business_name(raw_name)
        name_suffixless = self.strip_legal_suffix(name_norm)
        name_tokens = name_norm.split()

        addr_norm = self.normalize_address(raw_addr)
        addr_compact = self.compact_address(raw_addr)
        addr_tokens = addr_norm.split()
        numeric_tokens = self.extract_numeric_tokens(raw_addr)
        postal_code = self.extract_postal_code(raw_addr, country=raw_cty)
        house_number = self.extract_house_number(raw_addr)

        cty_norm = self.normalize_country(raw_cty)

        return {
            "entity_id": eid,
            "source": self.get_source_prefix(eid),
            "business_name_raw": raw_name,
            "business_address_raw": raw_addr,
            "country_raw": raw_cty,
            "business_name_norm": name_norm,
            "business_name_compact": name_compact,
            "business_name_suffixless": name_suffixless,
            "business_name_tokens": name_tokens,
            "business_address_norm": addr_norm,
            "business_address_compact": addr_compact,
            "business_address_tokens": addr_tokens,
            "business_address_numeric_tokens": numeric_tokens,
            "postal_code_candidate": postal_code,
            "house_number_candidate": house_number,
            "country_norm": cty_norm,
            "name_missing": len(name_norm) == 0,
            "address_missing": len(addr_norm) == 0,
            "country_missing": len(cty_norm) == 0
        }

if __name__ == "__main__":
    normalizer = BusinessNormalizer()
    sample_rec = {
        "entity_id": "S1-00001",
        "business_name": "ABC & Sons Corporation, Inc.",
        "business_address": "123 St. Mark Rd., Apt 4B, 400001",
        "country": "India"
    }
    enriched = normalizer.normalize_record(sample_rec)
    print("Sample enriched record:")
    for k, v in enriched.items():
        print(f"  {k}: {v}")
