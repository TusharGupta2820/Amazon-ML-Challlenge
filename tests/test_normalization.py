import os
import sys
import unittest

# Ensure src/preprocessing is on python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "preprocessing"))
from normalization import BusinessNormalizer

class TestBusinessNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = BusinessNormalizer()

    def test_unicode_normalization(self):
        self.assertEqual(self.normalizer.normalize_unicode("Café Société"), "Cafe Societe")
        self.assertEqual(self.normalizer.normalize_unicode("Aéroports de Paris"), "Aeroports de Paris")

    def test_business_name_normalization(self):
        self.assertEqual(self.normalizer.normalize_business_name("  ABC  &  Sons,  Inc.  "), "abc and sons inc")
        self.assertEqual(self.normalizer.normalize_business_name("AT&T Corp."), "at and t corp")

    def test_compact_business_name(self):
        self.assertEqual(self.normalizer.compact_business_name("ABC & Sons, Inc."), "abcandsonsinc")

    def test_legal_suffix_stripping(self):
        self.assertEqual(self.normalizer.strip_legal_suffix("abc and sons inc"), "abc and sons")
        self.assertEqual(self.normalizer.strip_legal_suffix("reliance pvt ltd"), "reliance")
        self.assertEqual(self.normalizer.strip_legal_suffix("tata private limited"), "tata")
        # Internal word should not be stripped
        self.assertEqual(self.normalizer.strip_legal_suffix("incorporated logistics"), "incorporated logistics")

    def test_address_normalization(self):
        self.assertEqual(
            self.normalizer.normalize_address("123 St. Mark Rd., Ste 4B"),
            "123 street mark road suite 4b"
        )
        self.assertEqual(
            self.normalizer.normalize_address("456 Ave. of Americas, Apt 12"),
            "456 avenue of americas apartment 12"
        )

    def test_numeric_extraction(self):
        self.assertEqual(self.normalizer.extract_numeric_tokens("12 MG Road 400001"), ["12", "400001"])

    def test_postal_code_extraction(self):
        self.assertEqual(self.normalizer.extract_postal_code("123 Main St, NY 10001", country="US"), "10001")
        self.assertEqual(self.normalizer.extract_postal_code("12 MG Road 400001", country="India"), "400001")
        self.assertEqual(self.normalizer.extract_postal_code("75008 Paris 75008", country="France"), "75008")

    def test_house_number_extraction(self):
        self.assertEqual(self.normalizer.extract_house_number("123-A Main St"), "123-A")
        self.assertEqual(self.normalizer.extract_house_number("456 Market Road"), "456")

    def test_country_open_set_normalization(self):
        self.assertEqual(self.normalizer.normalize_country("  US  "), "us")
        self.assertEqual(self.normalizer.normalize_country("India"), "india")
        self.assertEqual(self.normalizer.normalize_country("France"), "france")
        self.assertEqual(self.normalizer.normalize_country("Germany"), "germany")

    def test_raw_preservation(self):
        raw_rec = {
            "entity_id": "S1-001",
            "business_name": "ABC & Sons, Inc.",
            "business_address": "123 St. Mark Rd.",
            "country": "India"
        }
        enriched = self.normalizer.normalize_record(raw_rec)
        # Raw fields must be unchanged
        self.assertEqual(enriched["business_name_raw"], "ABC & Sons, Inc.")
        self.assertEqual(enriched["business_address_raw"], "123 St. Mark Rd.")
        self.assertEqual(enriched["country_raw"], "India")
        self.assertEqual(raw_rec["business_name"], "ABC & Sons, Inc.")

if __name__ == "__main__":
    unittest.main()
