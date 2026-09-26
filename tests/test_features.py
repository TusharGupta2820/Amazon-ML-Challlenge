import os
import sys
import unittest
import numpy as np

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(base_dir, "src", "preprocessing"))
sys.path.insert(0, os.path.join(base_dir, "src", "features"))

from normalization import BusinessNormalizer
from name_features import NameFeatureExtractor
from address_features import AddressFeatureExtractor
from country_features import CountryFeatureExtractor
from blocking_features import BlockingAndSourceFeatureExtractor

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.normalizer = BusinessNormalizer()
        self.name_ext = NameFeatureExtractor()
        self.addr_ext = AddressFeatureExtractor()
        self.country_ext = CountryFeatureExtractor()
        self.block_ext = BlockingAndSourceFeatureExtractor()

    def test_01_identical_raw_names(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Acme Supply Corp", "business_address": "123 Main St", "country": "US"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "Acme Supply Corp", "business_address": "123 Main St", "country": "US"})
        f = self.name_ext.extract_pair_features(r1, r2)
        self.assertEqual(f["name_exact_raw"], 1)

    def test_02_identical_normalized_names(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Acme Supply Corp.", "business_address": "123 Main St", "country": "US"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "acme supply corp", "business_address": "123 Main St", "country": "US"})
        f = self.name_ext.extract_pair_features(r1, r2)
        self.assertEqual(f["name_exact_normalized"], 1)

    def test_03_suffixless_exact_match(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Acme Supply LLC", "business_address": "123 Main St", "country": "US"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "Acme Supply Inc", "business_address": "123 Main St", "country": "US"})
        f = self.name_ext.extract_pair_features(r1, r2)
        self.assertEqual(f["name_exact_suffixless"], 1)

    def test_04_identical_addresses(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Acme", "business_address": "123 Main Street Suite 400", "country": "US"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "Acme", "business_address": "123 Main Street Suite 400", "country": "US"})
        f = self.addr_ext.extract_pair_features(r1, r2)
        self.assertEqual(f["address_exact_normalized"], 1)

    def test_05_postal_conflict(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Acme", "business_address": "123 Main St NY 10001", "country": "US"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "Acme", "business_address": "123 Main St NY 90210", "country": "US"})
        f = self.addr_ext.extract_pair_features(r1, r2)
        self.assertEqual(f["postal_conflict"], 1)

    def test_06_postal_exact(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Acme", "business_address": "123 Main St NY 10001", "country": "US"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "Acme", "business_address": "456 Oak St NY 10001", "country": "US"})
        f = self.addr_ext.extract_pair_features(r1, r2)
        self.assertEqual(f["postal_exact"], 1)

    def test_07_house_number_exact(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Acme", "business_address": "100 Broadway", "country": "US"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "Acme", "business_address": "100 Broadway", "country": "US"})
        f = self.addr_ext.extract_pair_features(r1, r2)
        self.assertEqual(f["house_number_exact"], 1)

    def test_08_house_number_conflict(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Acme", "business_address": "100 Broadway", "country": "US"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "Acme", "business_address": "200 Broadway", "country": "US"})
        f = self.addr_ext.extract_pair_features(r1, r2)
        self.assertEqual(f["house_number_conflict"], 1)

    def test_09_missing_address(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Acme", "business_address": "", "country": "US"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "Acme", "business_address": "100 Broadway", "country": "US"})
        f = self.addr_ext.extract_pair_features(r1, r2)
        self.assertEqual(f["address_missing_s1"], 1)
        self.assertEqual(f["address_missing_either"], 1)

    def test_10_same_country(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Acme", "business_address": "100 Broadway", "country": "France"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "Acme", "business_address": "100 Broadway", "country": "France"})
        f = self.country_ext.extract_pair_features(r1, r2)
        self.assertEqual(f["country_exact_match"], 1)

    def test_11_different_country(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Acme", "business_address": "100 Broadway", "country": "US"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "Acme", "business_address": "100 Broadway", "country": "France"})
        f = self.country_ext.extract_pair_features(r1, r2)
        self.assertEqual(f["country_mismatch"], 1)

    def test_12_source_s2_s3_flags(self):
        r2 = self.normalizer.normalize_record({"entity_id": "S2-100", "business_name": "Acme", "business_address": "100 Broadway", "country": "US"})
        f = self.block_ext.extract_pair_features(set(), set(), set(), set(), "S2-100", "S2", r2)
        self.assertEqual(f["candidate_is_s2"], 1)
        self.assertEqual(f["candidate_is_s3"], 0)

    def test_13_blocking_flags_reconstruction(self):
        r2 = self.normalizer.normalize_record({"entity_id": "S2-100", "business_name": "Acme", "business_address": "100 Broadway", "country": "US"})
        f = self.block_ext.extract_pair_features({"S2-100"}, {"S2-100"}, set(), set(), "S2-100", "S2", r2)
        self.assertEqual(f["blocked_by_A_suffixless_name"], 1)
        self.assertEqual(f["blocked_by_B_rare_token"], 1)
        self.assertEqual(f["blocking_strategy_count"], 2)
        self.assertEqual(f["blocking_multi_strategy"], 1)

    def test_14_no_nan_values(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Test Company", "business_address": "123 Test St", "country": "US"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "Test Company Inc", "business_address": "123 Test St", "country": "US"})
        f1 = self.name_ext.extract_pair_features(r1, r2)
        f2 = self.addr_ext.extract_pair_features(r1, r2)
        for k, v in list(f1.items()) + list(f2.items()):
            self.assertFalse(np.isnan(v), f"NaN found in feature {k}")

    def test_15_similarity_ranges(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Acme Corp", "business_address": "123 Main St", "country": "US"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "Beta LLC", "business_address": "456 Oak St", "country": "US"})
        f = self.name_ext.extract_pair_features(r1, r2)
        self.assertTrue(0.0 <= f["name_levenshtein_similarity"] <= 1.0)
        self.assertTrue(0.0 <= f["name_jaro_winkler_similarity"] <= 1.0)
        self.assertTrue(0.0 <= f["name_token_jaccard"] <= 1.0)

    def test_16_deterministic_ordering(self):
        r1 = self.normalizer.normalize_record({"entity_id": "S1-1", "business_name": "Delta Systems", "business_address": "789 Pine St", "country": "US"})
        r2 = self.normalizer.normalize_record({"entity_id": "S2-1", "business_name": "Delta Systems Inc", "business_address": "789 Pine St", "country": "US"})
        f_a = self.name_ext.extract_pair_features(r1, r2)
        f_b = self.name_ext.extract_pair_features(r1, r2)
        self.assertEqual(f_a, f_b)

if __name__ == "__main__":
    unittest.main()
