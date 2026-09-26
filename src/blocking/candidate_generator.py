import os
import sys
import json
from exact_name_index import SuffixlessExactNameIndex
from token_index import RareTokenIndex
from address_index import AddressPostcodeIndex
from ngram_retrieval import CharacterNGramIndex

DEFAULT_CONFIG_PATH = os.path.join("configs", "blocking_config.json")

class MultiStrategyCandidateGenerator:
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self._load_config()
        self.strategies = []
        self._init_strategies()

    def _load_config(self):
        if os.path.isfile(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "max_suffixless_group_size": 50,
                "max_token_doc_freq": 5000,
                "rare_tokens_per_name": 2,
                "ngram_size": 3,
                "ngram_top_k": 20,
                "enable_strategy_a": True,
                "enable_strategy_b": True,
                "enable_strategy_c": True,
                "enable_strategy_d": True
            }

    def _init_strategies(self):
        if self.config.get("enable_strategy_a", True):
            self.strategies.append(
                SuffixlessExactNameIndex(max_group_size=self.config.get("max_suffixless_group_size", 50))
            )
        if self.config.get("enable_strategy_b", True):
            self.strategies.append(
                RareTokenIndex(
                    max_token_doc_freq=self.config.get("max_token_doc_freq", 5000),
                    rare_tokens_per_name=self.config.get("rare_tokens_per_name", 2)
                )
            )
        if self.config.get("enable_strategy_c", True):
            self.strategies.append(
                AddressPostcodeIndex()
            )
        if self.config.get("enable_strategy_d", True):
            self.strategies.append(
                CharacterNGramIndex(
                    ngram_size=self.config.get("ngram_size", 3),
                    top_k=self.config.get("ngram_top_k", 20)
                )
            )

    def build_indexes(self, s2_records, s3_records):
        print(f"Building {len(self.strategies)} blocking strategy indexes...", flush=True)
        for strategy in self.strategies:
            strategy.build(s2_records, s3_records)
        print("All blocking indexes built successfully!", flush=True)

    def generate_candidates_for_s1(self, s1_record, active_strategies=None):
        """Generate candidate S2 and S3 IDs for an S1 record.
        Optionally filter active_strategies for ablation experiments.
        """
        all_s2_cands = set()
        all_s3_cands = set()

        target_strategies = active_strategies if active_strategies is not None else self.strategies

        for strategy in target_strategies:
            s2_cands, s3_cands = strategy.get_candidates(s1_record)
            all_s2_cands.update(s2_cands)
            all_s3_cands.update(s3_cands)

        # Deterministic sorting
        sorted_s2 = sorted(all_s2_cands)
        sorted_s3 = sorted(all_s3_cands)

        return sorted_s2 + sorted_s3, sorted_s2, sorted_s3

    def generate_candidates_with_per_strategy_breakdown(self, s1_record):
        s2_a, s3_a = self.strategies[0].get_candidates(s1_record)
        s2_b, s3_b = self.strategies[1].get_candidates(s1_record)
        s2_c, s3_c = self.strategies[2].get_candidates(s1_record)
        s2_d, s3_d = self.strategies[3].get_candidates(s1_record)

        set_a = s2_a | s3_a
        set_b = s2_b | s3_b
        set_c = s2_c | s3_c
        set_d = s2_d | s3_d

        all_cands = sorted(set_a | set_b | set_c | set_d)
        return all_cands, set_a, set_b, set_c, set_d

