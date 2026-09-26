import os
import sys

class BaseBlockingIndex:
    def __init__(self, name="BaseIndex"):
        self.name = name
        self.s2_index = {}
        self.s3_index = {}

    def build(self, s2_records, s3_records):
        """Build index from normalized S2 and S3 record streams/lists."""
        raise NotImplementedError

    def get_candidates(self, s1_record):
        """Retrieve candidate IDs for a given normalized S1 record.
        Returns tuple: (set_of_s2_ids, set_of_s3_ids)
        """
        raise NotImplementedError
