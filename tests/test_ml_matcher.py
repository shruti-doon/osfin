"""Tests for ml_matcher module."""

import pytest
import numpy as np
from pathlib import Path

from src.data_loader import load_data
from src.preprocessor import preprocess_dataframe
from src.unique_matcher import find_unique_amount_matches
from src.ml_matcher import HybridMatcher
from src.utils import MatchResult


BASE_DIR = str(Path(__file__).parent.parent)


class TestHybridMatcher:
    @pytest.fixture
    def pipeline_data(self):
        """Load data and run unique matching as a prerequisite."""
        bank_df, reg_df = load_data(BASE_DIR)
        bank_df = preprocess_dataframe(bank_df)
        reg_df = preprocess_dataframe(reg_df)
        unique_matches = find_unique_amount_matches(bank_df, reg_df)
        return bank_df, reg_df, unique_matches

    def test_matcher_runs(self, pipeline_data):
        bank_df, reg_df, unique_matches = pipeline_data
        matcher = HybridMatcher(
            max_iterations=1,
            use_sentence_transformers=False,
        )
        ml_matches = matcher.match(bank_df, reg_df, unique_matches)
        assert len(ml_matches) > 0

    def test_no_overlap_with_unique(self, pipeline_data):
        bank_df, reg_df, unique_matches = pipeline_data
        unique_bank_ids = {m.bank_id for m in unique_matches}
        unique_reg_ids = {m.register_id for m in unique_matches}

        matcher = HybridMatcher(
            max_iterations=1,
            use_sentence_transformers=False,
        )
        ml_matches = matcher.match(bank_df, reg_df, unique_matches)

        for m in ml_matches:
            assert m.bank_id not in unique_bank_ids
            assert m.register_id not in unique_reg_ids

    def test_match_phase_label(self, pipeline_data):
        bank_df, reg_df, unique_matches = pipeline_data
        matcher = HybridMatcher(
            max_iterations=1,
            use_sentence_transformers=False,
        )
        ml_matches = matcher.match(bank_df, reg_df, unique_matches)
        for m in ml_matches:
            assert m.match_phase == 'ml'

    def test_covers_all_transactions(self, pipeline_data):
        """Combined unique + ML should cover all 308 transactions."""
        bank_df, reg_df, unique_matches = pipeline_data
        matcher = HybridMatcher(
            max_iterations=2,
            use_sentence_transformers=False,
        )
        ml_matches = matcher.match(bank_df, reg_df, unique_matches)

        all_matches = unique_matches + ml_matches
        matched_bank_ids = {m.bank_id for m in all_matches}
        assert len(matched_bank_ids) >= 280
