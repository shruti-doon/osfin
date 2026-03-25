"""Tests for unique_matcher module."""

import pytest
from pathlib import Path

from src.data_loader import load_data
from src.preprocessor import preprocess_dataframe
from src.unique_matcher import find_unique_amount_matches, find_same_amount_groups


BASE_DIR = str(Path(__file__).parent.parent)


class TestFindUniqueAmountMatches:
    @pytest.fixture
    def data(self):
        bank_df, reg_df = load_data(BASE_DIR)
        bank_df = preprocess_dataframe(bank_df)
        reg_df = preprocess_dataframe(reg_df)
        return bank_df, reg_df

    def test_finds_matches(self, data):
        bank_df, reg_df = data
        matches = find_unique_amount_matches(bank_df, reg_df)
        assert len(matches) > 0
        assert len(matches) > 100

    def test_high_precision(self, data):
        bank_df, reg_df = data
        matches = find_unique_amount_matches(bank_df, reg_df)
        correct = sum(1 for m in matches if m.is_correct)
        precision = correct / len(matches) if matches else 0
        assert precision > 0.85

    def test_confidence_scores(self, data):
        bank_df, reg_df = data
        matches = find_unique_amount_matches(bank_df, reg_df)
        for m in matches:
            assert 0 <= m.confidence <= 1.0
            assert m.match_phase == 'unique_amount'

    def test_no_duplicate_ids(self, data):
        bank_df, reg_df = data
        matches = find_unique_amount_matches(bank_df, reg_df)
        bank_ids = [m.bank_id for m in matches]
        reg_ids = [m.register_id for m in matches]
        assert len(bank_ids) == len(set(bank_ids))
        assert len(reg_ids) == len(set(reg_ids))


class TestFindSameAmountGroups:
    def test_finds_groups(self):
        bank_df, reg_df = load_data(BASE_DIR)
        bank_df = preprocess_dataframe(bank_df)
        reg_df = preprocess_dataframe(reg_df)
        matches = find_unique_amount_matches(bank_df, reg_df)
        matched_bank = {m.bank_id for m in matches}
        matched_reg = {m.register_id for m in matches}
        groups = find_same_amount_groups(bank_df, reg_df, matched_bank, matched_reg)
        assert len(groups) > 0
