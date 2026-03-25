"""Tests for preprocessor module."""

import pytest
import pandas as pd
from pathlib import Path

from src.preprocessor import (
    normalize_description,
    map_to_category,
    extract_text_tokens,
    compute_date_features,
    preprocess_dataframe,
)
from src.data_loader import load_data


BASE_DIR = str(Path(__file__).parent.parent)


class TestNormalizeDescription:
    def test_basic_normalization(self):
        assert normalize_description("BP GAS #1775") == "bp gas"

    def test_caps_handling(self):
        assert normalize_description("TRADER JOES") == "trader joes"

    def test_hash_removal(self):
        assert normalize_description("KROGER #6864") == "kroger"

    def test_typo_lowercase(self):
        assert normalize_description("AAZON.COM") == "aazon.com"

    def test_empty_input(self):
        assert normalize_description("") == ""
        assert normalize_description(None) == ""


class TestMapToCategory:
    def test_gas_station(self):
        assert map_to_category("BP GAS #1775") == "gas station"
        assert map_to_category("CHEVRON #9838") == "gas station"
        assert map_to_category("EXXON #1995") == "gas station"

    def test_grocery(self):
        assert map_to_category("KROGER #6864") == "grocery"
        assert map_to_category("TRADER JOES") == "grocery"
        assert map_to_category("WHOLE FOODS #3931") == "grocery"

    def test_subscription(self):
        assert map_to_category("NETFLIX") == "subscription"
        assert map_to_category("SPOTIFY") == "subscription"

    def test_online_purchase(self):
        category = map_to_category("AMAZON.COM")
        assert category in ("amazon", "online purchase", "grocery")
        assert map_to_category("ONLINE ORDER #4005") == "online purchase"


class TestExtractTextTokens:
    def test_tokenization(self):
        tokens = extract_text_tokens("BP GAS #1775")
        assert "bp" in tokens
        assert "gas" in tokens

    def test_no_numbers(self):
        tokens = extract_text_tokens("KROGER #6864")
        assert all(not t.isdigit() for t in tokens)


class TestComputeDateFeatures:
    def test_cyclic_encoding(self):
        date = pd.Timestamp("2023-01-15")
        features = compute_date_features(date)
        assert 'day_sin' in features
        assert 'day_cos' in features
        assert -1 <= features['day_sin'] <= 1
        assert -1 <= features['day_cos'] <= 1


class TestPreprocessDataframe:
    def test_adds_columns(self):
        bank_df, _ = load_data(BASE_DIR)
        processed = preprocess_dataframe(bank_df)
        assert 'desc_normalized' in processed.columns
        assert 'category' in processed.columns
        assert 'day_sin' in processed.columns
        assert 'amount_log' in processed.columns
        assert 'type_num' in processed.columns
