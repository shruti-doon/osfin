"""Tests for data_loader module."""

import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import load_bank_statements, load_check_register, load_data


BASE_DIR = str(Path(__file__).parent.parent)


class TestLoadBankStatements:
    def test_loads_correct_shape(self):
        df = load_bank_statements(f'{BASE_DIR}/bank_statements.csv')
        assert len(df) == 308
        assert 'transaction_id' in df.columns
        assert 'date' in df.columns
        assert 'amount' in df.columns

    def test_date_parsing(self):
        df = load_bank_statements(f'{BASE_DIR}/bank_statements.csv')
        assert pd.api.types.is_datetime64_any_dtype(df['date'])

    def test_type_normalization(self):
        df = load_bank_statements(f'{BASE_DIR}/bank_statements.csv')
        assert set(df['type_normalized'].unique()) == {'DR', 'CR'}

    def test_signed_amount(self):
        df = load_bank_statements(f'{BASE_DIR}/bank_statements.csv')
        debits = df[df['type_normalized'] == 'DR']
        credits = df[df['type_normalized'] == 'CR']
        assert (debits['signed_amount'] < 0).all()
        assert (credits['signed_amount'] > 0).all()


class TestLoadCheckRegister:
    def test_loads_correct_shape(self):
        df = load_check_register(f'{BASE_DIR}/check_register.csv')
        assert len(df) == 308
        assert 'transaction_id' in df.columns
        assert 'category' in df.columns

    def test_type_already_dr_cr(self):
        df = load_check_register(f'{BASE_DIR}/check_register.csv')
        assert set(df['type_normalized'].unique()) == {'DR', 'CR'}

    def test_notes_filled(self):
        df = load_check_register(f'{BASE_DIR}/check_register.csv')
        assert df['notes'].isna().sum() == 0


class TestLoadData:
    def test_loads_both(self):
        bank_df, reg_df = load_data(BASE_DIR)
        assert len(bank_df) == 308
        assert len(reg_df) == 308
