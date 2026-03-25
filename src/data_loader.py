"""Data loading module for CSV parsing and normalization."""

import pandas as pd
from pathlib import Path
from typing import Tuple


def load_bank_statements(filepath: str) -> pd.DataFrame:
    """Load and normalize bank statements CSV.

    Columns: transaction_id, date, description, amount, type, balance
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].astype(float).round(2)

    type_map = {'DEBIT': 'DR', 'CREDIT': 'CR', 'debit': 'DR', 'credit': 'CR'}
    df['type_normalized'] = df['type'].map(type_map).fillna(df['type'].str.upper())

    df['signed_amount'] = df.apply(
        lambda r: -r['amount'] if r['type_normalized'] == 'DR' else r['amount'],
        axis=1
    )

    df['source'] = 'bank'
    return df


def load_check_register(filepath: str) -> pd.DataFrame:
    """Load and normalize check register CSV.

    Columns: transaction_id, date, description, amount, type, category, notes
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['date'] = pd.to_datetime(df['date'])
    df['amount'] = df['amount'].astype(float).round(2)

    df['type_normalized'] = df['type'].str.upper()

    df['signed_amount'] = df.apply(
        lambda r: -r['amount'] if r['type_normalized'] == 'DR' else r['amount'],
        axis=1
    )

    df['notes'] = df['notes'].fillna('')
    df['source'] = 'register'
    return df


def load_data(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both data files from the project directory.

    Returns:
        (bank_df, register_df) tuple
    """
    base = Path(base_dir)
    bank_df = load_bank_statements(str(base / 'bank_statements.csv'))
    register_df = load_check_register(str(base / 'check_register.csv'))
    return bank_df, register_df
