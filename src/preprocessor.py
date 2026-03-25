"""Text preprocessing and feature engineering for reconciliation."""

import re
import numpy as np
import pandas as pd
from typing import List, Dict

from src.utils import normalize_text, MERCHANT_ALIASES, categorize_description


def normalize_description(text: str) -> str:
    """Full description normalization pipeline.

    Steps:
    1. Lowercase & strip
    2. Remove # numbers
    3. Remove standalone numbers
    4. Map merchant name variants to canonical names
    5. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Remove hash numbers (e.g., '#1775', '#4099')
    text = re.sub(r'#\s*\d+', '', text)

    # Remove trailing standalone numbers
    text = re.sub(r'\b\d+\b', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def map_to_category(description: str) -> str:
    """Map a bank description to a semantic category using heuristics."""
    desc = normalize_description(description)

    # Try exact prefix matches first
    for pattern, category in MERCHANT_ALIASES.items():
        if desc.startswith(pattern) or pattern in desc:
            return category

    return categorize_description(description)


def extract_text_tokens(description: str) -> List[str]:
    """Tokenize a description into meaningful words."""
    desc = normalize_description(description)
    # Remove single-char tokens
    tokens = [t for t in desc.split() if len(t) > 1]
    return tokens


def compute_date_features(date: pd.Timestamp) -> Dict[str, float]:
    """Compute date-based features.

    Returns:
        dict with day_sin, day_cos (cyclic day-of-month encoding),
        month, day_of_week
    """
    day = date.day
    days_in_month = date.days_in_month

    # Cyclic encoding of day-of-month
    day_sin = np.sin(2 * np.pi * day / days_in_month)
    day_cos = np.cos(2 * np.pi * day / days_in_month)

    return {
        'day_sin': day_sin,
        'day_cos': day_cos,
        'month': date.month,
        'day_of_week': date.dayofweek,
    }


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add preprocessed features to a transaction dataframe.

    Adds columns:
    - desc_normalized: cleaned description text
    - category: semantic category
    - desc_tokens: list of tokens
    - day_sin, day_cos: cyclic date features
    - amount_log: log-scaled absolute amount
    """
    df = df.copy()

    # Normalize descriptions
    df['desc_normalized'] = df['description'].apply(normalize_description)

    # Category mapping
    df['category'] = df['description'].apply(map_to_category)

    # Tokenize
    df['desc_tokens'] = df['description'].apply(extract_text_tokens)

    # Date features
    date_features = df['date'].apply(compute_date_features).apply(pd.Series)
    df = pd.concat([df, date_features], axis=1)

    # Amount features
    df['amount_abs'] = df['amount'].abs()
    df['amount_log'] = np.log1p(df['amount_abs'])

    # Type as numeric (DR=0, CR=1)
    df['type_num'] = (df['type_normalized'] == 'CR').astype(int)

    return df
