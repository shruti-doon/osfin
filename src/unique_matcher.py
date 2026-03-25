"""Phase 1: Unique amount matching.

Matches transactions that have a unique amount in both bank statements
and check register. Applies confidence scoring based on date proximity
and type agreement.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

from src.utils import MatchResult


def find_unique_amount_matches(
    bank_df: pd.DataFrame,
    register_df: pd.DataFrame,
    date_penalty_threshold: int = 5,
    date_penalty_per_day: float = 0.05,
    type_mismatch_penalty: float = 0.2,
    min_confidence: float = 0.0,
) -> List[MatchResult]:
    """Find transactions with unique amounts in both datasets.

    Algorithm:
    1. Round amounts to 2 decimal places
    2. Find amounts appearing exactly once in bank AND once in register
    3. Match these pairs, compute confidence scores
    4. Handle near-duplicate amounts with date/type tiebreaking

    Args:
        bank_df: Preprocessed bank statements DataFrame
        register_df: Preprocessed check register DataFrame
        date_penalty_threshold: Days of date diff before penalty kicks in
        date_penalty_per_day: Confidence penalty per extra day beyond threshold
        type_mismatch_penalty: Confidence penalty for type disagreement
        min_confidence: Minimum confidence to include in results

    Returns:
        List of MatchResult objects
    """
    matches = []

    bank_amounts = bank_df.groupby('amount').agg(
        count=('transaction_id', 'size'),
        ids=('transaction_id', list),
        dates=('date', list),
        types=('type_normalized', list),
    ).reset_index()

    reg_amounts = register_df.groupby('amount').agg(
        count=('transaction_id', 'size'),
        ids=('transaction_id', list),
        dates=('date', list),
        types=('type_normalized', list),
    ).reset_index()

    bank_unique = bank_amounts[bank_amounts['count'] == 1]
    reg_unique = reg_amounts[reg_amounts['count'] == 1]

    merged = bank_unique.merge(
        reg_unique,
        on='amount',
        suffixes=('_bank', '_reg')
    )

    for _, row in merged.iterrows():
        bank_id = row['ids_bank'][0]
        reg_id = row['ids_reg'][0]
        bank_date = row['dates_bank'][0]
        reg_date = row['dates_reg'][0]
        bank_type = row['types_bank'][0]
        reg_type = row['types_reg'][0]

        confidence = 1.0
        flags = []

        date_diff = abs((bank_date - reg_date).days)
        if date_diff > date_penalty_threshold:
            extra_days = date_diff - date_penalty_threshold
            penalty = extra_days * date_penalty_per_day
            confidence -= penalty
            flags.append(f'date_diff={date_diff}d')

        if bank_type != reg_type:
            confidence -= type_mismatch_penalty
            flags.append(f'type_mismatch:{bank_type}≠{reg_type}')

        confidence = max(confidence, 0.0)

        if confidence >= min_confidence:
            matches.append(MatchResult(
                bank_id=bank_id,
                register_id=reg_id,
                confidence=round(confidence, 4),
                match_phase='unique_amount',
                flags=flags,
            ))

    return matches


def find_same_amount_groups(
    bank_df: pd.DataFrame,
    register_df: pd.DataFrame,
    matched_bank_ids: set,
    matched_reg_ids: set,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Find groups of unmatched transactions sharing the same amount.

    These will be passed to the ML matcher for disambiguation.

    Returns:
        List of (bank_group_df, register_group_df) tuples
    """
    bank_remaining = bank_df[~bank_df['transaction_id'].isin(matched_bank_ids)]
    reg_remaining = register_df[~register_df['transaction_id'].isin(matched_reg_ids)]

    groups = []
    bank_by_amount = bank_remaining.groupby('amount')
    reg_by_amount = reg_remaining.groupby('amount')

    for amount in bank_by_amount.groups.keys():
        if amount in reg_by_amount.groups:
            b_group = bank_by_amount.get_group(amount)
            r_group = reg_by_amount.get_group(amount)
            groups.append((b_group, r_group))

    return groups
