"""Pipeline orchestrator for the financial reconciliation system.

Coordinates the full flow: load → preprocess → match → evaluate → report.
"""

import time
from typing import List, Optional, Tuple
from pathlib import Path

import pandas as pd

from src.data_loader import load_data
from src.preprocessor import preprocess_dataframe
from src.unique_matcher import find_unique_amount_matches
from src.ml_matcher import HybridMatcher
from src.evaluator import compute_metrics, compute_metrics_by_phase, generate_evaluation_report
from src.utils import MatchResult


class ReconciliationPipeline:
    """Orchestrates the full reconciliation pipeline."""

    def __init__(
        self,
        base_dir: str,
        use_sentence_transformers: bool = True,
        svd_components: int = 40,
        max_iterations: int = 3,
    ):
        self.base_dir = base_dir
        self.use_sentence_transformers = use_sentence_transformers
        self.svd_components = svd_components
        self.max_iterations = max_iterations

        self.bank_df: Optional[pd.DataFrame] = None
        self.register_df: Optional[pd.DataFrame] = None
        self.unique_matches: List[MatchResult] = []
        self.ml_matches: List[MatchResult] = []
        self.all_matches: List[MatchResult] = []
        self.timings: dict = {}

    def load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Phase 0: Load and preprocess data."""
        t0 = time.time()

        bank_df, register_df = load_data(self.base_dir)
        self.bank_df = preprocess_dataframe(bank_df)
        self.register_df = preprocess_dataframe(register_df)

        self.timings['load_preprocess'] = time.time() - t0
        return self.bank_df, self.register_df

    def run_unique_matching(self) -> List[MatchResult]:
        """Phase 1: Unique amount matching."""
        if self.bank_df is None:
            self.load_and_preprocess()

        t0 = time.time()
        self.unique_matches = find_unique_amount_matches(
            self.bank_df, self.register_df
        )
        self.timings['unique_matching'] = time.time() - t0

        return self.unique_matches

    def run_ml_matching(self) -> List[MatchResult]:
        """Phase 2: ML-based matching for remaining transactions."""
        if not self.unique_matches:
            self.run_unique_matching()

        t0 = time.time()

        matched_bank_ids = {m.bank_id for m in self.unique_matches}
        matched_reg_ids = {m.register_id for m in self.unique_matches}

        matcher = HybridMatcher(
            svd_components=self.svd_components,
            max_iterations=self.max_iterations,
            use_sentence_transformers=self.use_sentence_transformers,
        )

        self.ml_matches = matcher.match(
            self.bank_df,
            self.register_df,
            seed_matches=self.unique_matches,
            matched_bank_ids=matched_bank_ids,
            matched_reg_ids=matched_reg_ids,
        )

        self.timings['ml_matching'] = time.time() - t0
        return self.ml_matches

    def run_full_pipeline(self) -> List[MatchResult]:
        """Run the complete reconciliation pipeline.

        Returns:
            All matches (unique + ML)
        """
        self.load_and_preprocess()
        self.run_unique_matching()
        self.run_ml_matching()

        self.all_matches = self.unique_matches + self.ml_matches
        return self.all_matches

    def get_metrics(self) -> dict:
        """Get evaluation metrics for all matches."""
        return compute_metrics(self.all_matches)

    def get_phase_metrics(self) -> dict:
        """Get metrics broken down by matching phase."""
        return compute_metrics_by_phase(self.all_matches)

    def get_report(self) -> str:
        """Generate the full evaluation report."""
        return generate_evaluation_report(self.all_matches)

    def get_unmatched(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get transactions that were not matched."""
        matched_bank = {m.bank_id for m in self.all_matches}
        matched_reg = {m.register_id for m in self.all_matches}

        unmatched_bank = self.bank_df[~self.bank_df['transaction_id'].isin(matched_bank)]
        unmatched_reg = self.register_df[~self.register_df['transaction_id'].isin(matched_reg)]

        return unmatched_bank, unmatched_reg

    def export_matches_csv(self, output_path: str):
        """Export matches to CSV file."""
        rows = []
        for m in self.all_matches:
            bank_row = self.bank_df[self.bank_df['transaction_id'] == m.bank_id].iloc[0]
            reg_row = self.register_df[self.register_df['transaction_id'] == m.register_id].iloc[0]

            rows.append({
                'bank_id': m.bank_id,
                'register_id': m.register_id,
                'bank_description': bank_row['description'],
                'register_description': reg_row['description'],
                'amount': bank_row['amount'],
                'bank_date': bank_row['date'].strftime('%Y-%m-%d'),
                'register_date': reg_row['date'].strftime('%Y-%m-%d'),
                'confidence': m.confidence,
                'match_phase': m.match_phase,
                'is_correct': m.is_correct,
                'flags': '; '.join(m.flags),
            })

        pd.DataFrame(rows).to_csv(output_path, index=False)
