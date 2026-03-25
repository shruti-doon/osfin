"""Tests for evaluator module."""

import pytest

from src.utils import MatchResult
from src.evaluator import (
    compute_metrics,
    compute_metrics_by_phase,
    compute_metrics_by_confidence,
    find_incorrect_matches,
    find_hardest_matches,
    generate_evaluation_report,
)


def make_match(bank_num, reg_num, confidence=0.9, phase='unique_amount'):
    """Helper to create a MatchResult."""
    return MatchResult(
        bank_id=f'B{bank_num:04d}',
        register_id=f'R{reg_num:04d}',
        confidence=confidence,
        match_phase=phase,
    )


class TestComputeMetrics:
    def test_perfect_matches(self):
        matches = [make_match(i, i) for i in range(1, 309)]
        metrics = compute_metrics(matches)
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0

    def test_no_matches(self):
        metrics = compute_metrics([])
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0

    def test_partial_correct(self):
        matches = [
            make_match(1, 1),
            make_match(2, 3),
        ]
        metrics = compute_metrics(matches)
        assert metrics['precision'] == 0.5
        assert metrics['correct_count'] == 1
        assert metrics['total_matches'] == 2

    def test_is_correct_logic(self):
        m1 = make_match(47, 47)
        assert m1.is_correct is True
        m2 = make_match(47, 48)
        assert m2.is_correct is False


class TestComputeMetricsByPhase:
    def test_separates_phases(self):
        matches = [
            make_match(1, 1, phase='unique_amount'),
            make_match(2, 2, phase='ml'),
            make_match(3, 4, phase='ml'),
        ]
        by_phase = compute_metrics_by_phase(matches)
        assert 'unique_amount' in by_phase
        assert 'ml' in by_phase
        assert by_phase['unique_amount']['precision'] == 1.0
        assert by_phase['ml']['precision'] == 0.5


class TestComputeMetricsByConfidence:
    def test_buckets(self):
        matches = [
            make_match(1, 1, confidence=0.95),
            make_match(2, 2, confidence=0.75),
            make_match(3, 4, confidence=0.2),
        ]
        by_conf = compute_metrics_by_confidence(matches)
        assert len(by_conf) > 0


class TestFindIncorrect:
    def test_finds_incorrect(self):
        matches = [
            make_match(1, 1),
            make_match(2, 3),
        ]
        incorrect = find_incorrect_matches(matches)
        assert len(incorrect) == 1
        assert incorrect[0].bank_id == 'B0002'


class TestFindHardest:
    def test_finds_lowest_confidence(self):
        matches = [
            make_match(1, 1, confidence=0.95),
            make_match(2, 2, confidence=0.5),
            make_match(3, 3, confidence=0.3),
        ]
        hardest = find_hardest_matches(matches, n=2)
        assert len(hardest) == 2
        assert hardest[0].confidence == 0.3


class TestGenerateReport:
    def test_report_is_string(self):
        matches = [make_match(i, i) for i in range(1, 11)]
        report = generate_evaluation_report(matches)
        assert isinstance(report, str)
        assert 'OVERALL METRICS' in report
