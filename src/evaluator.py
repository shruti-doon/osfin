"""Evaluation module for computing precision, recall, and F1 score.

Uses ground truth based on transaction ID correspondence: B0047 <-> R0047.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

from src.utils import MatchResult, extract_id_number


def compute_metrics(matches: List[MatchResult]) -> Dict[str, float]:
    """Compute precision, recall, and F1 from matches.

    Ground truth: B0047 should match R0047 (same numeric suffix).
    Total expected matches: 308.

    Returns:
        Dict with precision, recall, f1, correct_count, total_matches
    """
    total_expected = 308

    if not matches:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'correct_count': 0,
            'total_matches': 0,
            'total_expected': total_expected,
        }

    correct = sum(1 for m in matches if m.is_correct)
    total = len(matches)

    precision = correct / total if total > 0 else 0.0
    recall = correct / total_expected
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'correct_count': correct,
        'total_matches': total,
        'total_expected': total_expected,
    }


def compute_metrics_by_phase(matches: List[MatchResult]) -> Dict[str, Dict[str, float]]:
    """Compute metrics broken down by match phase.

    Returns:
        Dict mapping phase name -> metrics dict
    """
    phases = defaultdict(list)
    for m in matches:
        phases[m.match_phase].append(m)

    results = {}
    for phase, phase_matches in phases.items():
        correct = sum(1 for m in phase_matches if m.is_correct)
        total = len(phase_matches)
        precision = correct / total if total > 0 else 0.0
        results[phase] = {
            'precision': round(precision, 4),
            'correct_count': correct,
            'total_matches': total,
        }

    return results


def compute_metrics_by_confidence(
    matches: List[MatchResult],
    buckets: List[Tuple[float, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics broken down by confidence bucket.

    Args:
        matches: List of MatchResult
        buckets: List of (low, high) tuples for confidence ranges

    Returns:
        Dict mapping bucket label -> metrics dict
    """
    if buckets is None:
        buckets = [
            (0.0, 0.3),
            (0.3, 0.5),
            (0.5, 0.7),
            (0.7, 0.85),
            (0.85, 1.01),
        ]

    results = {}
    for low, high in buckets:
        bucket_matches = [m for m in matches if low <= m.confidence < high]
        if bucket_matches:
            correct = sum(1 for m in bucket_matches if m.is_correct)
            total = len(bucket_matches)
            precision = correct / total if total > 0 else 0.0
            label = f'{low:.1f}-{high:.1f}'
            results[label] = {
                'precision': round(precision, 4),
                'correct_count': correct,
                'total_matches': total,
            }

    return results


def find_incorrect_matches(matches: List[MatchResult]) -> List[MatchResult]:
    """Return all incorrect matches for analysis."""
    return [m for m in matches if not m.is_correct]


def find_hardest_matches(
    matches: List[MatchResult],
    n: int = 10,
) -> List[MatchResult]:
    """Find the N lowest-confidence correct matches (hardest to match).

    These represent the most challenging reconciliation cases.
    """
    correct = [m for m in matches if m.is_correct]
    return sorted(correct, key=lambda m: m.confidence)[:n]


def generate_evaluation_report(matches: List[MatchResult]) -> str:
    """Generate a human-readable evaluation report.

    Returns:
        Formatted string with all metrics
    """
    overall = compute_metrics(matches)
    by_phase = compute_metrics_by_phase(matches)
    by_confidence = compute_metrics_by_confidence(matches)
    incorrect = find_incorrect_matches(matches)
    hardest = find_hardest_matches(matches)

    lines = []
    lines.append("=" * 60)
    lines.append("RECONCILIATION EVALUATION REPORT")
    lines.append("=" * 60)

    lines.append(f"\n{'OVERALL METRICS':^60}")
    lines.append("-" * 60)
    lines.append(f"  Precision:       {overall['precision']:.2%}")
    lines.append(f"  Recall:          {overall['recall']:.2%}")
    lines.append(f"  F1 Score:        {overall['f1']:.2%}")
    lines.append(f"  Correct matches: {overall['correct_count']} / {overall['total_matches']}")
    lines.append(f"  Expected total:  {overall['total_expected']}")

    lines.append(f"\n{'METRICS BY PHASE':^60}")
    lines.append("-" * 60)
    for phase, metrics in by_phase.items():
        lines.append(f"  {phase}:")
        lines.append(f"    Precision:  {metrics['precision']:.2%}")
        lines.append(f"    Correct:    {metrics['correct_count']} / {metrics['total_matches']}")

    lines.append(f"\n{'METRICS BY CONFIDENCE':^60}")
    lines.append("-" * 60)
    for bucket, metrics in by_confidence.items():
        lines.append(f"  [{bucket}]:")
        lines.append(f"    Precision:  {metrics['precision']:.2%}")
        lines.append(f"    Count:      {metrics['total_matches']}")

    if incorrect:
        lines.append(f"\n{'INCORRECT MATCHES ({len(incorrect)})':^60}")
        lines.append("-" * 60)
        for m in incorrect[:20]:
            lines.append(f"  {m.bank_id} -> {m.register_id} "
                        f"(conf={m.confidence:.3f}, phase={m.match_phase})")

    if hardest:
        lines.append(f"\n{'HARDEST CORRECT MATCHES':^60}")
        lines.append("-" * 60)
        for m in hardest:
            lines.append(f"  {m.bank_id} -> {m.register_id} "
                        f"(conf={m.confidence:.3f}, phase={m.match_phase})")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
