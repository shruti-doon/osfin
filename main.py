#!/usr/bin/env python3
"""CLI entry point for the Financial Reconciliation System.

Usage:
    python main.py run              # Full pipeline
    python main.py --phase unique   # Only unique-amount matching
    python main.py --phase ml       # Only ML matching (assumes unique done)
    python main.py --report         # Generate analysis report
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from src.reconciler import ReconciliationPipeline
from src.evaluator import (
    compute_metrics,
    compute_metrics_by_phase,
    compute_metrics_by_confidence,
    find_incorrect_matches,
    find_hardest_matches,
)

console = Console()


def display_matches_table(matches, bank_df, register_df, title="Matches", max_rows=30):
    """Display matches in a rich table."""
    table = Table(title=title, show_lines=True)
    table.add_column("Bank ID", style="cyan", width=8)
    table.add_column("Register ID", style="green", width=10)
    table.add_column("Bank Description", width=22)
    table.add_column("Register Description", width=22)
    table.add_column("Amount", justify="right", width=10)
    table.add_column("Confidence", justify="right", style="yellow", width=10)
    table.add_column("Phase", width=14)
    table.add_column("Correct", justify="center", width=8)

    for m in matches[:max_rows]:
        bank_row = bank_df[bank_df['transaction_id'] == m.bank_id]
        reg_row = register_df[register_df['transaction_id'] == m.register_id]

        bank_desc = bank_row.iloc[0]['description'] if len(bank_row) > 0 else '?'
        reg_desc = reg_row.iloc[0]['description'] if len(reg_row) > 0 else '?'
        amount = f"${bank_row.iloc[0]['amount']:.2f}" if len(bank_row) > 0 else '?'

        correct_style = "[green]✓[/green]" if m.is_correct else "[red]✗[/red]"

        conf_color = "green" if m.confidence >= 0.85 else "yellow" if m.confidence >= 0.5 else "red"

        table.add_row(
            m.bank_id,
            m.register_id,
            bank_desc[:22],
            reg_desc[:22],
            amount,
            f"[{conf_color}]{m.confidence:.3f}[/{conf_color}]",
            m.match_phase,
            correct_style,
        )

    if len(matches) > max_rows:
        table.add_row("...", "...", f"({len(matches) - max_rows} more)", "", "", "", "", "")

    console.print(table)


def display_metrics(metrics, title="Overall Metrics"):
    """Display metrics in a rich panel."""
    panel_text = (
        f"[bold]Precision:[/bold] {metrics['precision']:.2%}\n"
        f"[bold]Recall:[/bold]    {metrics['recall']:.2%}\n"
        f"[bold]F1 Score:[/bold]  {metrics['f1']:.2%}\n"
        f"[bold]Correct:[/bold]   {metrics['correct_count']} / {metrics['total_matches']}\n"
        f"[bold]Expected:[/bold]  {metrics['total_expected']}"
    )
    console.print(Panel(panel_text, title=title, border_style="green"))


def display_phase_metrics(phase_metrics):
    """Display per-phase metrics."""
    table = Table(title="Metrics by Phase")
    table.add_column("Phase", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Total", justify="right")

    for phase, metrics in phase_metrics.items():
        table.add_row(
            phase,
            f"{metrics['precision']:.2%}",
            str(metrics['correct_count']),
            str(metrics['total_matches']),
        )

    console.print(table)


def display_confidence_metrics(conf_metrics):
    """Display per-confidence-bucket metrics."""
    table = Table(title="Metrics by Confidence Bucket")
    table.add_column("Bucket", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Count", justify="right")

    for bucket, metrics in conf_metrics.items():
        table.add_row(
            bucket,
            f"{metrics['precision']:.2%}",
            str(metrics['total_matches']),
        )

    console.print(table)


def run_pipeline(args):
    """Run the full reconciliation pipeline."""
    base_dir = args.data_dir or str(Path(__file__).parent)
    use_st = not args.no_embeddings

    pipeline = ReconciliationPipeline(
        base_dir=base_dir,
        use_sentence_transformers=use_st,
        svd_components=args.svd_components,
        max_iterations=args.iterations,
    )

    console.print(Panel(
        "[bold blue]Financial Reconciliation System[/bold blue]\n"
        "Matching bank statements with check register using hybrid ML",
        border_style="blue",
    ))

    # Phase selection
    phase = args.phase or 'all'

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load data
        task = progress.add_task("Loading and preprocessing data...", total=None)
        pipeline.load_and_preprocess()
        progress.update(task, description=f"[green]✓ Loaded {len(pipeline.bank_df)} bank + {len(pipeline.register_df)} register transactions")

        if phase in ('all', 'unique'):
            # Phase 1
            task = progress.add_task("Running unique-amount matching...", total=None)
            pipeline.run_unique_matching()
            n_unique = len(pipeline.unique_matches)
            correct_unique = sum(1 for m in pipeline.unique_matches if m.is_correct)
            progress.update(task, description=f"[green]✓ Phase 1: {n_unique} unique matches ({correct_unique} correct)")

        if phase in ('all', 'ml'):
            # Phase 2
            task = progress.add_task("Running ML-based matching...", total=None)
            pipeline.run_ml_matching()
            n_ml = len(pipeline.ml_matches)
            correct_ml = sum(1 for m in pipeline.ml_matches if m.is_correct)
            progress.update(task, description=f"[green]✓ Phase 2: {n_ml} ML matches ({correct_ml} correct)")

    # Combine matches
    pipeline.all_matches = pipeline.unique_matches + pipeline.ml_matches

    console.print()

    # Display results
    if args.show_matches:
        display_matches_table(
            pipeline.all_matches,
            pipeline.bank_df,
            pipeline.register_df,
            title="All Matched Transactions",
            max_rows=args.max_rows,
        )
        console.print()

    # Metrics
    metrics = compute_metrics(pipeline.all_matches)
    display_metrics(metrics)
    console.print()

    phase_metrics = compute_metrics_by_phase(pipeline.all_matches)
    display_phase_metrics(phase_metrics)
    console.print()

    conf_metrics = compute_metrics_by_confidence(pipeline.all_matches)
    display_confidence_metrics(conf_metrics)
    console.print()

    # Incorrect matches
    incorrect = find_incorrect_matches(pipeline.all_matches)
    if incorrect:
        console.print(f"[red]Incorrect matches: {len(incorrect)}[/red]")
        display_matches_table(
            incorrect,
            pipeline.bank_df,
            pipeline.register_df,
            title="Incorrect Matches",
            max_rows=20,
        )
        console.print()

    # Hardest correct matches
    hardest = find_hardest_matches(pipeline.all_matches, n=10)
    if hardest:
        display_matches_table(
            hardest,
            pipeline.bank_df,
            pipeline.register_df,
            title="Hardest Correct Matches (lowest confidence)",
            max_rows=10,
        )

    # Timing
    console.print(Panel(
        "\n".join(f"  {k}: {v:.2f}s" for k, v in pipeline.timings.items()),
        title="Timing",
        border_style="dim",
    ))

    # Export
    if args.export:
        export_path = Path(base_dir) / args.export
        pipeline.export_matches_csv(str(export_path))
        console.print(f"\n[green]Exported matches to {export_path}[/green]")

    return pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Financial Reconciliation System — Match bank statements with check register",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run                         # Full pipeline
  python main.py run --phase unique          # Only unique-amount matching
  python main.py run --phase ml              # Only ML matching
  python main.py run --show-matches          # Show match table
  python main.py run --export results.csv    # Export to CSV
  python main.py run --no-embeddings         # Use TF-IDF fallback instead of sentence-transformers
        """,
    )

    subparsers = parser.add_subparsers(dest='command')
    run_parser = subparsers.add_parser('run', help='Run the reconciliation pipeline')

    run_parser.add_argument('--phase', choices=['all', 'unique', 'ml'], default='all',
                            help='Which matching phase to run')
    run_parser.add_argument('--data-dir', type=str, default=None,
                            help='Path to data directory')
    run_parser.add_argument('--show-matches', action='store_true',
                            help='Display full match table')
    run_parser.add_argument('--max-rows', type=int, default=30,
                            help='Max rows to show in tables')
    run_parser.add_argument('--export', type=str, default=None,
                            help='Export matches to CSV file')
    run_parser.add_argument('--no-embeddings', action='store_true',
                            help='Skip sentence-transformers, use TF-IDF fallback')
    run_parser.add_argument('--svd-components', type=int, default=40,
                            help='Number of SVD components')
    run_parser.add_argument('--iterations', type=int, default=3,
                            help='Max iterative matching rounds')

    args = parser.parse_args()

    if args.command == 'run':
        run_pipeline(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
