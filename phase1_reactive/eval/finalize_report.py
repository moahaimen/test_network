#!/usr/bin/env python3
"""Assemble the combined Phase-1 markdown report from existing outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phase1_reactive.eval.report_builder import build_phase1_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the combined Phase-1 report")
    parser.add_argument("--base_dir", default="results/phase1_reactive")
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    summary_df = _read_csv(base / "eval" / "summary_all.csv")
    failure_df = _read_csv(base / "failures" / "summary_all.csv")
    generalization_df = _read_csv(base / "generalization" / "summary_all.csv")
    build_phase1_report(summary_df=summary_df, failure_df=failure_df, generalization_df=generalization_df, output_path=base / "report.md")
    print(f"Wrote combined report: {base / 'report.md'}")


if __name__ == "__main__":
    main()
