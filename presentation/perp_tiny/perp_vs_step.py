# plot_perplexity_vs_step_truncated.py
"""
Usage:
  python plot_perplexity_vs_step_truncated.py \
      --history /path/to/wandb_export_history.csv \
      --out /path/to/perplexity_vs_step_truncated.png \
      [--ycap 99]       # percentile cap (default=99). Or pass a float (e.g., 250) for a fixed ymax.

This script:
- Reads the *history* CSV that has per-step perplexities.
- Detects the step column (defaults to 'Step' if present).
- Plots all columns that end with " - ppl" vs step.
- Truncates the y-axis either at a percentile (default P99) or a fixed ymax.
"""

import argparse
import math
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_step_col(df: pd.DataFrame) -> str:
    # Prefer a canonical name if present
    for cand in [
        "Step",
        "step",
        "steps",
        "global_step",
        "iteration",
        "iter",
        "batch",
    ]:
        if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
            return cand
    # Fallback: first numeric, monotonic non-decreasing column
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            series = df[c].dropna()
            if len(series) > 2 and (series.diff().dropna() >= 0).all():
                return c
    # Final fallback: first numeric column
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("Could not find a suitable step/batch column.")


def collect_ppl_columns(df: pd.DataFrame) -> List[str]:
    cols = [
        c
        for c in df.columns
        if c.lower().endswith(" - ppl")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        # try a looser match
        cols = [
            c
            for c in df.columns
            if "ppl" in c.lower() and pd.api.types.is_numeric_dtype(df[c])
        ]
    if not cols:
        raise ValueError(
            "No perplexity columns found (expected headers ending with ' - ppl')."
        )
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--history",
        required=False,
        help="History CSV (per-step values).",
        default="presentation/perp_tiny/wandb_export_2025-11-01T00_00_25.335-07_00.csv",
    )
    ap.add_argument(
        "--out",
        required=False,
        help="Output PNG path.",
        default="presentation/perp_tiny/perp_vs_step.pdf",
    )
    ap.add_argument(
        "--ycap",
        default="99",
        help="Percentile (e.g., '99') or fixed float (e.g., '250') for ymax.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.history)

    step_col = find_step_col(df)
    ppl_cols = collect_ppl_columns(df)

    # Compute cap
    ycap: float
    try:
        # if user passed a float, use as absolute cap
        ycap = float(args.ycap)
        use_percentile = False
    except ValueError:
        # if not a float, treat as percentile
        use_percentile = True
        perc = float(args.ycap)
        vals = pd.concat(
            [df[c].dropna().astype(float) for c in ppl_cols], axis=0
        )
        ycap = float(np.nanpercentile(vals, perc))

    plt.figure()
    for c in ppl_cols:
        plt.plot(df[step_col], df[c], label=c.replace(" - ppl", ""))

    plt.xlabel(step_col)
    plt.ylabel("Perplexity (ppl)")
    plt.ylim(0, ycap)
    title_suffix = f"P{args.ycap}" if use_percentile else f"{ycap:g}"
    plt.title(f"Perplexity vs Batch/Step (y truncated at {title_suffix})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
