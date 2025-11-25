# plot_loss_single.py
"""
Usage:
  python plot_loss_single.py \
      [--csv presentation/loss_plotting/wandb_export_2025-11-01T23_16_59.256-07_00.csv] \
      [--out presentation/loss_plotting/loss.pdf] \
      [--x auto|step|time] \
      [--ycap NONE|<percent>|<float>] \
      [--logy] \
      [--include "regex"] \
      [--exclude "regex"]

Notes:
- Auto-detects multiple loss-like columns (e.g., '*loss*', '* - loss') and plots them all.
- Prefers 'Step' (or similar) on x; if missing, falls back to elapsed seconds from a runtime/timestamp column; if neither, uses index.
- Y-axis cap can be percentile or fixed number; applied across all selected series.
"""

import argparse
import re
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LOSS_KEYWORDS = ("loss",)


def find_step_col(df: pd.DataFrame) -> Optional[str]:
    for cand in [
        "Step",
        "step",
        "steps",
        "global_step",
        "iteration",
        "iter",
        "batch",
        "batches",
    ]:
        if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
            return cand
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            if len(s) > 2 and (s.diff().dropna() >= 0).all():
                return c
    return None


def find_time_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "_runtime",
        "runtime",
        "elapsed",
        "elapsed_seconds",
        "time",
        "wall_time",
        "wall_clock",
        "seconds",
        "minutes",
        "hours",
        "_timestamp",
    ]
    present = [
        c
        for c in candidates
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if present:
        order = {n: i for i, n in enumerate(candidates)}
        present.sort(key=lambda c: order.get(c, 999))
        return present[0]
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            if (
                len(s) > 2
                and (s.diff().dropna() >= 0).all()
                and s.max() > 100
                and s.min() >= 0
            ):
                return c
    return None


def to_elapsed_seconds(series: pd.Series, name_hint: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    hint = (name_hint or "").lower()
    if "minute" in hint:
        s = s * 60.0
    elif "hour" in hint:
        s = s * 3600.0
    if s.max() > 1e6:
        s = s - s.min()
    return s - s.min()


def candidate_loss_cols(
    df: pd.DataFrame, include_pat: Optional[str], exclude_pat: Optional[str]
) -> List[str]:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric = [
        c for c in numeric if not (c.endswith("__MIN") or c.endswith("__MAX"))
    ]

    def is_lossy(name: str) -> bool:
        n = name.lower()
        return any(k in n for k in LOSS_KEYWORDS) or n.endswith(" - loss")

    cols = [c for c in numeric if is_lossy(c)]

    if include_pat:
        inc = re.compile(include_pat, re.IGNORECASE)
        cols = [c for c in cols if inc.search(c)]
    if exclude_pat:
        exc = re.compile(exclude_pat, re.IGNORECASE)
        cols = [c for c in cols if not exc.search(c)]

    cols.sort(
        key=lambda n: (
            ("val" in n.lower()) or ("valid" in n.lower()),
            n.lower(),
        )
    )
    return cols


def clean_label(name: str) -> str:
    n = name.replace(" - loss", "")
    n = n.replace("_", " ").replace("/", " / ")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        required=False,
        default="presentation/loss_plotting/wandb_export_2025-11-01T23_16_59.256-07_00.csv",
        help="Input CSV with loss history.",
    )
    ap.add_argument(
        "--out",
        required=False,
        default="presentation/loss_plotting/loss.pdf",
        help="Output PDF file.",
    )
    ap.add_argument(
        "--x",
        choices=["auto", "step", "time"],
        default="auto",
        help="X-axis source: auto-detect, force step, or force time.",
    )
    ap.add_argument(
        "--ycap",
        default="NONE",
        help="NONE (no cap), a percentile like '99', or a fixed float like '50' for ymax.",
    )
    ap.add_argument(
        "--logy", action="store_true", help="Plot with logarithmic y-scale."
    )
    ap.add_argument(
        "--include",
        default=None,
        help="Optional regex to include only matching column names.",
    )
    ap.add_argument(
        "--exclude",
        default=None,
        help="Optional regex to exclude matching column names.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if len(df) < 2:
        raise SystemExit(
            "Not enough rows to plot a line chart (need a history CSV, not a single-row summary)."
        )

    # X detection
    step_col = find_step_col(df)
    time_col = find_time_col(df)

    if args.x == "step" or (args.x == "auto" and step_col is not None):
        x = pd.to_numeric(df[step_col], errors="coerce")
        x_label = step_col
    else:
        if time_col is None:
            x = pd.Series(np.arange(len(df)))
            x_label = "index"
        else:
            x = to_elapsed_seconds(df[time_col], time_col)
            x_label = f"{time_col} (elapsed seconds)"

    # Find all loss-like columns
    ycols = candidate_loss_cols(df, args.include, args.exclude)
    if not ycols:
        raise SystemExit(
            "No loss-like columns found (looked for names containing 'loss')."
        )

    # Optional y-cap
    ymax = None
    cap_arg = args.ycap.strip()
    if cap_arg.upper() != "NONE":
        try:
            ymax = float(cap_arg)
        except ValueError:
            perc = float(cap_arg)
            combined = pd.concat(
                [pd.to_numeric(df[c], errors="coerce") for c in ycols], axis=0
            )
            ymax = float(
                np.nanpercentile(combined.dropna().astype(float), perc)
            )

    # Plot
    plt.figure()
    for c in ycols:
        y = pd.to_numeric(df[c], errors="coerce")
        plt.plot(x, y, label=clean_label(c))

    plt.xlabel(x_label)
    plt.ylabel("Loss")
    if args.logy:
        plt.yscale("log")
    if ymax is not None:
        plt.ylim(bottom=0)
        plt.ylim(top=ymax)
    title_bits = []
    if args.x == "step" or (args.x == "auto" and step_col is not None):
        title_bits.append("vs step")
    elif time_col is not None:
        title_bits.append("vs elapsed time")
    else:
        title_bits.append("vs index")
    if ymax is not None:
        title_bits.append(f"(y≤{cap_arg})")
    plt.title("Loss " + " ".join(title_bits))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote: {args.out}")
    print(f"Plotted {len(ycols)} series:", ", ".join(ycols))


if __name__ == "__main__":
    main()
