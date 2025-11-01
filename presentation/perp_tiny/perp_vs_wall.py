# plot_perplexity_vs_estimated_wallclock.py
"""
Usage:
  python plot_perplexity_vs_estimated_wallclock.py \
      [--history presentation/perp_tiny/wandb_export_2025-11-01T00_00_25.335-07_00.csv] \
      [--summary presentation/perp_tiny/wandb_export_2025-11-01T00_00_34.600-07_00.csv] \
      [--out_linear presentation/perp_tiny/perp_vs_time.pdf] \
      [--out_log presentation/perp_tiny/perp_vs_time_log.pdf] \
      [--also_runtime presentation/perp_tiny/runtime.pdf] \
      [--also_bestppl presentation/perp_tiny/best_ppl.pdf]

What it does:
- Reads the *history* CSV (per-step perplexities) and the *summary* CSV (runtime + step maxima).
- Detects runs from columns ending with " - ppl" (e.g., "stateleaky - ppl").
- Estimates wall-clock for each point as:
      elapsed_seconds ≈ step * (total_runtime / max_step)
  using runtime from "<run> - _runtime__MAX" (or "_runtime") and step max from "<run> - _step__MAX" (or "_step").
- Produces two overlays: linear y-scale and log y-scale, saved as PDFs by default.
- Optionally saves bar charts for total runtime and best (min) ppl, also as PDFs by default.
"""

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_step_col(df: pd.DataFrame) -> str:
    # Prefer common names
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
    raise ValueError(
        "Could not find a suitable step/batch column in history CSV."
    )


def ppl_runs_from_history(df_history: pd.DataFrame) -> List[str]:
    ppl_cols = [c for c in df_history.columns if c.lower().endswith(" - ppl")]
    if not ppl_cols:
        ppl_cols = [c for c in df_history.columns if "ppl" in c.lower()]
    runs = sorted({c.rsplit(" - ", 1)[0] for c in ppl_cols})
    if not runs:
        raise ValueError(
            "No runs found (expected columns ending with ' - ppl')."
        )
    return runs


def get_value(
    df_summary: pd.DataFrame, run: str, candidates: List[str]
) -> Optional[float]:
    for suffix in candidates:
        col = f"{run} - {suffix}"
        if col in df_summary.columns and pd.notna(df_summary[col]).any():
            return float(df_summary[col].iloc[0])
    return None


def compute_sec_per_step(
    df_summary: pd.DataFrame, run: str
) -> Optional[float]:
    total_runtime = get_value(df_summary, run, ["_runtime__MAX", "_runtime"])
    max_step = get_value(df_summary, run, ["_step__MAX", "_step"])
    if total_runtime is None or max_step is None or max_step <= 0:
        return None
    return float(total_runtime) / float(max_step)


def maybe_bar(ax, labels, values, ylabel, title):
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--history",
        required=False,
        default="presentation/perp_tiny/wandb_export_2025-11-01T00_00_25.335-07_00.csv",
        help="History CSV (per-step values).",
    )
    ap.add_argument(
        "--summary",
        required=False,
        default="presentation/perp_tiny/wandb_export_2025-11-01T00_00_34.600-07_00.csv",
        help="Summary CSV (runtime and step maxima).",
    )
    ap.add_argument(
        "--out_linear",
        required=False,
        default="presentation/perp_tiny/perp_vs_time.pdf",
        help="Output PDF for linear y-scale overlay.",
    )
    ap.add_argument(
        "--out_log",
        required=False,
        default="presentation/perp_tiny/perp_vs_time_log.pdf",
        help="Output PDF for log y-scale overlay.",
    )
    ap.add_argument(
        "--also_runtime",
        required=False,
        default="presentation/perp_tiny/runtime.pdf",
        help="(Optional) Save runtime comparison bar chart PDF.",
    )
    ap.add_argument(
        "--also_bestppl",
        required=False,
        default="presentation/perp_tiny/best_ppl.pdf",
        help="(Optional) Save best ppl comparison bar chart PDF.",
    )
    args = ap.parse_args()

    df_hist = pd.read_csv(args.history)
    df_sum = pd.read_csv(args.summary)

    step_col = find_step_col(df_hist)
    runs = ppl_runs_from_history(df_hist)

    # Compute avg seconds-per-step per run
    sec_per_step: Dict[str, float] = {}
    for r in runs:
        sps = compute_sec_per_step(df_sum, r)
        if sps is not None:
            sec_per_step[r] = sps
        else:
            print(
                f"[warn] Missing runtime/step info for run '{r}' in summary CSV; skipping in overlay."
            )

    if not sec_per_step:
        raise RuntimeError(
            "No runs had both runtime and step maxima in the summary CSV."
        )

    # # --- Overlay (linear y) ---
    # plt.figure()
    # for r in runs:
    #     ppl_col = f"{r} - ppl" if f"{r} - ppl" in df_hist.columns else None
    #     if (r in sec_per_step) and ppl_col is not None:
    #         secs = df_hist[step_col].astype(float) * sec_per_step[r]
    #         plt.plot(secs, df_hist[ppl_col].astype(float), label=r)
    # plt.xlabel("Estimated elapsed seconds (avg sec/step)")
    # plt.ylabel("Perplexity (ppl)")
    # plt.title("Perplexity vs Estimated Wall-Clock Time")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(args.out_linear)
    # print(f"Wrote: {args.out_linear}")

    # --- Overlay (log y) ---
    plt.figure()
    for r in runs:
        ppl_col = f"{r} - ppl" if f"{r} - ppl" in df_hist.columns else None
        if (r in sec_per_step) and ppl_col is not None:
            secs = df_hist[step_col].astype(float) * sec_per_step[r]
            plt.plot(secs, df_hist[ppl_col].astype(float), label=r)
    plt.xlabel("Elapsed seconds (avg sec/step)")
    plt.ylabel("Perplexity (log scale)")
    plt.yscale("log")
    plt.title("Perplexity vs Wall-Clock Time")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_log)
    print(f"Wrote: {args.out_log}")

    # # --- Runtime bar chart (optional; default path given) ---
    # if args.also_runtime:
    #     labels, values = [], []
    #     for r in runs:
    #         rt = get_value(df_sum, r, ["_runtime__MAX", "_runtime"])
    #         if rt is not None:
    #             labels.append(r)
    #             values.append(float(rt))
    #     if labels:
    #         fig, ax = plt.subplots()
    #         maybe_bar(
    #             ax,
    #             labels,
    #             values,
    #             "Total runtime (seconds)",
    #             "Total Wall-Clock Runtime by Run",
    #         )
    #         fig.tight_layout()
    #         fig.savefig(args.also_runtime)
    #         print(f"Wrote: {args.also_runtime}")
    #     else:
    #         print(
    #             "[warn] No runtime fields found in summary CSV for runtime chart."
    #         )

    # # --- Best ppl bar chart (optional; default path given) ---
    # if args.also_bestppl:
    #     labels, values = [], []
    #     for r in runs:
    #         best = get_value(df_sum, r, ["ppl__MIN", "ppl"])
    #         if best is not None:
    #             labels.append(r)
    #             values.append(float(best))
    #     if labels:
    #         fig, ax = plt.subplots()
    #         maybe_bar(
    #             ax,
    #             labels,
    #             values,
    #             "Best perplexity (min)",
    #             "Best Achieved Perplexity by Run",
    #         )
    #         fig.tight_layout()
    #         fig.savefig(args.also_bestppl)
    #         print(f"Wrote: {args.also_bestppl}")
    #     else:
    #         print(
    #             "[warn] No ppl or ppl__MIN fields found in summary CSV for best-ppl chart."
    #         )


if __name__ == "__main__":
    main()
