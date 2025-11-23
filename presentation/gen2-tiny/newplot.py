import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = "/home/localuser/Documents/projects/snntorch2/snntorch/presentation/gen2-tiny/runs"


def load_run(run_path):
    dfs = []
    for fname in sorted(os.listdir(run_path)):
        if fname.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(run_path, fname))
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("_step").reset_index(drop=True)
    return df


# ------------------------------------------------------------
# Correct run assignments
# ------------------------------------------------------------
gen2_df = load_run(os.path.join(BASE, "run1"))  # orange
stateleaky_df = load_run(os.path.join(BASE, "run2"))  # blue

# ------------------------------------------------------------
# Shared range + 600k clip
# ------------------------------------------------------------
max_step = min(gen2_df["_step"].max(), stateleaky_df["_step"].max())
clip_max = 600_000

gen2_df = gen2_df[
    (gen2_df["_step"] <= max_step) & (gen2_df["_step"] <= clip_max)
]
stateleaky_df = stateleaky_df[
    (stateleaky_df["_step"] <= max_step) & (stateleaky_df["_step"] <= clip_max)
]

# ------------------------------------------------------------
# Plot styling (COLOR + LABEL ONLY — scale unchanged)
# ------------------------------------------------------------
plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
)

fig, ax = plt.subplots(figsize=(7, 6), dpi=200)

# Colors match your target example (tab:blue, tab:orange)
ax.plot(
    stateleaky_df["_step"],
    stateleaky_df["ppl"],
    color="tab:blue",
    alpha=0.85,
    linewidth=1.5,
    label="StateLeaky",
)

ax.plot(
    gen2_df["_step"],
    gen2_df["ppl"],
    color="#2ca02c",
    alpha=0.85,
    linewidth=1.5,
    label="AssociativeLeaky",
)

# ------------------------------------------------------------
# ***THE EXACT SAME SCALE WE TUNED EARLIER***
# ------------------------------------------------------------
yticks = [400, 300, 200, 100, 80, 60, 40, 30, 20, 10, 8, 7]

ax.set_yscale("log")
ax.set_ylim(7, 400)
ax.set_yticks(yticks)
ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

# X-axis stays the same
ax.set_xticks([0, 200_000, 400_000, 600_000])
ax.set_xticklabels(["0", "200k", "400k", "600k"])

# Axis labels (added, not changed scale)
ax.set_xlabel("Step")
ax.set_ylabel("Perplexity")

# Nice clean grid
ax.grid(True, color="#cccccc", alpha=0.4, linewidth=0.8)

# Legend style
leg = ax.legend(frameon=True)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("#cccccc")
leg.get_frame().set_alpha(0.9)

plt.tight_layout()
plt.savefig("presentation/gen2-tiny/newplot.pdf")
