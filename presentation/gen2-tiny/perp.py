import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Load CSV
# ============================================================
csv_path = (
    "presentation/gen2-tiny/wandb_export_2025-11-23T00_39_37.709-08_00.csv"
)
df = pd.read_csv(csv_path)

# Column names
step_col = "Step"
ppl_gen2 = "gen2-spiking - ppl"  # RED
ppl_state = "stateleaky-tuned - ppl"  # BLUE

# Extract
steps = df[step_col].values
gen2 = df[ppl_gen2].values
state = df[ppl_state].values

# Avoid log issues
gen2 = np.clip(gen2, 1e-12, None)
state = np.clip(state, 1e-12, None)

# Both must be valid
valid = ~np.isnan(gen2) & ~np.isnan(state)
steps = steps[valid]
gen2 = gen2[valid]
state = state[valid]

# Clip x-range
max_x = 600_000
mask = steps <= max_x
steps = steps[mask]
gen2 = gen2[mask]
state = state[mask]

# ============================================================
# IEEE-style global parameters
# ============================================================
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

fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)

# ============================================================
# Plot curves (RAW, no smoothing)
# ============================================================
ax.plot(
    steps,
    state,
    color="tab:blue",
    linewidth=1.5,
    alpha=0.8,
    label="stateleaky",
)

ax.plot(
    steps,
    gen2,
    color="red",
    linewidth=1.5,
    alpha=0.8,
    label="gen2",
)

# ============================================================
# Axes + formatting
# ============================================================
ax.set_yscale("log")  # remove if you want linear
ax.set_title("Perplexity vs Step")
ax.set_xlabel("Step")
ax.set_ylabel("Perplexity")

# Grid style matching your examples
ax.grid(True, color="#cccccc", alpha=0.4, linewidth=0.8)

# Abbreviated x-ticks
xticks = [0, 200_000, 400_000, 600_000]
ax.set_xticks(xticks)
ax.set_xticklabels(["0", "200k", "400k", "600k"])

# Legend styling like your examples
leg = ax.legend(frameon=True)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_alpha(0.9)
leg.get_frame().set_edgecolor("#cccccc")

plt.tight_layout()
plt.savefig("presentation/gen2-tiny/perp.pdf")
