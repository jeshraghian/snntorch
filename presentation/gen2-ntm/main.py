import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ================================
# Load + sort data
# ================================
csv_path = (
    "presentation/gen2-ntm/wandb_export_2025-11-23T03_09_27.976-08_00.csv"
)
df = pd.read_csv(csv_path)

df = df.sort_values("Step").reset_index(drop=True)

# Column groups
models = {
    "stateleaky": ["1e-4", "3e-4", "5e-4"],
    "gen2-assoc-eval": ["1e-4", "3e-4", "5e-4"],
}

# Line colors
state_colors = {
    "1e-4": "#6baed6",
    "3e-4": "#3182bd",
    "5e-4": "#08519c",
}

gen2_colors = {
    "1e-4": "#74c476",
    "3e-4": "#31a354",
    "5e-4": "#006d2c",
}

# ================================
# Matplotlib styling (IEEE-like)
# ================================
plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 13,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
)

fig, ax = plt.subplots(figsize=(9, 7), dpi=200)

# ================================
# Plot curves with visible shading
# ================================
for model, lrs in models.items():
    for lr in lrs:

        base = f"{model}-{lr} - eval_bit_acc"

        steps = df["Step"].values
        main = df[base].values
        lo = df[f"{base}__MIN"].values
        hi = df[f"{base}__MAX"].values

        # Choose color sets
        if model == "stateleaky":
            line_color = state_colors[lr]
            label = f"StateLeaky (LR={lr})"
        else:
            line_color = gen2_colors[lr]
            label = f"AssociativeLeaky (LR={lr})"

        # Convert line color to RGBA for shading
        rgba = list(mcolors.to_rgba(line_color))
        rgba[3] = 0.28  # strong but clean visibility

        # Shaded band (behind line)
        ax.fill_between(steps, lo, hi, color=rgba, linewidth=0, zorder=1)

        # Main line
        ax.plot(
            steps,
            main,
            color=line_color,
            linewidth=2.2,
            alpha=0.95,
            label=label,
            zorder=3,
        )

# ================================
# Axes formatting
# ================================
ax.set_xlabel("Step")
ax.set_ylabel("Bit Accuracy")
ax.set_title("Bit Accuracy vs Step (All Learning Rates)")

# Manual x-axis ticks (from your example)
xticks = [0, 5000, 10000]
ax.set_xticks(xticks)
ax.set_xticklabels(["0", "5k", "10k"])

ax.grid(True, color="#cccccc", alpha=0.4, linewidth=0.8)

# Legend formatting
leg = ax.legend(frameon=True, ncol=2)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("#cccccc")
leg.get_frame().set_alpha(0.90)

plt.tight_layout()
plt.savefig("presentation/gen2-ntm/main.pdf")
print("Saved: presentation/gen2-ntm/main.pdf")
