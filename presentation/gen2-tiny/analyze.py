import os
import pandas as pd

BASE = "/home/localuser/Documents/projects/snntorch2/snntorch/presentation/gen2-tiny/runs"


def analyze_parquet(path):
    """Return all metadata needed to build final plotting script."""
    try:
        df = pd.read_parquet(path)
        info = {
            "rows": len(df),
            "columns": list(df.columns),
        }
        # Try to extract step statistics if present
        if "Step" in df.columns:
            info["step_min"] = int(df["Step"].min())
            info["step_max"] = int(df["Step"].max())
        return info
    except Exception as e:
        return {"error": str(e)}


summary = {}

# Discover all runs
for run_name in sorted(os.listdir(BASE)):
    run_path = os.path.join(BASE, run_name)
    if not os.path.isdir(run_path):
        continue

    summary[run_name] = {}

    # Discover parquet shards inside each run
    for fname in sorted(os.listdir(run_path)):
        if not fname.endswith(".parquet"):
            continue

        full_path = os.path.join(run_path, fname)
        summary[run_name][fname] = analyze_parquet(full_path)

# Print all results cleanly
import json

print(json.dumps(summary, indent=4))
