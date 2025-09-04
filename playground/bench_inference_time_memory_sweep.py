# timing/memory sweep across batch sizes and channel counts

import time
import numpy as np
import torch
import matplotlib.pyplot as plt

import snntorch as snn
from snntorch._neurons.stateleaky import StateLeaky
from tqdm import tqdm


# Sweep configurations: (batch_size, channels)
SWEEP_CONFIGS = [
    (1, 5),
    (10, 20),
    (100, 100),
]

# Same timestep schedule as baseline
TIMESTEPS = np.logspace(1, 4.5, num=10, dtype=int)

device = "cuda:1"
torch.set_grad_enabled(False)


def get_peak_bytes(cuda_device):
    # Peak allocated bytes (global since process start or last reset)
    return torch.cuda.max_memory_allocated(cuda_device)


def get_cur_bytes(cuda_device):
    # Currently allocated bytes
    return torch.cuda.memory_allocated(cuda_device)


def bench_leaky(num_steps: int, batch_size: int, channels: int) -> float:
    lif = snn.Leaky(beta=0.9).to(device)
    # Scalar input per step, broadcasted internally
    x = torch.rand(num_steps, device=device)
    mem = torch.zeros(batch_size, channels, device=device)
    spk = torch.zeros(batch_size, channels, device=device)

    start_time = time.time()
    for step_idx in range(num_steps):
        spk, mem = lif(x[step_idx], mem=mem)
    end_time = time.time()
    return end_time - start_time


def bench_stateleaky(num_steps: int, batch_size: int, channels: int) -> float:
    lif = StateLeaky(beta=0.9, channels=channels).to(device)
    # Shape (T, B, C)
    input_tensor = torch.arange(
        1,
        num_steps * batch_size * channels + 1,
        device=device,
        dtype=torch.float32,
    ).view(num_steps, batch_size, channels)
    start_time = time.time()
    lif.forward(input_tensor)
    end_time = time.time()
    return end_time - start_time


with torch.no_grad():
    # Collect results per configuration
    results = []

    for batch_size, channels in SWEEP_CONFIGS:
        times_leaky, times_state = [], []
        mems_leaky, mems_state = [], []  # Δpeak MiB per run
        bases_leaky, bases_state = [], []  # baselines MiB
        peaks_leaky, peaks_state = [], []  # absolute peaks MiB

        for steps in tqdm(TIMESTEPS, desc=f"B{batch_size}-C{channels}"):
            n_runs = 2
            t1_runs, t2_runs = [], []
            m1_runs, m2_runs = [], []
            b1_runs, b2_runs = [], []
            p1_runs, p2_runs = [], []

            for _ in range(n_runs):
                # --- Leaky ---
                torch.cuda.synchronize()
                base1 = get_cur_bytes(device)
                t1 = bench_leaky(int(steps), batch_size, channels)
                torch.cuda.synchronize()
                peak1 = get_peak_bytes(device)
                d1 = max(0, peak1 - base1) / 1024**2

                t1_runs.append(t1)
                m1_runs.append(d1)
                b1_runs.append(base1 / 1024**2)
                p1_runs.append(peak1 / 1024**2)

                # --- StateLeaky ---
                torch.cuda.synchronize()
                base2 = get_cur_bytes(device)
                t2 = bench_stateleaky(int(steps), batch_size, channels)
                torch.cuda.synchronize()
                peak2 = get_peak_bytes(device)
                d2 = max(0, peak2 - base2) / 1024**2

                t2_runs.append(t2)
                m2_runs.append(d2)
                b2_runs.append(base2 / 1024**2)
                p2_runs.append(peak2 / 1024**2)

            times_leaky.append(np.mean(t1_runs))
            times_state.append(np.mean(t2_runs))
            mems_leaky.append(np.mean(m1_runs))
            mems_state.append(np.mean(m2_runs))
            bases_leaky.append(np.mean(b1_runs))
            peaks_leaky.append(np.mean(p1_runs))
            bases_state.append(np.mean(b2_runs))
            peaks_state.append(np.mean(p2_runs))

        results.append(
            {
                "batch_size": batch_size,
                "channels": channels,
                "times_leaky": np.array(times_leaky),
                "times_state": np.array(times_state),
                "mems_leaky": np.array(mems_leaky),
                "mems_state": np.array(mems_state),
                "bases_leaky": np.array(bases_leaky),
                "peaks_leaky": np.array(peaks_leaky),
                "bases_state": np.array(bases_state),
                "peaks_state": np.array(peaks_state),
            }
        )

    # ---- Plots: side-by-side ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=False)
    ax_time, ax_mem = axes

    # Choose colors per config, linestyle per neuron type
    cmap = plt.get_cmap("tab10")
    for idx, res in enumerate(results):
        color = cmap(idx % 10)
        label_suffix = f"B{res['batch_size']}-C{res['channels']}"

        ax_time.plot(
            TIMESTEPS,
            res["times_leaky"],
            linestyle="-",
            color=color,
            label=f"Leaky {label_suffix}",
        )
        ax_time.plot(
            TIMESTEPS,
            res["times_state"],
            linestyle="--",
            color=color,
            label=f"StateLeaky {label_suffix}",
        )

        ax_mem.plot(
            TIMESTEPS,
            res["mems_leaky"],
            linestyle="-",
            color=color,
            label=f"Leaky ΔMiB {label_suffix}",
        )
        ax_mem.plot(
            TIMESTEPS,
            res["mems_state"],
            linestyle="--",
            color=color,
            label=f"State ΔMiB {label_suffix}",
        )

    # Time subplot formatting
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.grid(True, which="both", ls="-", alpha=0.2)
    ax_time.set_xlabel("Number of Timesteps")
    ax_time.set_ylabel("Time (seconds)")
    ax_time.set_title("SNN Performance (Time) - Sweep")
    ax_time.legend(ncol=2, fontsize=8)

    # Memory subplot formatting
    ax_mem.set_xscale("log")
    ax_mem.set_yscale("log")
    ax_mem.grid(True, which="both", ls="-", alpha=0.2)
    ax_mem.set_xlabel("Number of Timesteps")
    ax_mem.set_ylabel("Δ Peak Allocated (MiB)")
    ax_mem.set_title("SNN Memory (Incremental Peak per Run) - Sweep")
    ax_mem.legend(ncol=2, fontsize=8)

    plt.tight_layout()

    # Print summaries per configuration
    for res in results:
        print(
            f"\nBenchmark Results (Time) for B{res['batch_size']} C{res['channels']}:"
        )
        print("Timesteps | Leaky (s) | StateLeaky (s) | Ratio (State/Leaky)")
        for i, steps in enumerate(TIMESTEPS):
            ratio = res["times_state"][i] / res["times_leaky"][i]
            print(
                f"{int(steps):9d} | {res['times_leaky'][i]:9.4f} | {res['times_state'][i]:13.4f} | {ratio:16.2f}"
            )

        print(
            f"\nBenchmark Results (Memory - Δpeak per run) for B{res['batch_size']} C{res['channels']}:"
        )
        print("Timesteps | Leaky ΔMiB | StateLeaky ΔMiB | Ratio (State/Leaky)")
        for i, steps in enumerate(TIMESTEPS):
            denom = res["mems_leaky"][i]
            ratio = (res["mems_state"][i] / denom) if denom else float("inf")
            print(
                f"{int(steps):9d} | {res['mems_leaky'][i]:10.4f} | {res['mems_state'][i]:16.4f} | {ratio:16.2f}"
            )

    fig.savefig("snn_performance_comparison.png", format="png")
