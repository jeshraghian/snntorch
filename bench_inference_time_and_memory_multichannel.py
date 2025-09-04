# timing plot for StateLeaky vs Leaky (with baseline/peak/Δpeak reporting)

import torch
import time
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from snntorch._neurons.stateleaky import StateLeaky

BATCH_SIZE = 10
CHANNELS = 20
BETA = 0.9
TIMESTEPS = np.logspace(1, 5.85, num=10, dtype=int)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f'Device {device.type}')

def get_peak_memory(device):
    """Gets the peak GPU memory usage in MB."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 1024**2   
    elif device.type == 'mps':
        return float(torch.mps.current_allocated_memory()) / (1024 ** 2)
    return 0

def sync_gpu(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    return


# Benching constant beta of 0.9
def bench_type1(timesteps):
    lif = StateLeaky(beta=BETA, channels=CHANNELS, learn_beta=False).to(device)
    input_ = torch.arange(
        1,
        timesteps * BATCH_SIZE * CHANNELS + 1,
        device=device,
        dtype=torch.float32,
    ).view(timesteps, BATCH_SIZE, CHANNELS)
    start_time = time.time()
    lif.forward(input_)
    end_time = time.time()
    return end_time - start_time


# Benching learnable beta with initial beta of 0.9
def bench_type2(timesteps):
    lif = StateLeaky(beta=BETA, channels=CHANNELS, learn_beta=True).to(device)
    input_ = torch.arange(
        1,
        timesteps * BATCH_SIZE * CHANNELS + 1,
        device=device,
        dtype=torch.float32,
    ).view(timesteps, BATCH_SIZE, CHANNELS)
    start_time = time.time()
    lif.forward(input_)
    end_time = time.time()
    return end_time - start_time


with torch.no_grad():
    times1 = []
    times2 = []
    mems1 = []
    mems2 = []

    for steps in tqdm(TIMESTEPS):
        n_runs = 2
        times_1_run_accumulated = []
        times_2_run_accumulated = []
        mems_1_run_accumulated = []
        mems_2_run_accumulated = []

        for _ in range(n_runs):
            # --- Type 1 ---
            sync_gpu(device)
            time1 = bench_type1(steps)
            sync_gpu(device)
            mem1 = get_peak_memory(device)

            # --- Type 2 ---
            sync_gpu(device)
            time2 = bench_type2(steps)
            sync_gpu(device)
            mem2 = get_peak_memory(device)

            times_1_run_accumulated.append(time1)
            times_2_run_accumulated.append(time2)
            mems_1_run_accumulated.append(mem1)
            mems_2_run_accumulated.append(mem2)

        times1.append(np.mean(times_1_run_accumulated))
        times2.append(np.mean(times_2_run_accumulated))
        mems1.append(np.mean(mems_1_run_accumulated))
        mems2.append(np.mean(mems_2_run_accumulated))

    # ---- Plots: side-by-side ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=False)
    ax_time, ax_mem = axes

    # Time subplot
    ax_time.plot(TIMESTEPS, times1, "b-", label=f'Type 1 (StateLeaky)')
    ax_time.plot(TIMESTEPS, times2, "r-", label=f'Type 2 (StateLeaky) with learnable beta')
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.grid(True, which="both", ls="-", alpha=0.2)
    ax_time.set_xlabel("Number of Timesteps")
    ax_time.set_ylabel("Time (seconds)")
    ax_time.set_title(f'SNN Performance (Time) with beta={BETA}')
    ax_time.legend()

    # Memory subplot (Δpeak per run)
    ax_mem.plot(TIMESTEPS, mems1, "b-", label=f'Type 1 mem Δpeak')
    ax_mem.plot(TIMESTEPS, mems2, "r-", label=f'Type 2 mem Δpeak with learnable beta')
    ax_mem.set_xscale("log")
    ax_mem.set_yscale("log")
    ax_mem.grid(True, which="both", ls="-", alpha=0.2)
    ax_mem.set_xlabel("Number of Timesteps")
    ax_mem.set_ylabel("Δ Peak Allocated (MiB)")
    ax_mem.set_title(f"SNN Memory (Incremental Peak per Run) with beta={BETA}")
    ax_mem.legend()

    plt.tight_layout()

    # ---- Optional: print baselines/peaks to verify behavior ----
    print("\nBenchmark Results (Time):")
    print("Timesteps | StateLeaky (s) | StateLeaky with learnable beta (s) | Ratio (T2/T1)")
    for i, steps in enumerate(TIMESTEPS):
        print(f"{int(steps):9d} | {times1[i]:9.4f} | {times2[i]:13.4f} | {times2[i]/times1[i]:10.2f}")

    print("\nBenchmark Results (Memory - Δpeak per run):")
    print("Timesteps | StateLeaky ΔMiB | StateLeaky with learnable beta ΔMiB | Ratio (T2/T1)")
    for i, steps in enumerate(TIMESTEPS):
        r = mems2[i] / mems1[i] if mems1[i] else float("inf")
        print(f"{int(steps):9d} | {mems1[i]:10.4f} | {mems2[i]:16.4f} | {r:10.2f}")

    fig.savefig("snn_performance_comparison.png", format="png")
