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
TIMESTEPS = np.logspace(1, 5.85, num=10, dtype=int)

device = "cuda:1"
torch.set_grad_enabled(False)  # no autograd in this benchmark


def get_peak_bytes(device):
    # Peak allocated bytes (global since process start or last reset)
    return torch.cuda.max_memory_allocated(device)


def get_cur_bytes(device):
    # Currently allocated bytes
    return torch.cuda.memory_allocated(device)


def bench_type1(num_steps):
    lif = snn.Leaky(beta=0.9).to(device)
    x = torch.rand(num_steps, device=device)
    mem = torch.zeros(BATCH_SIZE, CHANNELS, device=device)
    spk = torch.zeros(BATCH_SIZE, CHANNELS, device=device)

    start_time = time.time()
    for step in range(num_steps):
        spk, mem = lif(x[step], mem=mem)
    end_time = time.time()
    return end_time - start_time


def bench_type2(timesteps):
    lif = StateLeaky(beta=0.9, channels=CHANNELS).to(device)
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
    times1, times2 = [], []
    mems1, mems2 = [], []  # Δpeak in MiB
    bases1, bases2 = [], []  # baselines in MiB (for visibility)
    peaks1, peaks2 = [], []  # absolute peaks in MiB

    for steps in tqdm(TIMESTEPS):
        n_runs = 2
        t1_runs, t2_runs = [], []
        m1_runs, m2_runs = [], []
        b1_runs, b2_runs = [], []
        p1_runs, p2_runs = [], []

        for _ in range(n_runs):
            # --- Type 1 ---
            torch.cuda.synchronize()
            base1 = get_cur_bytes(device)  # baseline now
            t1 = bench_type1(steps)
            torch.cuda.synchronize()
            peak1 = get_peak_bytes(device)  # global peak
            d1 = max(0, peak1 - base1) / 1024**2  # Δpeak MiB

            t1_runs.append(t1)
            m1_runs.append(d1)
            b1_runs.append(base1 / 1024**2)
            p1_runs.append(peak1 / 1024**2)

            # --- Type 2 ---
            torch.cuda.synchronize()
            base2 = get_cur_bytes(device)
            t2 = bench_type2(steps)
            torch.cuda.synchronize()
            peak2 = get_peak_bytes(device)
            d2 = max(0, peak2 - base2) / 1024**2

            t2_runs.append(t2)
            m2_runs.append(d2)
            b2_runs.append(base2 / 1024**2)
            p2_runs.append(peak2 / 1024**2)

        times1.append(np.mean(t1_runs))
        times2.append(np.mean(t2_runs))
        mems1.append(np.mean(m1_runs))
        mems2.append(np.mean(m2_runs))
        bases1.append(np.mean(b1_runs))
        peaks1.append(np.mean(p1_runs))
        bases2.append(np.mean(b2_runs))
        peaks2.append(np.mean(p2_runs))

    # ---- Plots: side-by-side ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=False)
    ax_time, ax_mem = axes

    # Time subplot
    ax_time.plot(TIMESTEPS, times1, "b-", label="Type 1 (Leaky)")
    ax_time.plot(TIMESTEPS, times2, "r-", label="Type 2 (StateLeaky)")
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.grid(True, which="both", ls="-", alpha=0.2)
    ax_time.set_xlabel("Number of Timesteps")
    ax_time.set_ylabel("Time (seconds)")
    ax_time.set_title("SNN Performance (Time)")
    ax_time.legend()

    # Memory subplot (Δpeak per run)
    ax_mem.plot(TIMESTEPS, mems1, "b-", label="Type 1 mem Δpeak")
    ax_mem.plot(TIMESTEPS, mems2, "r-", label="Type 2 mem Δpeak")
    ax_mem.set_xscale("log")
    ax_mem.set_yscale("log")
    ax_mem.grid(True, which="both", ls="-", alpha=0.2)
    ax_mem.set_xlabel("Number of Timesteps")
    ax_mem.set_ylabel("Δ Peak Allocated (MiB)")
    ax_mem.set_title("SNN Memory (Incremental Peak per Run)")
    ax_mem.legend()

    plt.tight_layout()

    # ---- Optional: print baselines/peaks to verify behavior ----
    print("\nBenchmark Results (Time):")
    print("Timesteps | Leaky (s) | StateLeaky (s) | Ratio (T2/T1)")
    for i, steps in enumerate(TIMESTEPS):
        print(
            f"{int(steps):9d} | {times1[i]:9.4f} | {times2[i]:13.4f} | {times2[i]/times1[i]:10.2f}"
        )

    print("\nBenchmark Results (Memory - Δpeak per run):")
    print("Timesteps | Leaky ΔMiB | StateLeaky ΔMiB | Ratio (T2/T1)")
    for i, steps in enumerate(TIMESTEPS):
        r = mems2[i] / mems1[i] if mems1[i] else float("inf")
        print(
            f"{int(steps):9d} | {mems1[i]:10.4f} | {mems2[i]:16.4f} | {r:10.2f}"
        )

    print("\nBaselines and absolute peaks (MiB) for sanity:")
    print("Timesteps | Base1 | Peak1 || Base2 | Peak2")
    for i, steps in enumerate(TIMESTEPS):
        print(
            f"{int(steps):9d} | {bases1[i]:6.1f} | {peaks1[i]:6.1f} || {bases2[i]:6.1f} | {peaks2[i]:6.1f}"
        )

    fig.savefig("snn_performance_comparison.png", format="png")
