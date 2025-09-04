import torch
import time
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from snntorch._neurons.stateleaky import StateLeaky

BATCH_SIZE = 1
CHANNELS = 20
TIMESTEPS = np.logspace(1, 3, num=10, dtype=int)

device = "cuda"


def get_peak_memory(device):
    """Gets the peak GPU memory usage in MB."""
    return torch.cuda.max_memory_allocated(device) / 1024**2


def bench_type1(num_steps):
    lif = snn.Leaky(beta=0.9).to(device)
    x = torch.rand(num_steps).to(device)
    mem = torch.zeros(BATCH_SIZE, CHANNELS).to(device)
    spk = torch.zeros(BATCH_SIZE, CHANNELS).to(device)

    start_time = time.time()
    for step in range(num_steps):
        spk, mem = lif(x[step], mem=mem)
    end_time = time.time()

    return end_time - start_time


def bench_type2(timesteps):
    lif = StateLeaky(beta=0.9, channels=CHANNELS).to(device)
    input_ = (
        torch.arange(1, timesteps * BATCH_SIZE * CHANNELS + 1)
        .float()
        .view(timesteps, BATCH_SIZE, CHANNELS)
        .to(device)
    )

    start_time = time.time()
    lif.forward(input_)
    end_time = time.time()

    return end_time - start_time


with torch.no_grad():
    # run benchmarks
    times1 = []
    times2 = []
    mems1 = []
    mems2 = []
    for steps in TIMESTEPS:
        # run each benchmark multiple times and take average for more stable results
        n_runs = 2

        times_1_run_accumulated = []
        times_2_run_accumulated = []
        mems_1_run_accumulated = []
        mems_2_run_accumulated = []
        for _ in range(n_runs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.reset_max_memory_allocated(device)
            torch.cuda.reset_max_memory_cached(device)
            torch.cuda.reset_accumulated_memory_stats(device)
            time1 = bench_type1(steps)
            torch.cuda.synchronize()
            mem1 = get_peak_memory(device)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.reset_max_memory_allocated(device)
            torch.cuda.reset_max_memory_cached(device)
            torch.cuda.reset_accumulated_memory_stats(device)
            time2 = bench_type2(steps)
            torch.cuda.synchronize()
            mem2 = get_peak_memory(device)
            times_1_run_accumulated.append(time1)
            times_2_run_accumulated.append(time2)
            mems_1_run_accumulated.append(mem1)
            mems_2_run_accumulated.append(mem2)

        times1.append(np.mean(times_1_run_accumulated))
        times2.append(np.mean(times_2_run_accumulated))
        mems1.append(np.mean(mems_1_run_accumulated))
        mems2.append(np.mean(mems_2_run_accumulated))

    # create the plot for time
    plt.figure(figsize=(10, 6))
    plt.plot(TIMESTEPS, times1, "b-", label="Type 1 (Leaky)")
    plt.plot(TIMESTEPS, times2, "r-", label="Type 2 (StateLeaky)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Time (seconds)")
    plt.title("SNN Performance Comparison")
    plt.legend()

    # create the plot for memory
    plt.figure(figsize=(10, 6))
    plt.plot(TIMESTEPS, mems1, "b-", label="Type 1 (Leaky)")
    plt.plot(TIMESTEPS, mems2, "r-", label="Type 2 (StateLeaky)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Memory (MB)")
    plt.title("SNN Memory Comparison")
    plt.legend()

    print("Benchmark Results:")
    print("\nTimesteps  |  Leaky (s)  |  Linear Leaky (s)  |  Ratio (T2/T1)")
    print("-" * 55)
    for i, steps in enumerate(TIMESTEPS):
        ratio = times2[i] / times1[i]
        print(
            f"{steps:9d} | {times1[i]:10.4f} | {times2[i]:10.4f} | {ratio:10.2f}"
        )

    plt.savefig("snn_performance_comparison_andrew.png", format="png")
