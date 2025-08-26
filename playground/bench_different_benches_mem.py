# timing plot for StateLeaky vs Leaky

import torch
import time
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from snntorch._neurons.stateleaky import StateLeaky

BATCH_SIZE = 20
CHANNELS = 20

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
    input_ = torch.arange(1, timesteps * BATCH_SIZE * CHANNELS + 1).float().view(timesteps, BATCH_SIZE, CHANNELS).to(device)
    
    start_time = time.time()
    lif.forward(input_)
    end_time = time.time()
    
    return end_time - start_time

def sync_gpu(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    return

# Define timesteps on log scale
timesteps = np.logspace(1, 5, num=10, dtype=int)

# Run benchmarks
times1 = []
times2 = []
mems1 = []
mems2 = []
for steps in timesteps:
    # Run each benchmark multiple times and take average for more stable results
    n_runs = 2
    
    times_1_run_accumulated = []
    times_2_run_accumulated = []
    mems_1_run_accumulated = []
    mems_2_run_accumulated = []
    for _ in range(n_runs):
        sync_gpu(device)
        # torch.cuda.reset_peak_memory_stats(device)
        time1 = bench_type1(steps)
        sync_gpu(device)
        mem1 = get_peak_memory(device)

        sync_gpu(device)
        # torch.cuda.reset_peak_memory_stats(device)
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

# Create the plot for time
plt.figure(figsize=(10, 6))
plt.plot(timesteps, times1, 'b-', label='Type 1 (Leaky)')
plt.plot(timesteps, times2, 'r-', label='Type 2 (StateLeaky)')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel('Number of Timesteps')
plt.ylabel('Time (seconds)')
plt.title('SNN Performance Comparison')
plt.legend()
plt.savefig("snn_performance_comparison_time.svg", format="svg")

# Print the results
print("Benchmark Results:")
print("\nTimesteps  |  Leaky (s)  |  Linear Leaky (s)  |  Ratio (T2/T1)")
print("-" * 55)
for i, steps in enumerate(timesteps):
    ratio = times2[i] / times1[i]
    print(f"{steps:9d} | {times1[i]:10.4f} | {times2[i]:10.4f} | {ratio:10.2f}")

plt.show()


# Create the plot for memory
plt.figure(figsize=(10, 6))
plt.plot(timesteps, mems1, 'b-', label='Type 1 mem (Leaky)')
plt.plot(timesteps, mems2, 'r-', label='Type 2 mem (StateLeaky)')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel('Number of Timesteps')
plt.ylabel('Memory (MiB)')
plt.title('SNN Performance Comparison')
plt.legend()
plt.savefig("snn_performance_comparison_mem.svg", format="svg")

# Print the results
print("Benchmark Results:")
print("\nTimesteps  |  Leaky (s)  |  Linear Leaky (s)  |  Ratio (T2/T1)")
print("-" * 55)
for i, steps in enumerate(timesteps):
    ratio = mems2[i] / mems1[i]
    print(f"{steps:9d} | {mems1[i]:10.4f} | {mems2[i]:10.4f} | {ratio:10.2f}")

plt.show()