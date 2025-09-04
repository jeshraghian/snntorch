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
TIMESTEPS = np.logspace(1, 3.5, num=10, dtype=int)
NUM_SEEDS = 10

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
    results_infer = []

    for batch_size, channels in SWEEP_CONFIGS:
        sum_times_leaky = np.zeros(len(TIMESTEPS), dtype=float)
        sum_times_state = np.zeros(len(TIMESTEPS), dtype=float)
        sum_mems_leaky = np.zeros(len(TIMESTEPS), dtype=float)
        sum_mems_state = np.zeros(len(TIMESTEPS), dtype=float)
        sum_bases_leaky = np.zeros(len(TIMESTEPS), dtype=float)
        sum_peaks_leaky = np.zeros(len(TIMESTEPS), dtype=float)
        sum_bases_state = np.zeros(len(TIMESTEPS), dtype=float)
        sum_peaks_state = np.zeros(len(TIMESTEPS), dtype=float)

        for seed in range(NUM_SEEDS):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            times_leaky, times_state = [], []
            mems_leaky, mems_state = [], []
            bases_leaky, bases_state = [], []
            peaks_leaky, peaks_state = [], []

            for steps in tqdm(
                TIMESTEPS, desc=f"B{batch_size}-C{channels} [seed {seed}]"
            ):
                n_runs = 2
                t1_runs, t2_runs = [], []
                m1_runs, m2_runs = [], []
                b1_runs, b2_runs = [], []
                p1_runs, p2_runs = [], []

                for _ in range(n_runs):
                    # --- Leaky ---
                    torch.cuda.synchronize()
                    # torch.cuda.reset_peak_memory_stats(device)
                    # torch.cuda.reset_max_memory_allocated(device)
                    # torch.cuda.reset_max_memory_cached(device)
                    # torch.cuda.reset_accumulated_memory_stats(device)
                    base1 = get_cur_bytes(device)
                    t1 = bench_leaky(int(steps), batch_size, channels)
                    torch.cuda.synchronize()
                    peak1 = get_peak_bytes(device)
                    d1 = peak1
                    # d1 = max(0, peak1 - base1) / 1024**2

                    t1_runs.append(t1)
                    m1_runs.append(d1)
                    b1_runs.append(base1 / 1024**2)
                    p1_runs.append(peak1 / 1024**2)

                    # --- StateLeaky ---
                    torch.cuda.synchronize()
                    # torch.cuda.reset_peak_memory_stats(device)
                    # torch.cuda.reset_max_memory_allocated(device)
                    # torch.cuda.reset_max_memory_cached(device)
                    # torch.cuda.reset_accumulated_memory_stats(device)
                    base2 = get_cur_bytes(device)
                    t2 = bench_stateleaky(int(steps), batch_size, channels)
                    torch.cuda.synchronize()
                    peak2 = get_peak_bytes(device)
                    d2 = peak2
                    # d2 = max(0, peak2 - base2) / 1024**2

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

            sum_times_leaky += np.array(times_leaky)
            sum_times_state += np.array(times_state)
            sum_mems_leaky += np.array(mems_leaky)
            sum_mems_state += np.array(mems_state)
            sum_bases_leaky += np.array(bases_leaky)
            sum_peaks_leaky += np.array(peaks_leaky)
            sum_bases_state += np.array(bases_state)
            sum_peaks_state += np.array(peaks_state)

        results_infer.append(
            {
                "batch_size": batch_size,
                "channels": channels,
                "times_leaky": sum_times_leaky / NUM_SEEDS,
                "times_state": sum_times_state / NUM_SEEDS,
                "mems_leaky": sum_mems_leaky / NUM_SEEDS,
                "mems_state": sum_mems_state / NUM_SEEDS,
                "bases_leaky": sum_bases_leaky / NUM_SEEDS,
                "peaks_leaky": sum_peaks_leaky / NUM_SEEDS,
                "bases_state": sum_bases_state / NUM_SEEDS,
                "peaks_state": sum_peaks_state / NUM_SEEDS,
            }
        )

# --- Training-enabled benchmarks (forward + backward) ---


def bench_leaky_train(num_steps: int, batch_size: int, channels: int) -> float:
    lif = snn.Leaky(beta=0.9).to(device)
    x = torch.rand(num_steps, device=device, requires_grad=True)
    mem = torch.zeros(batch_size, channels, device=device)
    spk = torch.zeros(batch_size, channels, device=device)

    start_time = time.time()
    for step_idx in range(num_steps):
        spk, mem = lif(x[step_idx], mem=mem)
        if step_idx < num_steps - 1:
            mem = mem.detach()
    loss = mem.sum()
    loss.backward()
    end_time = time.time()

    # Clear grads to avoid accumulation across runs
    if x.grad is not None:
        x.grad = None
    return end_time - start_time


def bench_stateleaky_train(
    num_steps: int, batch_size: int, channels: int
) -> float:
    lif = StateLeaky(beta=0.9, channels=channels).to(device)
    input_tensor = torch.arange(
        1,
        num_steps * batch_size * channels + 1,
        device=device,
        dtype=torch.float32,
    ).view(num_steps, batch_size, channels)
    input_tensor.requires_grad_(True)

    start_time = time.time()
    out = lif.forward(input_tensor)
    if isinstance(out, tuple):
        # use membrane trace for scalar loss
        _, mem = out
        loss = mem.sum()
    else:
        loss = out.sum()
    loss.backward()
    end_time = time.time()

    if input_tensor.grad is not None:
        input_tensor.grad = None
    return end_time - start_time


with torch.enable_grad():
    results_train = []

    for batch_size, channels in SWEEP_CONFIGS:
        sum_times_leaky_t = np.zeros(len(TIMESTEPS), dtype=float)
        sum_times_state_t = np.zeros(len(TIMESTEPS), dtype=float)
        sum_mems_leaky_t = np.zeros(len(TIMESTEPS), dtype=float)
        sum_mems_state_t = np.zeros(len(TIMESTEPS), dtype=float)
        sum_bases_leaky_t = np.zeros(len(TIMESTEPS), dtype=float)
        sum_peaks_leaky_t = np.zeros(len(TIMESTEPS), dtype=float)
        sum_bases_state_t = np.zeros(len(TIMESTEPS), dtype=float)
        sum_peaks_state_t = np.zeros(len(TIMESTEPS), dtype=float)

        for seed in range(NUM_SEEDS):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            times_leaky_t, times_state_t = [], []
            mems_leaky_t, mems_state_t = [], []
            bases_leaky_t, bases_state_t = [], []
            peaks_leaky_t, peaks_state_t = [], []

            for steps in tqdm(
                TIMESTEPS,
                desc=f"B{batch_size}-C{channels} [train seed {seed}]",
            ):
                n_runs = 2
                t1_runs, t2_runs = [], []
                m1_runs, m2_runs = [], []
                b1_runs, b2_runs = [], []
                p1_runs, p2_runs = [], []

                for _ in range(n_runs):
                    # --- Leaky train ---
                    torch.cuda.synchronize()
                    # torch.cuda.reset_peak_memory_stats(device)
                    # torch.cuda.reset_max_memory_allocated(device)
                    # torch.cuda.reset_max_memory_cached(device)
                    # torch.cuda.reset_accumulated_memory_stats(device)
                    base1 = get_cur_bytes(device)
                    t1 = bench_leaky_train(int(steps), batch_size, channels)
                    torch.cuda.synchronize()
                    peak1 = get_peak_bytes(device)
                    d1 = peak1
                    # d1 = max(0, peak1 - base1) / 1024**2

                    t1_runs.append(t1)
                    m1_runs.append(d1)
                    b1_runs.append(base1 / 1024**2)
                    p1_runs.append(peak1 / 1024**2)

                    # --- StateLeaky train ---
                    torch.cuda.synchronize()
                    # torch.cuda.reset_peak_memory_stats(device)
                    # torch.cuda.reset_max_memory_allocated(device)
                    # torch.cuda.reset_max_memory_cached(device)
                    # torch.cuda.reset_accumulated_memory_stats(device)
                    base2 = get_cur_bytes(device)
                    t2 = bench_stateleaky_train(
                        int(steps), batch_size, channels
                    )
                    torch.cuda.synchronize()
                    peak2 = get_peak_bytes(device)
                    d2 = peak2
                    # d2 = max(0, peak2 - base2) / 1024**2

                    t2_runs.append(t2)
                    m2_runs.append(d2)
                    b2_runs.append(base2 / 1024**2)
                    p2_runs.append(peak2 / 1024**2)

                times_leaky_t.append(np.mean(t1_runs))
                times_state_t.append(np.mean(t2_runs))
                mems_leaky_t.append(np.mean(m1_runs))
                mems_state_t.append(np.mean(m2_runs))
                bases_leaky_t.append(np.mean(b1_runs))
                peaks_leaky_t.append(np.mean(p1_runs))
                bases_state_t.append(np.mean(b2_runs))
                peaks_state_t.append(np.mean(p2_runs))

            sum_times_leaky_t += np.array(times_leaky_t)
            sum_times_state_t += np.array(times_state_t)
            sum_mems_leaky_t += np.array(mems_leaky_t)
            sum_mems_state_t += np.array(mems_state_t)
            sum_bases_leaky_t += np.array(bases_leaky_t)
            sum_peaks_leaky_t += np.array(peaks_leaky_t)
            sum_bases_state_t += np.array(bases_state_t)
            sum_peaks_state_t += np.array(peaks_state_t)

        results_train.append(
            {
                "batch_size": batch_size,
                "channels": channels,
                "times_leaky": sum_times_leaky_t / NUM_SEEDS,
                "times_state": sum_times_state_t / NUM_SEEDS,
                "mems_leaky": sum_mems_leaky_t / NUM_SEEDS,
                "mems_state": sum_mems_state_t / NUM_SEEDS,
                "bases_leaky": sum_bases_leaky_t / NUM_SEEDS,
                "peaks_leaky": sum_peaks_leaky_t / NUM_SEEDS,
                "bases_state": sum_bases_state_t / NUM_SEEDS,
                "peaks_state": sum_peaks_state_t / NUM_SEEDS,
            }
        )

# ---- Plots: 2 rows (inference, training) ----
fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=False)
ax_time_inf, ax_mem_inf = axes[0]
ax_time_trn, ax_mem_trn = axes[1]

# Choose colors per config, linestyle per neuron type
cmap = plt.get_cmap("tab10")
for idx, res in enumerate(results_infer):
    color = cmap(idx % 10)
    label_suffix = f"B{res['batch_size']}-C{res['channels']}"

    ax_time_inf.plot(
        TIMESTEPS,
        res["times_leaky"],
        linestyle="-",
        color=color,
        label=f"Leaky {label_suffix}",
    )
    ax_time_inf.plot(
        TIMESTEPS,
        res["times_state"],
        linestyle="--",
        color=color,
        label=f"StateLeaky {label_suffix}",
    )

    ax_mem_inf.plot(
        TIMESTEPS,
        res["mems_leaky"],
        linestyle="-",
        color=color,
        label=f"Leaky Peak {label_suffix}",
    )
    ax_mem_inf.plot(
        TIMESTEPS,
        res["mems_state"],
        linestyle="--",
        color=color,
        label=f"State Peak {label_suffix}",
    )

for idx, res in enumerate(results_train):
    color = cmap(idx % 10)
    label_suffix = f"B{res['batch_size']}-C{res['channels']}"

    ax_time_trn.plot(
        TIMESTEPS,
        res["times_leaky"],
        linestyle="-",
        color=color,
        label=f"Leaky (train) {label_suffix}",
    )
    ax_time_trn.plot(
        TIMESTEPS,
        res["times_state"],
        linestyle="--",
        color=color,
        label=f"StateLeaky (train) {label_suffix}",
    )

    ax_mem_trn.plot(
        TIMESTEPS,
        res["mems_leaky"],
        linestyle="-",
        color=color,
        label=f"Leaky Peak (train) {label_suffix}",
    )
    ax_mem_trn.plot(
        TIMESTEPS,
        res["mems_state"],
        linestyle="--",
        color=color,
        label=f"State Peak (train) {label_suffix}",
    )

# Formatting
for ax in (ax_time_inf, ax_mem_inf, ax_time_trn, ax_mem_trn):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="-", alpha=0.2)

ax_time_inf.set_xlabel("Number of Timesteps")
ax_time_inf.set_ylabel("Time (seconds)")
ax_time_inf.set_title("SNN Performance (Time) - Inference")
ax_time_inf.legend(ncol=2, fontsize=8)

ax_mem_inf.set_xlabel("Number of Timesteps")
ax_mem_inf.set_ylabel("Peak Allocated (bytes)")
ax_mem_inf.set_title("SNN Memory (Peak) - Inference")
ax_mem_inf.legend(ncol=2, fontsize=8)

ax_time_trn.set_xlabel("Number of Timesteps")
ax_time_trn.set_ylabel("Time (seconds)")
ax_time_trn.set_title("SNN Performance (Time) - Training (fwd+bwd)")
ax_time_trn.legend(ncol=2, fontsize=8)

ax_mem_trn.set_xlabel("Number of Timesteps")
ax_mem_trn.set_ylabel("Peak Allocated (bytes)")
ax_mem_trn.set_title("SNN Memory (Peak) - Training (fwd+bwd)")
ax_mem_trn.legend(ncol=2, fontsize=8)

plt.tight_layout()

# Print summaries per configuration
for res in results_infer:
    print(
        f"\n[Inference] Benchmark Results (Time) for B{res['batch_size']} C{res['channels']}:"
    )
    print("Timesteps | Leaky (s) | StateLeaky (s) | Ratio (State/Leaky)")
    for i, steps in enumerate(TIMESTEPS):
        ratio = res["times_state"][i] / res["times_leaky"][i]
        print(
            f"{int(steps):9d} | {res['times_leaky'][i]:9.4f} | {res['times_state'][i]:13.4f} | {ratio:16.2f}"
        )

    print(
        f"\n[Inference] Benchmark Results (Memory - Peak) for B{res['batch_size']} C{res['channels']}:"
    )
    print("Timesteps | Leaky Peak | StateLeaky Peak | Ratio (State/Leaky)")
    for i, steps in enumerate(TIMESTEPS):
        denom = res["mems_leaky"][i]
        ratio = (res["mems_state"][i] / denom) if denom else float("inf")
        print(
            f"{int(steps):9d} | {res['mems_leaky'][i]:10.4f} | {res['mems_state'][i]:16.4f} | {ratio:16.2f}"
        )

for res in results_train:
    print(
        f"\n[Training] Benchmark Results (Time) for B{res['batch_size']} C{res['channels']}:"
    )
    print("Timesteps | Leaky (s) | StateLeaky (s) | Ratio (State/Leaky)")
    for i, steps in enumerate(TIMESTEPS):
        ratio = res["times_state"][i] / res["times_leaky"][i]
        print(
            f"{int(steps):9d} | {res['times_leaky'][i]:9.4f} | {res['times_state'][i]:13.4f} | {ratio:16.2f}"
        )

    print(
        f"\n[Training] Benchmark Results (Memory - Peak) for B{res['batch_size']} C{res['channels']}:"
    )
    print("Timesteps | Leaky Peak | StateLeaky Peak | Ratio (State/Leaky)")
    for i, steps in enumerate(TIMESTEPS):
        denom = res["mems_leaky"][i]
        ratio = (res["mems_state"][i] / denom) if denom else float("inf")
        print(
            f"{int(steps):9d} | {res['mems_leaky'][i]:10.4f} | {res['mems_state'][i]:16.4f} | {ratio:16.2f}"
        )

fig.savefig("snn_performance_comparison.png", format="png")
