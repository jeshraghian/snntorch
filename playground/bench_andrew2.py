import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import gc
import multiprocessing as mp

from snntorch._neurons.leaky import Leaky
from snntorch._neurons.stateleaky import StateLeaky
from tqdm import tqdm


# Sweep configurations: (batch_size, channels)
SWEEP_CONFIGS = [
    (1, 5),
    (10, 20),
    (100, 100),
]
N_RUNS = 10

# Same timestep schedule as baseline
TIMESTEPS = np.logspace(1, 4, num=10, dtype=int)

device = "cuda:1"
torch.set_grad_enabled(True)


def get_peak_bytes(cuda_device):
    return torch.cuda.max_memory_allocated(cuda_device)


def get_cur_bytes(cuda_device):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated(cuda_device)


# ------------------------------
# Benchmark kernels
# ------------------------------


def bench_leaky(
    num_steps: int, batch_size: int, channels: int, train: bool = False
) -> float:
    lif = Leaky(beta=0.9).to(device)

    if train:
        ctx = torch.enable_grad()
        x = torch.rand(num_steps, device=device, requires_grad=True)
    else:
        ctx = torch.no_grad()
        x = torch.rand(num_steps, device=device)

    mem = torch.zeros(batch_size, channels, device=device)
    spk = torch.zeros(batch_size, channels, device=device)

    start_time = time.time()
    with ctx:
        for step_idx in range(num_steps):
            spk, mem = lif(x[step_idx], mem=mem)
            if train and step_idx < num_steps - 1:
                mem = mem.detach()

        if train:
            loss = spk.sum() + mem.sum()
            loss.backward()
            if x.grad is not None:
                x.grad = None
            del loss
    end_time = time.time()

    del lif, x, mem, spk
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return end_time - start_time


def bench_stateleaky(
    num_steps: int, batch_size: int, channels: int, train: bool = False
) -> float:
    lif = StateLeaky(beta=0.9, channels=channels).to(device)
    input_tensor = torch.arange(
        1,
        num_steps * batch_size * channels + 1,
        device=device,
        dtype=torch.float32,
    ).view(num_steps, batch_size, channels)

    if train:
        input_tensor.requires_grad_(True)
        ctx = torch.enable_grad()
    else:
        ctx = torch.no_grad()

    start_time = time.time()
    with ctx:
        out = lif.forward(input_tensor)

        if train:
            if isinstance(out, tuple):
                spk, mem = out
                loss = spk.sum() + mem.sum()
            else:
                loss = out.sum()
            loss.backward()
            if input_tensor.grad is not None:
                input_tensor.grad = None
            del loss

    end_time = time.time()

    del lif, input_tensor, out
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return end_time - start_time


# ------------------------------
# Worker: run benchmarks for one config
# ------------------------------


def run_config(cfg):
    batch_size, channels = cfg
    results_infer = dict(
        batch_size=batch_size,
        channels=channels,
        times_leaky=[],
        times_state=[],
        mems_leaky=[],
        mems_state=[],
    )
    results_train = dict(
        batch_size=batch_size,
        channels=channels,
        times_leaky=[],
        times_state=[],
        mems_leaky=[],
        mems_state=[],
    )

    for steps in tqdm(TIMESTEPS, desc=f"B{batch_size}-C{channels}"):

        # --- Inference ---
        t1_runs, m1_runs, t2_runs, m2_runs = [], [], [], []

        for _ in range(N_RUNS):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            base1 = get_cur_bytes(device)
            t1 = bench_leaky(int(steps), batch_size, channels, train=False)
            peak1 = get_peak_bytes(device)
            d1 = max(0, peak1 - base1) / 1024**2

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            base2 = get_cur_bytes(device)
            t2 = bench_stateleaky(
                int(steps), batch_size, channels, train=False
            )
            peak2 = get_peak_bytes(device)
            d2 = max(0, peak2 - base2) / 1024**2

            t1_runs.append(t1)
            m1_runs.append(d1)
            t2_runs.append(t2)
            m2_runs.append(d2)

        results_infer["times_leaky"].append(np.mean(t1_runs))
        results_infer["times_state"].append(np.mean(t2_runs))
        results_infer["mems_leaky"].append(np.mean(m1_runs))
        results_infer["mems_state"].append(np.mean(m2_runs))

        # --- Training ---
        t1_runs, m1_runs, t2_runs, m2_runs = [], [], [], []
        for _ in range(N_RUNS):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            base1 = get_cur_bytes(device)
            t1 = bench_leaky(int(steps), batch_size, channels, train=True)
            peak1 = get_peak_bytes(device)
            d1 = max(0, peak1 - base1) / 1024**2

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            base2 = get_cur_bytes(device)
            t2 = bench_stateleaky(int(steps), batch_size, channels, train=True)
            peak2 = get_peak_bytes(device)
            d2 = max(0, peak2 - base2) / 1024**2

            t1_runs.append(t1)
            m1_runs.append(d1)
            t2_runs.append(t2)
            m2_runs.append(d2)

        results_train["times_leaky"].append(np.mean(t1_runs))
        results_train["times_state"].append(np.mean(t2_runs))
        results_train["mems_leaky"].append(np.mean(m1_runs))
        results_train["mems_state"].append(np.mean(m2_runs))

    return results_infer, results_train


# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        results = pool.map(run_config, SWEEP_CONFIGS)

    results_infer, results_train = zip(*results)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_time_inf, ax_mem_inf = axes[0]
    ax_time_trn, ax_mem_trn = axes[1]

    cmap = plt.get_cmap("tab10")
    for idx, res in enumerate(results_infer):
        color = cmap(idx % 10)
        label_suffix = f"B{res['batch_size']}-C{res['channels']}"
        ax_time_inf.plot(
            TIMESTEPS,
            res["times_leaky"],
            "-",
            color=color,
            label=f"Leaky {label_suffix}",
        )
        ax_time_inf.plot(
            TIMESTEPS,
            res["times_state"],
            "--",
            color=color,
            label=f"StateLeaky {label_suffix}",
        )
        ax_mem_inf.plot(
            TIMESTEPS,
            res["mems_leaky"],
            "-",
            color=color,
            label=f"Leaky {label_suffix}",
        )
        ax_mem_inf.plot(
            TIMESTEPS,
            res["mems_state"],
            "--",
            color=color,
            label=f"StateLeaky {label_suffix}",
        )

    for idx, res in enumerate(results_train):
        color = cmap(idx % 10)
        label_suffix = f"B{res['batch_size']}-C{res['channels']}"
        ax_time_trn.plot(
            TIMESTEPS,
            res["times_leaky"],
            "-",
            color=color,
            label=f"Leaky (train) {label_suffix}",
        )
        ax_time_trn.plot(
            TIMESTEPS,
            res["times_state"],
            "--",
            color=color,
            label=f"StateLeaky (train) {label_suffix}",
        )
        ax_mem_trn.plot(
            TIMESTEPS,
            res["mems_leaky"],
            "-",
            color=color,
            label=f"Leaky (train) {label_suffix}",
        )
        ax_mem_trn.plot(
            TIMESTEPS,
            res["mems_state"],
            "--",
            color=color,
            label=f"StateLeaky (train) {label_suffix}",
        )

    for ax in (ax_time_inf, ax_mem_inf, ax_time_trn, ax_mem_trn):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)

    ax_time_inf.set_title("SNN Performance (Time) - Inference")
    ax_time_inf.set_xlabel("Timesteps")
    ax_time_inf.set_ylabel("Time (s)")
    ax_mem_inf.set_title("SNN Memory (Peak) - Inference")
    ax_mem_inf.set_xlabel("Timesteps")
    ax_mem_inf.set_ylabel("Δ Memory (MB)")
    ax_time_trn.set_title("SNN Performance (Time) - Training")
    ax_time_trn.set_xlabel("Timesteps")
    ax_time_trn.set_ylabel("Time (s)")
    ax_mem_trn.set_title("SNN Memory (Peak) - Training")
    ax_mem_trn.set_xlabel("Timesteps")
    ax_mem_trn.set_ylabel("Δ Memory (MB)")

    ax_time_inf.legend(ncol=2, fontsize=8)
    ax_mem_inf.legend(ncol=2, fontsize=8)
    ax_time_trn.legend(ncol=2, fontsize=8)
    ax_mem_trn.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig("snn_performance_comparison.png", dpi=150)
    plt.show()
