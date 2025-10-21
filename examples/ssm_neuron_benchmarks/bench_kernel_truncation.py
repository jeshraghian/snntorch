import time
import gc
import argparse
import json
import subprocess
import sys
import os

from profilehooks import profile
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from snntorch._neurons.stateleaky import StateLeaky

# Sweep configurations: (batch_size, channels)
SWEEP_CONFIGS = [
    (64, 256),
]
N_RUNS = 10

# Same timestep schedule as other benches
TIMESTEPS = np.logspace(1, 4.5, num=10, dtype=int)
BATCHWISE_CHUNK_SIZE = 32

# Truncation sweep: None = full kernel, others are window sizes.
# For small T, values greater than T effectively become full.
TRUNCATION_STEPS = [None, 512, 4096]


device = "cuda:1"
torch.set_grad_enabled(True)


def _klabel(k):
    return "full" if k is None else f"k{int(k)}"


def get_peak_bytes(cuda_device):
    return torch.cuda.max_memory_allocated(cuda_device)


def get_cur_bytes(cuda_device):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated(cuda_device)


# @profile(skip=True, stdout=False, filename="kernel_trunc_state.prof")
def bench_stateleaky(
    num_steps: int,
    batch_size: int,
    channels: int,
    truncation_steps,
    train: bool = False,
) -> float:
    # define lif and input
    beta = torch.full((channels,), 0.9).to(device)
    lif = StateLeaky(
        beta=beta,
        channels=channels,
        learn_beta=True,
        kernel_truncation_steps=truncation_steps,
    ).to(device)

    input_tensor = (
        torch.arange(
            1,
            num_steps * batch_size * channels + 1,
            device=device,
            dtype=torch.float32,
        )
        .view(batch_size, channels, num_steps)
        .contiguous()
        .permute(2, 0, 1)
    )
    input_tensor.requires_grad_(False)

    # define context
    if train:
        ctx = torch.enable_grad()
    else:
        ctx = torch.no_grad()

    # define external linear to match StateLeaky usage elsewhere
    linear = torch.nn.Linear(channels, channels, bias=False).to(device)

    def forward_wrapper(lif_input):
        spk, mem = lif.forward(lif_input)
        return spk, mem

    # warmup
    b_start = 0
    b_end = min(b_start + BATCHWISE_CHUNK_SIZE, batch_size)
    z_chunk = linear(
        input_tensor[:, b_start:b_end, :].reshape(-1, channels)
    ).view(
        num_steps,
        b_end - b_start,
        channels,
    )
    forward_wrapper(z_chunk)

    torch.cuda.synchronize()

    with ctx:
        baseline_mem = get_cur_bytes(device)
        time.sleep(2)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        start_event.record()
        for b_start in range(0, batch_size, BATCHWISE_CHUNK_SIZE):
            b_end = min(b_start + BATCHWISE_CHUNK_SIZE, batch_size)
            z_chunk = linear(
                input_tensor[:, b_start:b_end, :].reshape(-1, channels)
            ).view(num_steps, b_end - b_start, channels)

            spk_chunk, _ = forward_wrapper(z_chunk)

            if train:
                chunk_loss = spk_chunk.sum()
                chunk_loss.backward()
                del chunk_loss

        if train:
            if linear.weight.grad is not None:
                linear.weight.grad = None
            if input_tensor.grad is not None:
                input_tensor.grad = None

    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()

    # clean up
    del lif, linear, input_tensor, z_chunk
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    return baseline_mem, start_event.elapsed_time(end_event) / 1000.0


def run_all_configs_one_run(run_idx: int):
    results_infer_all = []
    results_train_all = []

    klabels = [_klabel(k) for k in TRUNCATION_STEPS]

    for cfg in SWEEP_CONFIGS:
        batch_size, channels = cfg

        # Initialize result dicts with per-K arrays (StateLeaky only)
        results_infer = dict(
            batch_size=batch_size,
            channels=channels,
        )
        results_train = dict(
            batch_size=batch_size,
            channels=channels,
        )

        for kl in klabels:
            results_infer[f"times_state_{kl}"] = []
            results_infer[f"mems_state_{kl}"] = []

            results_train[f"times_state_{kl}"] = []
            results_train[f"mems_state_{kl}"] = []

        for steps in tqdm(
            TIMESTEPS, desc=f"RUN{run_idx} B{batch_size}-C{channels}"
        ):
            for trunc in TRUNCATION_STEPS:
                kl = _klabel(trunc)

                # --- Inference ---
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                baseline_mem, t = bench_stateleaky(
                    int(steps),
                    batch_size,
                    channels,
                    truncation_steps=trunc,
                    train=False,
                )
                peak = get_peak_bytes(device)
                dmem = max(0, peak - baseline_mem) / 1024**2
                results_infer[f"times_state_{kl}"].append(t)
                results_infer[f"mems_state_{kl}"].append(dmem)

                # --- Training ---
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                baseline_mem, t = bench_stateleaky(
                    int(steps),
                    batch_size,
                    channels,
                    truncation_steps=trunc,
                    train=True,
                )
                peak = get_peak_bytes(device)
                dmem = max(0, peak - baseline_mem) / 1024**2
                results_train[f"times_state_{kl}"].append(t)
                results_train[f"mems_state_{kl}"].append(dmem)

        results_infer_all.append(results_infer)
        results_train_all.append(results_train)

    return results_infer_all, results_train_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker", action="store_true", help="Run worker mode"
    )
    parser.add_argument("--run-idx", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.worker:
        # Worker mode: run benchmarks and dump JSON
        infer, train = run_all_configs_one_run(args.run_idx)
        out = {"infer": infer, "train": train}
        if args.output:
            with open(args.output, "w") as f:
                json.dump(out, f)
        else:
            print(json.dumps(out))
        sys.exit(0)

    # Main mode: launch workers
    klabels = [_klabel(k) for k in TRUNCATION_STEPS]
    METRIC_KEYS = []
    for kl in klabels:
        METRIC_KEYS.extend(
            [
                f"times_state_{kl}",
                f"mems_state_{kl}",
            ]
        )

    # Accumulators for mean/std across runs
    infer_sum = None
    infer_sumsq = None
    infer_meta = None

    train_sum = None
    train_sumsq = None
    train_meta = None

    for run_idx in range(N_RUNS):
        outfile = f"results_run{run_idx}.json"
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--run-idx",
            str(run_idx),
            "--output",
            outfile,
        ]
        subprocess.run(cmd, check=True)

        with open(outfile, "r") as f:
            data = json.load(f)
        infer, train = data["infer"], data["train"]

        if infer_sum is None:
            infer_sum = []
            infer_sumsq = []
            infer_meta = []
            for cfg in infer:
                infer_meta.append(
                    {
                        "batch_size": cfg["batch_size"],
                        "channels": cfg["channels"],
                    }
                )
                cfg_sum = {}
                cfg_sumsq = {}
                for k in METRIC_KEYS:
                    arr = np.array(cfg[k], dtype=float)
                    cfg_sum[k] = arr.copy()
                    cfg_sumsq[k] = arr**2
                infer_sum.append(cfg_sum)
                infer_sumsq.append(cfg_sumsq)
        else:
            for cfg_idx in range(len(infer)):
                for k in METRIC_KEYS:
                    arr = np.array(infer[cfg_idx][k], dtype=float)
                    infer_sum[cfg_idx][k] = infer_sum[cfg_idx][k] + arr
                    infer_sumsq[cfg_idx][k] = infer_sumsq[cfg_idx][k] + arr**2

        if train_sum is None:
            train_sum = []
            train_sumsq = []
            train_meta = []
            for cfg in train:
                train_meta.append(
                    {
                        "batch_size": cfg["batch_size"],
                        "channels": cfg["channels"],
                    }
                )
                cfg_sum = {}
                cfg_sumsq = {}
                for k in METRIC_KEYS:
                    arr = np.array(cfg[k], dtype=float)
                    cfg_sum[k] = arr.copy()
                    cfg_sumsq[k] = arr**2
                train_sum.append(cfg_sum)
                train_sumsq.append(cfg_sumsq)
        else:
            for cfg_idx in range(len(train)):
                for k in METRIC_KEYS:
                    arr = np.array(train[cfg_idx][k], dtype=float)
                    train_sum[cfg_idx][k] = train_sum[cfg_idx][k] + arr
                    train_sumsq[cfg_idx][k] = train_sumsq[cfg_idx][k] + arr**2

    # Compute mean and std across runs
    results_infer = []
    results_train = []
    for cfg_idx in range(len(SWEEP_CONFIGS)):
        cfg_res = {
            "batch_size": infer_meta[cfg_idx]["batch_size"],
            "channels": infer_meta[cfg_idx]["channels"],
        }
        for k in METRIC_KEYS:
            mean_arr = infer_sum[cfg_idx][k] / max(N_RUNS, 1)
            var_arr = infer_sumsq[cfg_idx][k] / max(N_RUNS, 1) - mean_arr**2
            var_arr = np.maximum(var_arr, 0.0)
            std_arr = np.sqrt(var_arr)
            cfg_res[k] = mean_arr.tolist()
            cfg_res[f"std_{k}"] = std_arr.tolist()
        results_infer.append(cfg_res)

        cfg_res_t = {
            "batch_size": train_meta[cfg_idx]["batch_size"],
            "channels": train_meta[cfg_idx]["channels"],
        }
        for k in METRIC_KEYS:
            mean_arr = train_sum[cfg_idx][k] / max(N_RUNS, 1)
            var_arr = train_sumsq[cfg_idx][k] / max(N_RUNS, 1) - mean_arr**2
            var_arr = np.maximum(var_arr, 0.0)
            std_arr = np.sqrt(var_arr)
            cfg_res_t[k] = mean_arr.tolist()
            cfg_res_t[f"std_{k}"] = std_arr.tolist()
        results_train.append(cfg_res_t)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_time_inf, ax_mem_inf = axes[0]
    ax_time_trn, ax_mem_trn = axes[1]

    cmap = plt.get_cmap("tab10")
    for idx, res in enumerate(results_infer):
        # Use color-only separation for truncation levels; uniform line style
        color_map = {kl: cmap(i % 10) for i, kl in enumerate(klabels)}
        label_suffix = f"B{res['batch_size']}-C{res['channels']}"

        for kl in klabels:
            color = color_map[kl]
            ax_time_inf.errorbar(
                TIMESTEPS,
                res[f"times_state_{kl}"],
                yerr=res.get(f"std_times_state_{kl}", None),
                fmt="-",
                color=color,
                label=f"State {kl} {label_suffix}",
                capsize=3,
            )

            ax_mem_inf.errorbar(
                TIMESTEPS,
                res[f"mems_state_{kl}"],
                yerr=res.get(f"std_mems_state_{kl}", None),
                fmt="-",
                color=color,
                label=f"State {kl} {label_suffix}",
                capsize=3,
            )

    for idx, res in enumerate(results_train):
        color_map = {kl: cmap(i % 10) for i, kl in enumerate(klabels)}
        label_suffix = f"B{res['batch_size']}-C{res['channels']}"

        for kl in klabels:
            color = color_map[kl]
            ax_time_trn.errorbar(
                TIMESTEPS,
                res[f"times_state_{kl}"],
                yerr=res.get(f"std_times_state_{kl}", None),
                fmt="-",
                color=color,
                label=f"State {kl} (train) {label_suffix}",
                capsize=3,
            )

            ax_mem_trn.errorbar(
                TIMESTEPS,
                res[f"mems_state_{kl}"],
                yerr=res.get(f"std_mems_state_{kl}", None),
                fmt="-",
                color=color,
                label=f"State {kl} (train) {label_suffix}",
                capsize=3,
            )

    for ax in (ax_time_inf, ax_mem_inf, ax_time_trn, ax_mem_trn):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)

    ax_time_inf.set_title("Kernel Truncation Sweep (Time) - Inference")
    ax_time_inf.set_xlabel("Timesteps")
    ax_time_inf.set_ylabel("Time (s)")
    ax_mem_inf.set_title("Kernel Truncation Sweep (Peak Memory) - Inference")
    ax_mem_inf.set_xlabel("Timesteps")
    ax_mem_inf.set_ylabel("Δ Memory (MB)")
    ax_time_trn.set_title("Kernel Truncation Sweep (Time) - Training")
    ax_time_trn.set_xlabel("Timesteps")
    ax_time_trn.set_ylabel("Time (s)")
    ax_mem_trn.set_title("Kernel Truncation Sweep (Peak Memory) - Training")
    ax_mem_trn.set_xlabel("Timesteps")
    ax_mem_trn.set_ylabel("Δ Memory (MB)")

    ax_time_inf.legend(ncol=2, fontsize=8)
    ax_mem_inf.legend(ncol=2, fontsize=8)
    ax_time_trn.legend(ncol=2, fontsize=8)
    ax_mem_trn.legend(ncol=2, fontsize=8)

    os.makedirs("snn_performance", exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        "snn_performance/snn_performance_kernel_truncation.png", dpi=150
    )
    plt.savefig("snn_performance/snn_performance_kernel_truncation.pdf")
