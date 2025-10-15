import time
import gc
import argparse
import json
import subprocess
import sys
import os

from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from snntorch._neurons.leaky import Leaky
from snntorch._neurons.stateleaky import StateLeaky
from snntorch._neurons.gen2 import Gen2SingleInput


# Sweep configurations: (batch_size, channels)
SWEEP_CONFIGS = [
    (64, 64),
]
N_RUNS = 1

TIMESTEPS = np.logspace(1, 4.5, num=10, dtype=int)[::2]
# TIMESTEPS = TIMESTEPS[:]
BATCHWISE_CHUNK_SIZE = 16

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(True)


def get_peak_bytes(cuda_device):
    return torch.cuda.max_memory_allocated(cuda_device)


def get_cur_bytes(cuda_device):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated(cuda_device)


# ---------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------
def bench_leaky(num_steps, batch_size, channels, train=False, multi_beta=True):
    beta = torch.full((channels,), 0.9).to(device) if multi_beta else 0.9
    lif = Leaky(beta=beta, learn_beta=True).to(device)

    input_tensor = torch.arange(
        1, num_steps * batch_size * channels + 1,
        device=device, dtype=torch.float32
    ).view(num_steps, batch_size, channels)

    ctx = torch.enable_grad() if train else torch.no_grad()
    linear = torch.nn.Linear(channels, channels, bias=False).to(device)
    mem = torch.zeros(batch_size, channels, device=device)
    spk = torch.zeros(batch_size, channels, device=device)

    lif.forward(linear(input_tensor[:2, :2, :]))
    time.sleep(2)
    baseline_mem = get_cur_bytes(device)
    start_event, end_event = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    start_event.record()

    with ctx:
        spk_sum = torch.zeros(batch_size, channels, device=device)
        for step_idx in range(num_steps):
            z = linear(input_tensor[step_idx])
            spk, mem = lif(z, mem=mem)
            spk_sum += spk
        if train:
            loss = spk_sum.sum()
            loss.backward()
            if linear.weight.grad is not None:
                linear.weight.grad = None
            if input_tensor.grad is not None:
                input_tensor.grad = None
            del loss

    end_event.record()
    end_event.synchronize()

    del lif, linear, input_tensor, mem, spk, spk_sum
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return baseline_mem, start_event.elapsed_time(end_event) / 1000.0


def bench_stateleaky(num_steps, batch_size, channels, train=False, multi_beta=True):
    beta = torch.full((channels,), 0.9).to(device) if multi_beta else 0.9
    lif = StateLeaky(beta=beta, channels=channels, learn_beta=True).to(device)
    input_tensor = (
        torch.arange(
            1, num_steps * batch_size * channels + 1,
            device=device, dtype=torch.float32
        )
        .view(batch_size, channels, num_steps)
        .permute(2, 0, 1)
    )
    input_tensor.requires_grad_(False)
    ctx = torch.enable_grad() if train else torch.no_grad()
    linear = torch.nn.Linear(channels, channels, bias=False).to(device)

    z_chunk = linear(input_tensor[:2, :2, :])
    lif.forward(z_chunk)
    torch.cuda.synchronize()

    with ctx:
        baseline_mem = get_cur_bytes(device)
        time.sleep(2)
        start_event, end_event = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        start_event.record()

        for b_start in range(0, batch_size, BATCHWISE_CHUNK_SIZE):
            b_end = min(b_start + BATCHWISE_CHUNK_SIZE, batch_size)
            z_chunk = linear(input_tensor[:, b_start:b_end, :])
            spk_chunk, _ = lif(z_chunk)
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

    del lif, linear, input_tensor, z_chunk
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return baseline_mem, start_event.elapsed_time(end_event) / 1000.0


def bench_gen2ssm(num_steps, batch_size, channels, train=False, multi_beta=True):
    model = Gen2SingleInput(channels, channels, channels).to(device)
    input_tensor = torch.arange(
        1, num_steps * batch_size * channels + 1,
        device=device, dtype=torch.float32
    ).view(num_steps, batch_size, channels)
    input_tensor.requires_grad_(False)
    ctx = torch.enable_grad() if train else torch.no_grad()
    linear = torch.nn.Linear(channels, channels, bias=False).to(device)

    _ = model(input_tensor[:2, :2, :])
    time.sleep(2)
    baseline_mem = get_cur_bytes(device)
    start_event, end_event = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    start_event.record()

    with ctx:
        for b_start in range(0, batch_size, BATCHWISE_CHUNK_SIZE):
            b_end = min(b_start + BATCHWISE_CHUNK_SIZE, batch_size)
            z_chunk = linear(input_tensor[:, b_start:b_end, :])
            S_chunk = model(z_chunk)
            if train:
                loss = S_chunk.sum()
                loss.backward()
                del loss
        if train:
            if linear.weight.grad is not None:
                linear.weight.grad = None
            if input_tensor.grad is not None:
                input_tensor.grad = None

    end_event.record()
    end_event.synchronize()

    del model, linear, input_tensor
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return baseline_mem, start_event.elapsed_time(end_event) / 1000.0


# ---------------------------------------------------------------------
# Run configs
# ---------------------------------------------------------------------
def run_all_configs_one_run(run_idx: int):
    results_infer_all, results_train_all = [], []
    for cfg in SWEEP_CONFIGS:
        batch_size, channels = cfg
        results_infer = {
            f"{k}_{t}": []
            for k in ["times_leaky", "times_state", "times_gen2", "mems_leaky", "mems_state", "mems_gen2"]
            for t in ["single", "multi"]}
        results_infer.update(batch_size=batch_size, channels=channels)
        results_train = {
            f"{k}_{t}": []
            for k in ["times_leaky", "times_state", "times_gen2", "mems_leaky", "mems_state", "mems_gen2"]
            for t in ["single", "multi"]}
        results_train.update(batch_size=batch_size, channels=channels)

        for steps in tqdm(TIMESTEPS, desc=f"RUN{run_idx} B{batch_size}-C{channels}"):
            for bench_func, prefix in [(bench_leaky, "leaky"), (bench_stateleaky, "state"), (bench_gen2ssm, "gen2")]:
                for multi in [False, True]:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                    baseline_mem, t = bench_func(steps, batch_size, channels, train=False, multi_beta=multi)
                    peak = get_peak_bytes(device)
                    dmem = max(0, peak - baseline_mem) / 1024**2
                    results_infer[f"times_{prefix}_{'multi' if multi else 'single'}"].append(t)
                    results_infer[f"mems_{prefix}_{'multi' if multi else 'single'}"].append(dmem)
            for bench_func, prefix in [(bench_leaky, "leaky"), (bench_stateleaky, "state"), (bench_gen2ssm, "gen2")]:
                for multi in [False, True]:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                    baseline_mem, t = bench_func(steps, batch_size, channels, train=True, multi_beta=multi)
                    peak = get_peak_bytes(device)
                    dmem = max(0, peak - baseline_mem) / 1024**2
                    results_train[f"times_{prefix}_{'multi' if multi else 'single'}"].append(t)
                    results_train[f"mems_{prefix}_{'multi' if multi else 'single'}"].append(dmem)

        results_infer_all.append(results_infer)
        results_train_all.append(results_train)
    return results_infer_all, results_train_all


# ---------------------------------------------------------------------
# Dunder main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-idx", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.worker:
        infer, train = run_all_configs_one_run(args.run_idx)
        out = {"infer": infer, "train": train}
        if args.output:
            with open(args.output, "w") as f:
                json.dump(out, f)
        else:
            print(json.dumps(out))
        sys.exit(0)

    # --- Accumulate + Plot ---
    METRIC_KEYS = ["times_leaky_single", "times_leaky_multi", "times_state_single", "times_state_multi",
                   "times_gen2_single", "times_gen2_multi", "mems_leaky_single", "mems_leaky_multi",
                   "mems_state_single", "mems_state_multi", "mems_gen2_single", "mems_gen2_multi",]

    infer_sum, infer_sumsq, infer_meta = None, None, None
    train_sum, train_sumsq, train_meta = None, None, None

    for run_idx in range(N_RUNS):
        outfile = f"results_run{run_idx}.json"
        cmd = [sys.executable, __file__, "--worker", "--run-idx", str(run_idx), "--output", outfile]
        subprocess.run(cmd, check=True)
        with open(outfile, "r") as f:
            data = json.load(f)
        infer, train = data["infer"], data["train"]

        if infer_sum is None:
            infer_sum, infer_sumsq, infer_meta = [], [], []
            for cfg in infer:
                infer_meta.append({"batch_size": cfg["batch_size"], "channels": cfg["channels"]})
                cfg_sum, cfg_sumsq = {}, {}
                for k in METRIC_KEYS:
                    arr = np.array(cfg[k], dtype=float)
                    cfg_sum[k], cfg_sumsq[k] = arr.copy(), arr**2
                infer_sum.append(cfg_sum)
                infer_sumsq.append(cfg_sumsq)
        else:
            for cfg_idx in range(len(infer)):
                for k in METRIC_KEYS:
                    arr = np.array(infer[cfg_idx][k], dtype=float)
                    infer_sum[cfg_idx][k] += arr
                    infer_sumsq[cfg_idx][k] += arr**2

        if train_sum is None:
            train_sum, train_sumsq, train_meta = [], [], []
            for cfg in train:
                train_meta.append({"batch_size": cfg["batch_size"], "channels": cfg["channels"]})
                cfg_sum, cfg_sumsq = {}, {}
                for k in METRIC_KEYS:
                    arr = np.array(cfg[k], dtype=float)
                    cfg_sum[k], cfg_sumsq[k] = arr.copy(), arr**2
                train_sum.append(cfg_sum)
                train_sumsq.append(cfg_sumsq)
        else:
            for cfg_idx in range(len(train)):
                for k in METRIC_KEYS:
                    arr = np.array(train[cfg_idx][k], dtype=float)
                    train_sum[cfg_idx][k] += arr
                    train_sumsq[cfg_idx][k] += arr**2

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_time_inf, ax_mem_inf = axes[0]
    ax_time_trn, ax_mem_trn = axes[1]

    cmap = plt.get_cmap("tab10")
    colors = {"leaky": cmap(0), "state": cmap(1), "gen2": cmap(2)}

    for idx, res in enumerate(infer_sum):
        res_mean, res_std = {}, {}
        for k in METRIC_KEYS:
            mean_arr = infer_sum[idx][k] / max(N_RUNS, 1)
            var_arr = infer_sumsq[idx][k] / max(N_RUNS, 1) - mean_arr**2
            std_arr = np.sqrt(np.maximum(var_arr, 0.0))
            res_mean[k], res_std[k] = mean_arr, std_arr

        label_suffix = f"B{infer_meta[idx]['batch_size']}-C{infer_meta[idx]['channels']}"
        for prefix, fmt_single, fmt_multi in [("leaky", "-", "-."), ("state", "--", ":"), ("gen2", "o-", "o--")]:
            color = colors[prefix]
            ax_time_inf.errorbar(
                TIMESTEPS, res_mean[f"times_{prefix}_single"],
                yerr=res_std[f"times_{prefix}_single"],
                fmt=fmt_single, color=color, label=f"{prefix.capitalize()} single {label_suffix}", capsize=3)
            ax_time_inf.errorbar(
                TIMESTEPS, res_mean[f"times_{prefix}_multi"],
                yerr=res_std[f"times_{prefix}_multi"],
                fmt=fmt_multi, color=color, label=f"{prefix.capitalize()} multi {label_suffix}", capsize=3)
            ax_mem_inf.errorbar(
                TIMESTEPS, res_mean[f"mems_{prefix}_single"],
                yerr=res_std[f"mems_{prefix}_single"],
                fmt=fmt_single, color=color, label=f"{prefix.capitalize()} single {label_suffix}", capsize=3)
            ax_mem_inf.errorbar(
                TIMESTEPS, res_mean[f"mems_{prefix}_multi"],
                yerr=res_std[f"mems_{prefix}_multi"],
                fmt=fmt_multi, color=color, label=f"{prefix.capitalize()} multi {label_suffix}", capsize=3)

    for idx, res in enumerate(train_sum):
        res_mean, res_std = {}, {}
        for k in METRIC_KEYS:
            mean_arr = train_sum[idx][k] / max(N_RUNS, 1)
            var_arr = train_sumsq[idx][k] / max(N_RUNS, 1) - mean_arr**2
            std_arr = np.sqrt(np.maximum(var_arr, 0.0))
            res_mean[k], res_std[k] = mean_arr, std_arr

        label_suffix = f"B{train_meta[idx]['batch_size']}-C{train_meta[idx]['channels']}"
        for prefix, fmt_single, fmt_multi in [("leaky", "-", "-."), ("state", "--", ":"), ("gen2", "o-", "o--")]:
            color = colors[prefix]
            ax_time_trn.errorbar(
                TIMESTEPS, res_mean[f"times_{prefix}_single"],
                yerr=res_std[f"times_{prefix}_single"],
                fmt=fmt_single, color=color, label=f"{prefix.capitalize()} single (train) {label_suffix}", capsize=3)
            ax_time_trn.errorbar(
                TIMESTEPS, res_mean[f"times_{prefix}_multi"],
                yerr=res_std[f"times_{prefix}_multi"],
                fmt=fmt_multi, color=color, label=f"{prefix.capitalize()} multi (train) {label_suffix}", capsize=3)
            ax_mem_trn.errorbar(
                TIMESTEPS, res_mean[f"mems_{prefix}_single"],
                yerr=res_std[f"mems_{prefix}_single"],
                fmt=fmt_single, color=color, label=f"{prefix.capitalize()} single (train) {label_suffix}", capsize=3)
            ax_mem_trn.errorbar(
                TIMESTEPS, res_mean[f"mems_{prefix}_multi"],
                yerr=res_std[f"mems_{prefix}_multi"],
                fmt=fmt_multi, color=color, label=f"{prefix.capitalize()} multi (train) {label_suffix}", capsize=3)

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

    os.makedirs("snn_performance", exist_ok=True)
    plt.tight_layout()
    plt.savefig("snn_performance/snn_performance_gen2.png", dpi=150)
    plt.show()
