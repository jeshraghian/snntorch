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

from snntorch._neurons.leaky import Leaky
from snntorch._neurons.stateleaky import StateLeaky


# Sweep configurations: (batch_size, channels)
SWEEP_CONFIGS = [
    (64, 256),
]
N_RUNS = 2

# Same timestep schedule as baseline
TIMESTEPS = np.logspace(1, 4.5, num=10, dtype=int)[-5:]
BATCHWISE_CHUNK_SIZE = 32


device = "cuda:1"
torch.set_grad_enabled(True)


def get_peak_bytes(cuda_device):
    return torch.cuda.max_memory_allocated(cuda_device)


def get_cur_bytes(cuda_device):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated(cuda_device)


def bench_leaky(
    num_steps: int,
    batch_size: int,
    channels: int,
    train: bool = False,
    multi_beta: bool = True,
) -> float:
    beta = torch.full((channels,), 0.9).to(device) if multi_beta else 0.9
    lif = Leaky(beta=beta, learn_beta=True).to(device)

    # Create per-timestep inputs with shape [T, B, C]
    input_tensor = torch.arange(
        1,
        num_steps * batch_size * channels + 1,
        device=device,
        dtype=torch.float32,
    ).view(num_steps, batch_size, channels)

    if train:
        input_tensor.requires_grad_(False)
        ctx = torch.enable_grad()
    else:
        ctx = torch.no_grad()

    # Linear projection: channels -> channels, no bias
    linear = torch.nn.Linear(channels, channels, bias=False).to(device)

    mem = torch.zeros(batch_size, channels, device=device)
    spk = torch.zeros(batch_size, channels, device=device)

    # warmup
    lif.forward(linear(input_tensor[:2, :2, :]))
    time.sleep(2)

    baseline_mem = get_cur_bytes(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    start_time = time.time()
    with ctx:
        spk_sum = torch.zeros(batch_size, channels, device=device)
        for step_idx in range(num_steps):
            z = linear(input_tensor[step_idx])
            spk, mem = lif(z, mem=mem)
            spk_sum += spk

        if train:
            # Use spk only
            loss = spk_sum.sum()
            loss.backward()
            if linear.weight.grad is not None:
                linear.weight.grad = None
            if input_tensor.grad is not None:
                input_tensor.grad = None
            del loss
    end_event.record()
    end_event.synchronize()
    end_time = time.time()

    del lif, linear, input_tensor, mem, spk, spk_sum
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return baseline_mem, start_event.elapsed_time(end_event) / 1000.0


# @profile(skip=True, stdout=False, filename="baseline.prof")
def bench_stateleaky(
    num_steps: int,
    batch_size: int,
    channels: int,
    train: bool = False,
    multi_beta: bool = True,
) -> float:
    # define lif and input
    beta = torch.full((channels,), 0.9).to(device) if multi_beta else 0.9
    lif = StateLeaky(beta=beta, channels=channels, learn_beta=True).to(device)
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

    # define linear
    linear = torch.nn.Linear(channels, channels, bias=False).to(device)

    def forward_wrapper(lif_input):
        spk, mem = lif.forward(lif_input)
        return spk, mem

    # conduct a warmup on the lif
    b_start = 0
    b_end = min(b_start + BATCHWISE_CHUNK_SIZE, batch_size)
    z_chunk = (
        linear(input_tensor[:, b_start:b_end, :].reshape(-1, channels)).view(
            num_steps,
            b_end - b_start,
            channels,
        )
        # .contiguous()
    )
    forward_wrapper(z_chunk)

    # make sure cuda is synchronized
    torch.cuda.synchronize()

    with ctx:
        # setup
        baseline_mem = get_cur_bytes(device)
        time.sleep(2)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        # record start event
        start_event.record()
        start_time = time.time()

        chunks_processed = 0
        log = False
        for b_start in range(0, batch_size, BATCHWISE_CHUNK_SIZE):
            if num_steps > 8000 and log:
                torch.cuda.synchronize()
                inc_time = time.time()
                print(f"inc_time: {inc_time - start_time}")
            chunks_processed += 1

            # chunked forward
            # will materialize in the output view
            b_end = min(b_start + BATCHWISE_CHUNK_SIZE, batch_size)
            z_chunk = linear(
                input_tensor[:, b_start:b_end, :].reshape(-1, channels)
            ).view(num_steps, b_end - b_start, channels)

            if num_steps > 8000 and log:
                print(f"z_chunk.is_contiguous(): {z_chunk.is_contiguous()}")
            if num_steps > 8000 and log:
                torch.cuda.synchronize()
                inc_time = time.time()
                print(f"postlinear_time: {inc_time - start_time}")
                print(f"z_chunk.shape: {z_chunk.shape}")
                print(f"z_chunk.stride(): {z_chunk.stride()}")

            # forward
            # stride doesn't seem to matter
            if num_steps > 8000 and log:
                torch.cuda.synchronize()
                inc_time = time.time()
                print(f"preforward_time: {inc_time - start_time}")
            spk_chunk, _ = forward_wrapper(z_chunk)
            assert spk_chunk.shape == (num_steps, b_end - b_start, channels)
            if num_steps > 8000 and log:
                torch.cuda.synchronize()
                inc_time = time.time()
                print(f"postforward_time: {inc_time - start_time}")

            # backwards w/ grad accum
            if train:
                chunk_loss = spk_chunk.sum()
                chunk_loss.backward()
                del chunk_loss
            if num_steps > 8000 and log:
                torch.cuda.synchronize()
                inc_time = time.time()
                print(f"postbackward_time: {inc_time - start_time}")

        # zero grads
        if train:
            if linear.weight.grad is not None:
                linear.weight.grad = None
            if input_tensor.grad is not None:
                input_tensor.grad = None

    if num_steps > 8000 and log:
        torch.cuda.synchronize()
        inc_time = time.time()
        print(f"end_time: {inc_time - start_time}")
        input()

    # end timer
    end_event.record()
    end_event.synchronize()
    torch.cuda.synchronize()
    end_time = time.time()

    print(f"chunks_processed: {chunks_processed}")

    # clean up
    del lif, linear, input_tensor, z_chunk
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    return baseline_mem, start_event.elapsed_time(end_event) / 1000.0


def run_all_configs_one_run(run_idx: int):
    results_infer_all = []
    results_train_all = []

    for cfg in SWEEP_CONFIGS:
        batch_size, channels = cfg

        results_infer = dict(
            batch_size=batch_size,
            channels=channels,
            times_leaky_single=[],
            times_leaky_multi=[],
            times_state_single=[],
            times_state_multi=[],
            mems_leaky_single=[],
            mems_leaky_multi=[],
            mems_state_single=[],
            mems_state_multi=[],
        )
        results_train = dict(
            batch_size=batch_size,
            channels=channels,
            times_leaky_single=[],
            times_leaky_multi=[],
            times_state_single=[],
            times_state_multi=[],
            mems_leaky_single=[],
            mems_leaky_multi=[],
            mems_state_single=[],
            mems_state_multi=[],
        )

        for steps in tqdm(
            TIMESTEPS, desc=f"RUN{run_idx} B{batch_size}-C{channels}"
        ):
            # --- Inference ---
            # Leaky single
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_mem, t = bench_leaky(
                int(steps), batch_size, channels, train=False, multi_beta=False
            )
            peak = get_peak_bytes(device)
            dmem = max(0, peak - baseline_mem) / 1024**2
            results_infer["times_leaky_single"].append(t)
            results_infer["mems_leaky_single"].append(dmem)

            # Leaky multi
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_mem, t = bench_leaky(
                int(steps), batch_size, channels, train=False, multi_beta=True
            )
            peak = get_peak_bytes(device)
            dmem = max(0, peak - baseline_mem) / 1024**2
            results_infer["times_leaky_multi"].append(t)
            results_infer["mems_leaky_multi"].append(dmem)

            # StateLeaky single
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_mem, t = bench_stateleaky(
                int(steps), batch_size, channels, train=False, multi_beta=False
            )
            peak = get_peak_bytes(device)
            dmem = max(0, peak - baseline_mem) / 1024**2
            results_infer["times_state_single"].append(t)
            results_infer["mems_state_single"].append(dmem)

            # StateLeaky multi
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_mem, t = bench_stateleaky(
                int(steps), batch_size, channels, train=False, multi_beta=True
            )
            peak = get_peak_bytes(device)
            dmem = max(0, peak - baseline_mem) / 1024**2
            results_infer["times_state_multi"].append(t)
            results_infer["mems_state_multi"].append(dmem)

            # --- Training ---
            # Leaky single
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_mem, t = bench_leaky(
                int(steps), batch_size, channels, train=True, multi_beta=False
            )
            peak = get_peak_bytes(device)
            dmem = max(0, peak - baseline_mem) / 1024**2
            results_train["times_leaky_single"].append(t)
            results_train["mems_leaky_single"].append(dmem)

            # Leaky multi
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_mem, t = bench_leaky(
                int(steps), batch_size, channels, train=True, multi_beta=True
            )
            peak = get_peak_bytes(device)
            dmem = max(0, peak - baseline_mem) / 1024**2
            results_train["times_leaky_multi"].append(t)
            results_train["mems_leaky_multi"].append(dmem)

            # StateLeaky single
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_mem, t = bench_stateleaky(
                int(steps), batch_size, channels, train=True, multi_beta=False
            )
            peak = get_peak_bytes(device)
            dmem = max(0, peak - baseline_mem) / 1024**2
            results_train["times_state_single"].append(t)
            results_train["mems_state_single"].append(dmem)

            # StateLeaky multi
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_mem, t = bench_stateleaky(
                int(steps), batch_size, channels, train=True, multi_beta=True
            )
            peak = get_peak_bytes(device)
            dmem = max(0, peak - baseline_mem) / 1024**2
            results_train["times_state_multi"].append(t)
            results_train["mems_state_multi"].append(dmem)

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
    METRIC_KEYS = [
        "times_leaky_single",
        "times_leaky_multi",
        "times_state_single",
        "times_state_multi",
        "mems_leaky_single",
        "mems_leaky_multi",
        "mems_state_single",
        "mems_state_multi",
    ]

    # Accumulators for mean/std across runs
    infer_sum = None
    infer_sumsq = None
    infer_meta = None  # holds batch_size/channels per cfg

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
            # initialize accumulators and metadata
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
        # Inference
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

        # Training
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_time_inf, ax_mem_inf = axes[0]
    ax_time_trn, ax_mem_trn = axes[1]

    cmap = plt.get_cmap("tab10")
    for idx, res in enumerate(results_infer):
        color = cmap(idx % 10)
        label_suffix = f"B{res['batch_size']}-C{res['channels']}"
        # Inference Time
        ax_time_inf.errorbar(
            TIMESTEPS,
            res["times_leaky_single"],
            yerr=res.get("std_times_leaky_single", None),
            fmt="-",
            color=color,
            label=f"Leaky single {label_suffix}",
            capsize=3,
        )
        ax_time_inf.errorbar(
            TIMESTEPS,
            res["times_leaky_multi"],
            yerr=res.get("std_times_leaky_multi", None),
            fmt="-.",
            color=color,
            label=f"Leaky multi {label_suffix}",
            capsize=3,
        )
        ax_time_inf.errorbar(
            TIMESTEPS,
            res["times_state_single"],
            yerr=res.get("std_times_state_single", None),
            fmt="--",
            color=color,
            label=f"StateLeaky single {label_suffix}",
            capsize=3,
        )
        ax_time_inf.errorbar(
            TIMESTEPS,
            res["times_state_multi"],
            yerr=res.get("std_times_state_multi", None),
            fmt=":",
            color=color,
            label=f"StateLeaky multi {label_suffix}",
            capsize=3,
        )

        # Inference Memory
        ax_mem_inf.errorbar(
            TIMESTEPS,
            res["mems_leaky_single"],
            yerr=res.get("std_mems_leaky_single", None),
            fmt="-",
            color=color,
            label=f"Leaky single {label_suffix}",
            capsize=3,
        )
        ax_mem_inf.errorbar(
            TIMESTEPS,
            res["mems_leaky_multi"],
            yerr=res.get("std_mems_leaky_multi", None),
            fmt="-.",
            color=color,
            label=f"Leaky multi {label_suffix}",
            capsize=3,
        )
        ax_mem_inf.errorbar(
            TIMESTEPS,
            res["mems_state_single"],
            yerr=res.get("std_mems_state_single", None),
            fmt="--",
            color=color,
            label=f"StateLeaky single {label_suffix}",
            capsize=3,
        )
        ax_mem_inf.errorbar(
            TIMESTEPS,
            res["mems_state_multi"],
            yerr=res.get("std_mems_state_multi", None),
            fmt=":",
            color=color,
            label=f"StateLeaky multi {label_suffix}",
            capsize=3,
        )

    for idx, res in enumerate(results_train):
        color = cmap(idx % 10)
        label_suffix = f"B{res['batch_size']}-C{res['channels']}"
        # Training Time
        ax_time_trn.errorbar(
            TIMESTEPS,
            res["times_leaky_single"],
            yerr=res.get("std_times_leaky_single", None),
            fmt="-",
            color=color,
            label=f"Leaky single (train) {label_suffix}",
            capsize=3,
        )
        ax_time_trn.errorbar(
            TIMESTEPS,
            res["times_leaky_multi"],
            yerr=res.get("std_times_leaky_multi", None),
            fmt="-.",
            color=color,
            label=f"Leaky multi (train) {label_suffix}",
            capsize=3,
        )
        ax_time_trn.errorbar(
            TIMESTEPS,
            res["times_state_single"],
            yerr=res.get("std_times_state_single", None),
            fmt="--",
            color=color,
            label=f"StateLeaky single (train) {label_suffix}",
            capsize=3,
        )
        ax_time_trn.errorbar(
            TIMESTEPS,
            res["times_state_multi"],
            yerr=res.get("std_times_state_multi", None),
            fmt=":",
            color=color,
            label=f"StateLeaky multi (train) {label_suffix}",
            capsize=3,
        )

        # Training Memory
        ax_mem_trn.errorbar(
            TIMESTEPS,
            res["mems_leaky_single"],
            yerr=res.get("std_mems_leaky_single", None),
            fmt="-",
            color=color,
            label=f"Leaky single (train) {label_suffix}",
            capsize=3,
        )
        ax_mem_trn.errorbar(
            TIMESTEPS,
            res["mems_leaky_multi"],
            yerr=res.get("std_mems_leaky_multi", None),
            fmt="-.",
            color=color,
            label=f"Leaky multi (train) {label_suffix}",
            capsize=3,
        )
        ax_mem_trn.errorbar(
            TIMESTEPS,
            res["mems_state_single"],
            yerr=res.get("std_mems_state_single", None),
            fmt="--",
            color=color,
            label=f"StateLeaky single (train) {label_suffix}",
            capsize=3,
        )
        ax_mem_trn.errorbar(
            TIMESTEPS,
            res["mems_state_multi"],
            yerr=res.get("std_mems_state_multi", None),
            fmt=":",
            color=color,
            label=f"StateLeaky multi (train) {label_suffix}",
            capsize=3,
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

    # mkdir if not exists
    os.makedirs("snn_performance", exist_ok=True)
    plt.tight_layout()
    plt.savefig("snn_performance/snn_performance_variants.png", dpi=150)
    plt.show()
