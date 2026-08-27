import torch


def measure_activation_delta(run_step):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # At this point, model + inputs should already be on GPU.
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    _ = run_step()  # e.g., one training step

    peak = torch.cuda.max_memory_allocated()
    delta_bytes = peak - baseline
    return delta_bytes


def measure_step_time(run_step):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    _ = run_step()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end)  # milliseconds


if __name__ == "__main__":

    def run_step():
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        return a @ b

    delta = measure_activation_delta(run_step)
    ms = measure_step_time(run_step)
    print("mem_profile demo: delta_bytes =", delta, " step_ms =", ms)
