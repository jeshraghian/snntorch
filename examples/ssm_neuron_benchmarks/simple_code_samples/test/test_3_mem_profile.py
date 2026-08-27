import os
import sys
import importlib.util
import torch


# Ensure project root on sys.path when running directly
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_measure_activation_delta_positive_or_skip():
    if not torch.cuda.is_available():
        print(
            "CUDA not available - skipping test_measure_activation_delta_positive_or_skip"
        )
        return

    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "3_mem_profile.py")
    )
    mod = _load_module_from_path("mem_profile_module", mod_path)

    # Define a run step that allocates noticeable GPU memory and does a small compute
    def run_step():
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
        c = a @ b
        return c

    delta = mod.measure_activation_delta(run_step)
    assert isinstance(delta, int)
    assert delta >= 0
    # Expect strictly positive increase due to the large allocation/computation
    assert delta > 0


def test_measure_step_time_positive_or_skip():
    if not torch.cuda.is_available():
        print(
            "CUDA not available - skipping test_measure_step_time_positive_or_skip"
        )
        return

    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "3_mem_profile.py")
    )
    mod = _load_module_from_path("mem_profile_module", mod_path)

    def run_step():
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
        c = a @ b
        return c

    ms = mod.measure_step_time(run_step)
    assert isinstance(ms, float)
    assert ms > 0.0
    # Sanity bound: should not be absurdly large for a single step
    assert ms < 60000.0


if __name__ == "__main__":
    test_measure_activation_delta_positive_or_skip()
    print("test_measure_activation_delta_positive_or_skip passed (or skipped)")
    test_measure_step_time_positive_or_skip()
    print("test_measure_step_time_positive_or_skip passed (or skipped)")
