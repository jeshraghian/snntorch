"""Latency interpolation must retain fractional membrane targets."""

import pytest
import torch

from snntorch import spikegen


@pytest.fixture(params=[torch.float32, torch.float64])
def default_dtype(request):
    original = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
@pytest.mark.parametrize("on_target,off_target", [(1, 0), (1.25, 0.25)])
@pytest.mark.parametrize("zero_time", [False, True])
def test_latency_interpolation_dtype(
    dtype, on_target, off_target, zero_time, default_dtype
):
    times = torch.tensor([[0 if zero_time else 1, 4], [2, 3]], dtype=dtype)
    original = times.clone()
    output_dtype = dtype if times.is_floating_point() else default_dtype
    expected = torch.full((5, 2, 2), off_target, dtype=output_dtype)
    for sample in range(2):
        for neuron in range(2):
            spike = int(times[sample, neuron])
            if spike == 0:
                expected[0, sample, neuron] = on_target
            else:
                for step in range(spike + 1):
                    expected[step, sample, neuron] = (
                        off_target + step / spike * (on_target - off_target)
                    )

    actual = spikegen.latency_interpolate(
        times, num_steps=5, on_target=on_target, off_target=off_target
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(times, original)
