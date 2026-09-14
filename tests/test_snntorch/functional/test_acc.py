#!/usr/bin/env python

"""Tests for accuracy metrics."""

import torch

from snntorch.functional import acc

torch.manual_seed(42)


class TestAcc:
    def test_accuracy_rate(self):
        spk_out = torch.zeros((5, 2, 4))
        spk_out[:, 0, 0] = 1
        spk_out[:, 1, 3] = 1
        targets = torch.tensor([0, 3])

        assert acc.accuracy_rate(spk_out, targets) == 1.0

    def test_accuracy_rate_population_code(self):
        spk_out = torch.zeros((5, 2, 4))
        spk_out[:, 0, 0:2] = 1
        spk_out[:, 1, 2:4] = 1
        targets = torch.tensor([0, 1])

        accuracy = acc.accuracy_rate(
            spk_out, targets, population_code=True, num_classes=2
        )

        assert accuracy == 1.0

    def test_accuracy_rate_population_code_dtype(self):
        # float64 inputs should not be downcast, which would lose
        # the ordering of close scores
        spk_out = torch.tensor(
            [[[1.00000002, 1.00000005]]], dtype=torch.float64
        )
        targets = torch.tensor([1])

        pop_code = acc._population_code(spk_out, num_classes=2, num_outputs=2)
        accuracy = acc.accuracy_rate(
            spk_out, targets, population_code=True, num_classes=2
        )

        assert pop_code.dtype == torch.float64
        assert accuracy == 1.0
