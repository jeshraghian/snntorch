#!/usr/bin/env python

"""Tests for `snntorch.spikevision.events_timeslices`."""

import numpy as np

from snntorch.spikevision import events_timeslices as et


def test_get_subsampled_coordinates():
    # columns: time, x, y, polarity
    evs = np.array([[0, 4, 6, 1], [10, 5, 9, 0]], dtype=np.int16)
    x_coords, y_coords = et.get_subsampled_coordinates(evs, ds_h=3, ds_w=2)
    np.testing.assert_array_equal(x_coords, [2, 2])
    np.testing.assert_array_equal(y_coords, [2, 3])

    # float coordinates are cast to integers
    x_coords, y_coords = et.get_subsampled_coordinates(
        evs.astype(float), ds_h=3, ds_w=2
    )
    assert x_coords.dtype.kind == "i" and y_coords.dtype.kind == "i"
    np.testing.assert_array_equal(x_coords, [2, 2])


def test_chunk_evs_pol_dvs():
    times = np.array([0, 500, 1500, 2500])
    # columns: x, y, polarity
    addrs = np.array([[0, 1, 0], [2, 3, 1], [2, 2, 1], [3, 3, 0]])
    chunks = et.chunk_evs_pol_dvs(
        times, addrs, deltat=1000, chunk_size=3, size=[2, 4, 4]
    )
    assert chunks.shape == (3, 2, 4, 4)
    assert chunks.sum() == 3
    assert chunks[1, 0, 0, 1] == 1
    assert chunks[1, 1, 2, 3] == 1
    assert chunks[2, 1, 2, 2] == 1


def test_frame_evs():
    times = np.array([0, 500, 1500, 2500])
    addrs = np.array([[0], [2], [2], [3]])
    chunks = et.frame_evs(
        times, addrs, deltat=1000, duration=3, size=[4], downsample=[1]
    )
    assert chunks.shape == (3, 4)
    np.testing.assert_array_equal(chunks[1], [1, 0, 1, 0])
    np.testing.assert_array_equal(chunks[2], [0, 0, 1, 0])
