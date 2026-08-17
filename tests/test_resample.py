import numpy as np

from da_edgeformer.data.resample import time_weighted_mean


def test_time_weighted_mean() -> None:
    timestamps = np.array([0.0, 4.0, 12.0, 20.0])
    values = np.array([10.0, 20.0, 30.0, 40.0])
    grid, result = time_weighted_mean(timestamps, values, 10, 10, 2)
    np.testing.assert_array_equal(grid, [0, 10])
    np.testing.assert_allclose(result, [16.0, 28.0])


def test_long_gaps_are_not_interpolated() -> None:
    timestamps = np.array([0.0, 10.0, 100.0, 110.0])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    _, result = time_weighted_mean(timestamps, values, 10, 10, 2)
    assert result[0] == 1.0
    assert np.isnan(result[1:10]).all()
    assert result[10] == 3.0
