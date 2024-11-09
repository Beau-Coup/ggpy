import time

import numpy as np

from ggpy import GP, RBF


def test_add_points():
    N = 128
    L_SCALE = 1.0
    L = 10.0

    xs = np.linspace(-L / 2, L / 2, N)
    x, y = np.meshgrid(xs, xs)
    xs = np.stack((x, y), axis=-1)
    xs = xs.reshape((-1, 2))
    test_map = x**2 + y**2

    rbf_kern = RBF(2, 0.1, L_SCALE)
    model = GP(rbf_kern, None)

    # Test that sampling twice is both faster and yields the same result
    x_ind = np.arange(N)
    xind, yind = np.meshgrid(x_ind, x_ind)
    xs_ind = np.stack((xind, yind), axis=-1)
    xs_ind = xs_ind.reshape((-1, 2))

    base_samples_x = xs_ind[::2]
    base_samples_y = test_map[tuple(base_samples_x.T)]

    start = time.time()
    model.add_points(base_samples_x, base_samples_y, True)
    base_time = time.time() - start
    base_cho = model.cho.copy()

    model = GP(rbf_kern, None)
    new_base_samples_x = base_samples_x[:-1]
    new_base_samples_y = test_map[tuple(new_base_samples_x.T)]
    second_samples_x = base_samples_x[-1]
    second_samples_y = test_map[tuple(second_samples_x.T)]

    model.add_points(new_base_samples_x, new_base_samples_y, True)
    start = time.time()
    model.add_points(second_samples_x, second_samples_y, True)
    upd_time = time.time() - start

    assert np.allclose(base_cho, model.cho)
    assert upd_time < base_time
    assert False, f"{upd_time}, {base_time}"
