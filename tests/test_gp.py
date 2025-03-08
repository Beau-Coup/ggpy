import time

import numpy as np
import pytest

from ggpy import GP, RBF, DMatrix, MultiGP


def test_dmatrix_kernel():
    """Test that the DMatrix kernel correctly implements a matrix with alternating zeros.

    This test:
    1. Creates a DMatrix kernel
    2. Generates random input points
    3. Evaluates the kernel
    4. Uses least squares to recover the 6x6 matrix structure
    5. Checks that the matrix has zeros in the right places
    """
    np.random.seed(42)

    # Create the GP to sample from
    noise = 1e-6
    kernel = DMatrix(noise)
    num_samples = 1000
    x_samples = np.random.randn(num_samples, 6)

    model = MultiGP(kernel, None)
    samp = model.sample(x_samples)

    # Dmatrix structure, i.e. (i + j) % 2 == 0
    pattern = np.zeros((6, 6), dtype=bool)
    pattern[0::2, 0::2] = True  # even-even
    pattern[1::2, 1::2] = True  # odd-odd

    # Define a random true dmatrix
    true_matrix = np.random.rand(6, 6)
    true_matrix[~pattern] = 0

    y_samples = (true_matrix @ x_samples.T).T
    # add noise
    y_noise = y_samples + np.random.normal(0, noise, y_samples.shape)

    # Fit the GP
    model.add_points(x_samples, y_noise, fit=True)

    # Generate test points at which to back out the matrix
    num_test = 50
    x_test = np.random.randn(num_test, 6)
    y_pred, _ = model.predict(x_test)

    # Now try to recover the matrix using least squares
    recovered_matrix = np.linalg.lstsq(x_test, y_pred, rcond=None)[0].T

    # check that the recovered matrix is actually close to the real one
    assert np.allclose(
        true_matrix, recovered_matrix
    ), f"Recovered matrix is not equal to the data-generating matrix"

    # Now use the sample from the prior distribution and back out the matrix, checking that it has the right form
    generated_matrix = np.linalg.lstsq(x_samples, samp, rcond=None)[0].T

    assert (
        np.abs(generated_matrix[~pattern]) < 1e-3
    ).all(), f"The matrix has the wrong structure, non-zeros in odd entries {generated_matrix}"
    assert (
        np.abs(generated_matrix[pattern]) > 1e-5
    ).all(), (
        f"The matrix has the wrong structure, zeros in even entries {generated_matrix}"
    )


def test_add_points():
    N = 64
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
