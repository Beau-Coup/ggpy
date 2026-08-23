import time

import jax
import jax.numpy as jnp
import numpy as np

from ggpy import GP, RBF, DMatrix, MultiGP, NaiveDMatrix

# The D-matrix structure both tests recover: nonzero only where (i + j) % 2 == 0.
PATTERN = np.zeros((6, 6), dtype=bool)
PATTERN[0::2, 0::2] = True  # even-even
PATTERN[1::2, 1::2] = True  # odd-odd


def test_dmatrix_kernel():
    """Test that the DMatrix kernel correctly implements a matrix with alternating zeros.

    This test:
    1. Creates a DMatrix kernel
    2. Generates random input points
    3. Evaluates the kernel
    4. Uses least squares to recover the 6x6 matrix structure
    5. Checks that the matrix has zeros in the right places
    """
    keys = jax.random.split(jax.random.key(42), 5)

    # Create the GP to sample from
    noise = 1e-10
    kernel = DMatrix(noise)
    num_samples = 100
    x_samples = jax.random.normal(keys[0], (num_samples, 6))

    model = MultiGP(kernel)
    samp = model.sample(keys[1], x_samples)

    # Define a random true dmatrix
    true_matrix = jax.random.uniform(keys[2], (6, 6))
    true_matrix = jnp.where(PATTERN, true_matrix, 0.0)

    y_samples = (true_matrix @ x_samples.T).T
    # add noise
    y_noise = y_samples + jax.random.normal(keys[3], y_samples.shape) * noise

    # Fit the GP
    model = model.add_points(x_samples, y_noise, fit=True)

    # Generate test points at which to back out the matrix
    num_test = 50
    x_test = jax.random.normal(keys[4], (num_test, 6))
    y_pred, _ = model.predict(x_test)

    # Now try to recover the matrix using least squares
    recovered_matrix = jnp.linalg.lstsq(x_test, y_pred, rcond=None)[0].T

    print(f"True: {true_matrix}")
    fit, unc = model.get_D()
    print(f"fit {fit}")
    print(f"uncertainty {unc}")

    # check that the recovered matrix is actually close to the real one
    assert np.allclose(true_matrix, recovered_matrix), (
        "Recovered matrix is not equal to the data-generating matrix"
    )

    # Now use the sample from the prior distribution and back out the matrix, checking that it has the right form
    generated_matrix = jnp.linalg.lstsq(x_samples, samp, rcond=None)[0].T

    assert (np.abs(generated_matrix[~PATTERN]) < 1e-3).all(), (
        f"The matrix has the wrong structure, non-zeros in odd entries {generated_matrix}"
    )
    assert (np.abs(generated_matrix[PATTERN]) > 1e-5).all(), (
        f"The matrix has the wrong structure, zeros in even entries {generated_matrix}"
    )


def test_naive_dmatrix_kernel():
    """Same recovery as above, but through the dense NaiveDMatrix kernel and a plain GP."""
    keys = jax.random.split(jax.random.key(43), 4)

    # Create the GP to sample from
    noise = 1e-10
    kernel = NaiveDMatrix(noise)
    num_samples = 100
    x_samples = jax.random.normal(keys[0], (num_samples, 6))

    model = GP(kernel)
    samp = model.sample(keys[1], x_samples)

    # A true dmatrix with entries spanning several orders of magnitude, to check
    # the fit is well conditioned and not just recovering something near unity.
    true_matrix = jnp.array(
        [
            [131029.16310579, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 653289.7, 0.0, 583323.62, 0.0, 748150.94],
            [0.0, 0.0, 784771.94, 0.0, 2304494.2, 0.0],
            [0.0, 583323.62, 0.0, 2021801.23904745, 0.0, -2254476.8],
            [0.0, 0.0, 2304494.2, 0.0, 69848000.0, 0.0],
            [0.0, 748150.94, 0.0, -2254476.8, 0.0, 53297996.8],
        ]
    )

    y_samples = (true_matrix @ x_samples.T).T
    # add noise
    y_noise = y_samples + jax.random.normal(keys[2], y_samples.shape) * jnp.sqrt(noise)

    # Fit the GP
    model = model.add_points(x_samples, y_noise, fit=True)

    # Generate test points at which to back out the matrix
    num_test = 50
    x_test = jax.random.normal(keys[3], (num_test, 6))
    y_pred = model.predict(x_test)

    # Now try to recover the matrix using least squares
    recovered_matrix = jnp.linalg.lstsq(x_test, y_pred, rcond=None)[0].T

    fit, unc = model.get_D()

    # check that the recovered matrix is actually close to the real one
    assert np.allclose(true_matrix, recovered_matrix, rtol=1e-5, atol=1e-5), (
        f"Recovered matrix {recovered_matrix} is not equal to the data-generating matrix {true_matrix}"
    )

    assert np.allclose(true_matrix, fit, rtol=1e-5, atol=1e-5), (
        f"Fit matrix {fit} is not equal to the data-generating matrix {true_matrix}"
    )

    # Now use the sample from the prior distribution and back out the matrix, checking that it has the right form
    generated_matrix = jnp.linalg.lstsq(
        x_samples, samp.reshape((num_samples, 6)), rcond=None
    )[0].T

    assert (np.abs(generated_matrix[~PATTERN]) < 1e-3).all(), (
        f"The matrix has the wrong structure, non-zeros in odd entries {generated_matrix}"
    )
    assert (np.abs(generated_matrix[PATTERN]) > 1e-5).all(), (
        f"The matrix has the wrong structure, zeros in even entries {generated_matrix}"
    )


def test_add_points():
    """Extending the Cholesky factor must match a full refit, and be cheaper.

    Both paths are run once to trigger compilation before anything is timed, and
    every timed result is blocked on -- JAX dispatch is asynchronous, so timing
    without ``block_until_ready`` would measure queueing rather than compute.
    """
    N = 64
    L_SCALE = 1.0
    L = 10.0

    xs = jnp.linspace(-L / 2, L / 2, N)
    x, y = jnp.meshgrid(xs, xs)
    test_map = x**2 + y**2

    rbf_kern = RBF(2, 0.1, L_SCALE)

    x_ind = jnp.arange(N)
    xind, yind = jnp.meshgrid(x_ind, x_ind)
    xs_ind = jnp.stack((xind, yind), axis=-1).reshape((-1, 2))

    base_x = xs_ind[::2]
    base_y = test_map[base_x[:, 0], base_x[:, 1]]

    partial_x = base_x[:-1]
    partial_y = test_map[partial_x[:, 0], partial_x[:, 1]]
    last_x = base_x[-1]
    last_y = test_map[last_x[0], last_x[1]]

    empty = GP(rbf_kern)
    fitted = GP(rbf_kern).add_points(partial_x, partial_y, True)
    fitted.cho.block_until_ready()

    # Warm up both jitted paths at exactly the shapes we are about to time.
    empty.add_points(base_x, base_y, True).cho.block_until_ready()
    fitted.add_points(last_x, last_y, True).cho.block_until_ready()

    start = time.perf_counter()
    full = empty.add_points(base_x, base_y, True)
    full.cho.block_until_ready()
    base_time = time.perf_counter() - start

    start = time.perf_counter()
    updated = fitted.add_points(last_x, last_y, True)
    updated.cho.block_until_ready()
    upd_time = time.perf_counter() - start

    assert np.allclose(full.cho, updated.cho)
    assert upd_time < base_time, (
        f"Incremental update ({upd_time:.4f}s) was not faster than a full "
        f"refit ({base_time:.4f}s)"
    )
