import time

import jax
import jax.numpy as jnp
import numpy as np

from ggpy import GP, RBF, DiagRBF, MultiGP


def test_multigp_matches_independent_gps():
    """MultiGP must equal fitting one scalar GP per output dimension.

    ``MultiGP`` exists purely as an optimization: for a diagonal kernel it
    factorizes ``output_dims`` matrices of size ``n`` instead of one of size
    ``n * output_dims``. The cheap path has to agree with the obvious one, which is
    what this pins down -- mean, covariance and the Cholesky factors themselves.
    """
    keys = jax.random.split(jax.random.key(0), 3)
    noise = 1e-3
    length_scales = jnp.array([1.0, 2.0, 0.5])
    scales = jnp.array([1.0, 0.5, 2.0])

    x = jax.random.normal(keys[0], (40, 2))
    y = jax.random.normal(keys[1], (40, 3))
    xs = jax.random.normal(keys[2], (15, 2))

    multi = MultiGP(DiagRBF(2, 3, noise, length_scales, scales))
    multi = multi.add_points(x, y, fit=True)
    mean, var = multi.predict(xs, var=True)

    assert mean.shape == (15, 3)
    assert var.shape == (3, 15, 15)

    for i in range(3):
        single = GP(RBF(2, noise, length_scales[i], scales[i]))
        single = single.add_points(x, y[:, i : i + 1], fit=True)
        s_mean, s_var = single.predict(xs, var=True)

        assert np.allclose(multi.chos[i], single.cho), (
            f"output dim {i}: Cholesky factor differs from the scalar GP"
        )
        assert np.allclose(mean[:, i], s_mean.squeeze(1)), (
            f"output dim {i}: posterior mean differs from the scalar GP"
        )
        assert np.allclose(var[i], s_var), (
            f"output dim {i}: posterior covariance differs from the scalar GP"
        )


def test_multigp_add_points_matches_refit():
    """MultiGP's incremental Cholesky extension matches a full refit, per dimension."""
    keys = jax.random.split(jax.random.key(1), 2)
    kernel = DiagRBF(2, 3, 1e-3, jnp.array([1.0, 2.0, 0.5]))

    x = jax.random.normal(keys[0], (30, 2))
    y = jax.random.normal(keys[1], (30, 3))

    full = MultiGP(kernel).add_points(x, y, fit=True)

    inc = MultiGP(kernel).add_points(x[:-5], y[:-5], fit=True)
    inc = inc.add_points(x[-5:], y[-5:], fit=True)

    assert np.allclose(full.chos, inc.chos)
    assert np.allclose(full.train_inputs, inc.train_inputs)
    assert np.allclose(full.predict(x, var=False)[0], inc.predict(x, var=False)[0])


def test_multigp_sample_recovers_its_own_statistics():
    """Samples drawn from a diagonal prior have the per-dimension scale asked for."""
    key = jax.random.key(2)
    scales = jnp.array([0.25, 4.0])
    kernel = DiagRBF(2, 2, 1e-6, jnp.array([2.0, 2.0]), scales)

    xs = jax.random.normal(key, (400, 2))
    keys = jax.random.split(key, 200)
    samples = jnp.stack([MultiGP(kernel).sample(k, xs) for k in keys])

    assert samples.shape == (200, 400, 2)
    assert np.all(np.isfinite(samples))

    # Prior variance at any point is the kernel's scale for that output dimension.
    empirical = samples.var(axis=(0, 1))
    assert np.allclose(empirical, scales, rtol=0.15), (
        f"prior sample variance {empirical} does not match the kernel scales {scales}"
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
