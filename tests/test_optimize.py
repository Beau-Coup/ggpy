"""Coverage for optax-based hyperparameter fitting.

The previous implementation drove ``scipy.optimize.minimize`` without gradients
(finite differences) and mutated ``kernel.hyper_params`` in place. This one takes
real gradients through ``jax.grad`` of the marginal likelihood and returns a new
kernel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ggpy import GP, RBF, DiagRBF, neg_log_likelihood, optimize
from ggpy.optimize import LOWER_FACTOR, UPPER_FACTOR


def _rbf_data(key, length_scale=1.0, n=60):
    """Draw a GP sample with a known length scale."""
    k_x, k_f = jax.random.split(key)
    x = jnp.sort(jax.random.uniform(k_x, (n, 1)) * 10.0, axis=0)
    truth = RBF(1, 1e-4, length_scale, 1.0)
    y = GP(truth).sample(k_f, x)
    return x, y


def test_gradients_are_finite_and_nonzero():
    """jax.grad of the NLL returns a kernel-shaped pytree with real gradients."""
    x, y = _rbf_data(jax.random.key(0))
    kernel = RBF(1, 1e-4, 2.0, 1.0)

    grads = jax.grad(neg_log_likelihood)(kernel, x, y)

    assert isinstance(grads, RBF), "gradient should share the kernel's pytree structure"
    assert np.isfinite(grads.length_scale), f"non-finite gradient {grads.length_scale}"
    assert abs(float(grads.length_scale)) > 0.0, "length_scale gradient is identically zero"


def test_optimize_decreases_likelihood():
    """The fitted kernel has a lower NLL than the one it started from."""
    x, y = _rbf_data(jax.random.key(1))
    kernel = RBF(1, 1e-4, 3.0, 1.0)

    before = float(neg_log_likelihood(kernel, x, y))
    fitted, history = optimize(kernel, x, y, steps=200, learning_rate=0.05,
                               return_history=True)
    after = float(neg_log_likelihood(fitted, x, y))

    assert after < before, f"NLL rose: {before} -> {after}"
    assert np.all(np.isfinite(history)), "NLL history contains non-finite values"
    assert float(history[-1]) < float(history[0])


def test_optimize_leaves_the_input_untouched():
    """Fitting returns a new kernel; the original is immutable."""
    x, y = _rbf_data(jax.random.key(2))
    kernel = RBF(1, 1e-4, 3.0, 1.0)

    fitted = optimize(kernel, x, y, steps=50, learning_rate=0.05)

    assert float(kernel.length_scale) == 3.0, "the input kernel was mutated"
    assert float(fitted.length_scale) != 3.0, "fitting did not move the parameter"
    # Untrained fields are carried through unchanged.
    assert float(fitted.scale) == float(kernel.scale)
    assert float(fitted.noise) == float(kernel.noise)


def test_optimize_respects_its_bounds():
    """Parameters stay inside the box the old L-BFGS-B fit was constrained to."""
    x, y = _rbf_data(jax.random.key(3))
    kernel = RBF(1, 1e-4, 3.0, 1.0)

    # A learning rate large enough to run away without the projection.
    fitted = optimize(kernel, x, y, steps=100, learning_rate=10.0)

    start = float(kernel.length_scale)
    assert start * LOWER_FACTOR - 1e-12 <= float(fitted.length_scale)
    assert float(fitted.length_scale) <= start * UPPER_FACTOR + 1e-12


def test_optimize_recovers_the_length_scale():
    """Fitting data drawn at one length scale moves the estimate toward it."""
    true_ls = 1.5
    x, y = _rbf_data(jax.random.key(4), length_scale=true_ls, n=80)

    start = RBF(1, 1e-4, 4.0, 1.0)
    fitted = optimize(start, x, y, steps=400, learning_rate=0.05)

    start_err = abs(float(start.length_scale) - true_ls)
    fit_err = abs(float(fitted.length_scale) - true_ls)
    assert fit_err < start_err, (
        f"length scale moved away from truth: {start.length_scale} -> "
        f"{fitted.length_scale} (true {true_ls})"
    )


def test_diagonal_kernel_optimizes():
    """DiagKernel's per-dimension likelihood is differentiable and fittable too."""
    keys = jax.random.split(jax.random.key(5), 2)
    x = jax.random.normal(keys[0], (40, 2))
    y = jax.random.normal(keys[1], (40, 3))

    kernel = DiagRBF(2, 3, 1e-3, jnp.array([3.0, 3.0, 3.0]))
    before = float(neg_log_likelihood(kernel, x, y))
    fitted = optimize(kernel, x, y, steps=200, learning_rate=0.02)
    after = float(neg_log_likelihood(fitted, x, y))

    assert after < before, f"NLL rose: {before} -> {after}"
    assert np.all(np.isfinite(np.asarray(fitted.length_scale)))
    # Each output dimension is fit independently, so they need not move together.
    assert fitted.length_scale.shape == (3,)


def test_optimize_rejects_kernels_without_parameters():
    from ggpy import MultiKernel

    kernel = MultiKernel(RBF(1, 1e-3, 1.0), RBF(1, 1e-3, 1.0))
    with pytest.raises(TypeError, match="no trainable parameters"):
        optimize(kernel, jnp.zeros((2, 2)), jnp.zeros((2, 2)))


def test_gp_optimize_refits_the_model():
    """``GP.optimize`` fits hyperparameters and returns a refitted model."""
    x, y = _rbf_data(jax.random.key(6), length_scale=1.5, n=50)

    model = GP(RBF(1, 1e-4, 4.0, 1.0)).add_points(x, y, fit=True)
    fitted = model.optimize(steps=300, learning_rate=0.05)

    assert float(model.kernel.length_scale) == 4.0, "the original model was mutated"
    assert fitted.cho is not None, "optimize should leave the model fitted"
    assert abs(float(fitted.kernel.length_scale) - 1.5) < abs(4.0 - 1.5)
    assert float(neg_log_likelihood(fitted.kernel, x, y)) < float(
        neg_log_likelihood(model.kernel, x, y)
    )
    # The refit factorization must be consistent with the new kernel.
    expected = GP(fitted.kernel).add_points(x, y, fit=True)
    assert np.allclose(fitted.cho, expected.cho)
