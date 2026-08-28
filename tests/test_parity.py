"""Numerical parity against the pre-JAX NumPy/SciPy implementation.

``tests/reference.npz`` holds golden outputs captured from the original NumPy code
(see ``tests/gen_reference.py`` for how they were produced). These tests assert the
JAX port reproduces them, which is what catches transcription slips in the rewritten
``.at[].set()`` scatters and index arithmetic -- errors the behavioral tests would
not notice.

The fixture is a frozen baseline, not something to regenerate: the NumPy
implementation it came from no longer exists in the tree.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from ggpy import GP, RBF, MultiKernel, RevLinear, fourier

REFERENCE = Path(__file__).parent / "reference.npz"

# Compare on a relative footing: covariance entries and Cholesky factors here span
# several orders of magnitude.
RTOL = 1e-7
ATOL = 1e-7


@pytest.fixture(scope="module")
def ref():
    assert REFERENCE.exists(), (
        f"{REFERENCE} is missing; it is a checked-in fixture and cannot be regenerated"
    )
    return np.load(REFERENCE)


def check(got, expected, name):
    got = np.asarray(got)
    expected = np.asarray(expected)
    assert got.shape == expected.shape, (
        f"{name}: shape {got.shape} != reference {expected.shape}"
    )
    assert np.allclose(got, expected, rtol=RTOL, atol=ATOL), (
        f"{name}: max abs diff {np.abs(got - expected).max():.3e} "
        f"(reference scale {np.abs(expected).max():.3e})"
    )


def test_kernel_eval_parity(ref):
    """Every kernel's eval and gradient reproduce the NumPy implementation."""
    x1_1d, x2_1d = ref["in_x1_1d"], ref["in_x2_1d"]
    x1_2d, x2_2d = ref["in_x1_2d"], ref["in_x2_2d"]
    x1_4d, x2_4d = ref["in_x1_4d"], ref["in_x2_4d"]

    rbf = RBF(2, 0.1, 2.0, 1.5)
    rev = RevLinear(0.01)
    mk = MultiKernel(RBF(2, 1e-2, 2.0, 1.0), RBF(2, 1e-3, 1.0, 0.8))
    lt = mk.transform(jnp.array([[1.0, 0.5], [0.25, -1.0], [0.0, 2.0]]), 0.01)

    cases = {
        "rbf_cross": rbf.eval(x1_2d, x2_2d),
        "rbf_self": rbf.eval(x1_2d),
        "rbf_self_nonoise": rbf.eval(x1_2d, noise=False),
        "rbf_grad": rbf.gradient(x1_2d, x2_2d),
        "rbf_power": rbf.power_spectrum(jnp.linspace(0.0, 3.0, 11)),
        "rev_cross": rev.eval(x1_1d, x2_1d),
        "rev_self": rev.eval(x1_1d),
        "rev_grad": rev.gradient(x1_1d, x2_1d),
        "multi_cross": mk.eval(x1_4d, x2_4d),
        "multi_self": mk.eval(x1_4d),
        "multi_noise": mk.noise,
        "lt_noise": lt.noise,
        "lt_cross": lt.eval(x1_4d, x2_4d),
        "lt_self": lt.eval(x1_4d),
        "lt_semi_left": lt.semi_left_eval(x1_4d, x2_4d),
    }
    for name, got in cases.items():
        check(got, ref[name], name)


def test_likelihood_parity(ref):
    """The NLL matches, now returned as a scalar rather than a (1, 1) array."""
    rbf = RBF(2, 0.1, 2.0, 1.5)

    got = rbf.neg_log_likelihood(ref["in_x1_2d"], ref["in_y_rbf"])
    assert np.ndim(got) == 0, "neg_log_likelihood should return a scalar"
    check(got, np.squeeze(ref["rbf_likelihood"]), "rbf_likelihood")


def test_gp_parity(ref):
    """GP fit, predict and incremental update all match."""
    x1_1d = ref["in_x1_1d"]
    x1_2d, x2_2d = ref["in_x1_2d"], ref["in_x2_2d"]
    x1_4d, x2_4d = ref["in_x1_4d"], ref["in_x2_4d"]
    y_train, y_lt = ref["in_y_train"], ref["in_y_lt"]

    def rbf():
        return RBF(2, 0.1, 2.0, 1.5)

    gp = GP(rbf()).add_points(x1_2d, y_train, fit=True)
    mean, var = gp.predict(x2_2d, var=True)
    check(gp.cho, ref["gp_cho"], "gp_cho")
    check(mean, ref["gp_mean"], "gp_mean")
    check(var, ref["gp_var"], "gp_var")

    # Extending the factorization must land in the same place as a full refit.
    inc = GP(rbf()).add_points(x1_2d[:-2], y_train[:-2], fit=True)
    inc = inc.add_points(x1_2d[-2:], y_train[-2:], fit=True)
    check(inc.cho, ref["gp_cho_incremental"], "gp_cho_incremental")
    check(inc.cho, ref["gp_cho"], "gp_cho_incremental vs full")

    m0, v0 = GP(rbf()).predict(x2_2d, var=True)
    check(m0, ref["gp_empty_mean"], "gp_empty_mean")
    check(v0, ref["gp_empty_var"], "gp_empty_var")

    mk = MultiKernel(RBF(2, 1e-2, 2.0, 1.0), RBF(2, 1e-3, 1.0, 0.8))
    lt = mk.transform(jnp.array([[1.0, 0.5], [0.25, -1.0], [0.0, 2.0]]), 0.01)
    gp_lt = GP(lt).add_points(x1_4d, y_lt, fit=True)
    check(gp_lt.cho, ref["gp_lt_cho"], "gp_lt_cho")
    mean, var = gp_lt.predict(x2_4d, var=True)
    check(mean, ref["gp_lt_mean"], "gp_lt_mean")
    check(var, ref["gp_lt_var"], "gp_lt_var")
    mean, var = gp_lt.predict_original(x2_4d, var=True)
    check(mean, ref["gp_lt_orig_mean"], "gp_lt_orig_mean")
    check(var, ref["gp_lt_orig_var"], "gp_lt_orig_var")

    gp_rev = GP(RevLinear(1e-2)).add_points(x1_1d, 1.3 * x1_1d, fit=True)
    check(gp_rev.cho, ref["gp_rev_cho"], "gp_rev_cho")
    check(gp_rev.get_line(), ref["gp_rev_line"], "gp_rev_line")


def test_fourier_parity(ref):
    """All twelve transforms reproduce the NumPy implementation."""
    sig, sig2 = ref["in_fourier_sig"], ref["in_fourier_sig2"]
    shift = jnp.array([(16 - 1) / 2, (16 - 1) / 2])

    cases = {
        "fourier_ft": fourier.ft(sig, 31.5, 31.5),
        "fourier_ift": fourier.ift(sig, 31.5, 31.5),
        "fourier_fft": fourier.fft(sig, 31.5, 31.5),
        "fourier_ifft": fourier.ifft(sig, 31.5, 31.5),
        "fourier_fft_v": fourier.fft_v(sig),
        "fourier_ifft_v": fourier.ifft_v(sig),
        "fourier_ft2": fourier.ft2(sig2, shift, shift),
        "fourier_ift2": fourier.ift2(sig2, shift, shift),
        "fourier_fft2": fourier.fft2(sig2, shift, shift),
        "fourier_ifft2": fourier.ifft2(sig2, shift, shift),
        "fourier_fft2_v": fourier.fft2_v(sig2),
        "fourier_ifft2_v": fourier.ifft2_v(sig2),
    }
    for name, got in cases.items():
        check(got, ref[name], name)
