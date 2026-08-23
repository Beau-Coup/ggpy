"""Direct coverage for the shifted-DFT module.

``fourier`` had no tests before the JAX port despite ``Stationary.sample_prior``
depending on it. These pin the relationships between the brute-force reference
implementations and their fast counterparts.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ggpy import fourier

N = 64
M = 16


@pytest.fixture(scope="module")
def sig():
    rng = np.random.default_rng(7)
    return jnp.asarray(rng.standard_normal(N) + 1j * rng.standard_normal(N))


@pytest.fixture(scope="module")
def sig2():
    rng = np.random.default_rng(11)
    return jnp.asarray(
        rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
    )


CENTER_1D = (N - 1) / 2
CENTER_2D = jnp.array([(M - 1) / 2, (M - 1) / 2])


def test_fft_matches_brute_force(sig):
    """The recursive transform agrees with the O(N^2) definition."""
    assert np.allclose(fourier.fft(sig, CENTER_1D, CENTER_1D), fourier.ft(sig, CENTER_1D, CENTER_1D))
    assert np.allclose(fourier.ifft(sig, CENTER_1D, CENTER_1D), fourier.ift(sig, CENTER_1D, CENTER_1D))


def test_1d_round_trips(sig):
    """Both 1-D pairs invert each other."""
    assert np.allclose(fourier.ift(fourier.ft(sig, CENTER_1D, CENTER_1D), CENTER_1D, CENTER_1D), sig)
    assert np.allclose(fourier.ifft(fourier.fft(sig, CENTER_1D, CENTER_1D), CENTER_1D, CENTER_1D), sig)


def test_vectorized_1d_round_trip(sig):
    """``ifft_v`` inverts ``fft_v`` up to the ``N`` it deliberately does not divide by.

    ``sample_prior`` relies on that missing normalization, so it is part of the
    contract rather than a bug.
    """
    assert np.allclose(fourier.ifft_v(fourier.fft_v(sig)) / N, sig)


def test_2d_round_trips(sig2):
    """The fast 2-D forward transform is inverted by both 2-D inverses."""
    fwd = fourier.fft2(sig2, CENTER_2D, CENTER_2D)
    assert np.allclose(fourier.ifft2(fwd, CENTER_2D, CENTER_2D), sig2)
    assert np.allclose(fourier.ift2(fwd, CENTER_2D, CENTER_2D), sig2)
    assert np.allclose(
        fourier.ift2(sig2, CENTER_2D, CENTER_2D),
        fourier.ifft2(sig2, CENTER_2D, CENTER_2D),
    )


def test_vectorized_2d_round_trip(sig2):
    """``ifft2_v`` inverts ``fft2_v`` up to the same missing ``M * M``."""
    assert np.allclose(fourier.ifft2_v(fourier.fft2_v(sig2)) / (M * M), sig2)


def test_ft2_is_transposed_relative_to_fft2(sig2):
    """``ft2`` returns the transpose of ``fft2``.

    Pre-existing behavior, carried over verbatim from the NumPy implementation: the
    brute-force ``ft2`` pairs the row index of the input with the *second* output
    index, so its result is transposed with respect to the fast ``fft2``. The
    inverses agree with each other because ``ift2`` writes its output transposed too,
    which cancels out. Documented here rather than silently "fixed", since callers
    may depend on either convention.
    """
    assert np.allclose(
        fourier.ft2(sig2, CENTER_2D, CENTER_2D),
        fourier.fft2(sig2, CENTER_2D, CENTER_2D).T,
    )
    assert not np.allclose(
        fourier.ft2(sig2, CENTER_2D, CENTER_2D),
        fourier.fft2(sig2, CENTER_2D, CENTER_2D),
    )


def test_transforms_are_jittable(sig, sig2):
    """The transforms trace cleanly, which is what sample_prior needs of them."""
    assert np.allclose(jax.jit(fourier.ifft_v)(sig), fourier.ifft_v(sig))
    assert np.allclose(jax.jit(fourier.ifft2_v)(sig2), fourier.ifft2_v(sig2))
