"""Shifted / centered discrete Fourier transforms.

The transforms here are *centered*: frequencies are measured from ``(N - 1) / 2``
rather than from zero, which is not what :func:`jax.numpy.fft.fft` computes. That is
why these are hand-rolled rather than delegating to ``jnp.fft``.

Only :func:`ifft_v` and :func:`ifft2_v` are used by the library proper (by
``Stationary.sample_prior``); the rest are the reference implementations they were
validated against, and are exercised by ``tests/test_fourier.py``.

Everything is written against ``jax.numpy`` and is jit/vmap-compatible: the
recursions and ``while`` loops below run over *static shapes*, so they unroll at
trace time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import _config  # noqa: F401  (enables x64)

__all__ = [
    "ft", "ift", "fft", "fft_v", "ifft", "ifft_v",
    "ft2", "ift2", "fft2", "fft2_v", "ifft2", "ifft2_v",
]


def ft(signal, f_shift, x_shift):
    """Brute-force centered DFT of a 1-D signal.

    For signals of even length the zero frequency component is ignored, which
    equates to assuming it is zero. Frequencies are centered at ``(N - 1) / 2``.

    :param signal: array of shape ``(N,)``, the signal to transform
    :param f_shift: frequency-space origin
    :param x_shift: position-space origin
    :return: the frequency-space representation, shape ``(N,)``.
        Note: for a pure real signal the real part of the transform is even and the
        imaginary part is odd, across the midpoint.
    """
    N = signal.shape[0]

    # Make the position and frequency arrays. Shape matters for broadcasting
    position = jnp.arange(N)
    frequency = jnp.arange(N).reshape(N, 1)

    # Apply the DFT, note that the frequency and position indices are the same
    return (
        jnp.exp(-2j * jnp.pi * (frequency - f_shift) * (position - x_shift) / N)
        @ signal
    )


def ift(transform, f_shift, x_shift):
    """Inverse of :func:`ft`, normalized with the cosmology convention."""
    N = transform.shape[0]

    # Have to change the indices to frequency components. (unitless)
    positions = jnp.arange(N)
    frequencies = jnp.arange(N).reshape(N, 1)

    signal = transform @ jnp.exp(
        2j * jnp.pi * (positions - x_shift) * (frequencies - f_shift) / N
    )

    return signal / N


def fft(signal, f_shift, x_shift):
    """Recursive Cooley-Tukey version of :func:`ft`, adapted to the shifted DFT.

    The recursion is over the static array length, so it unrolls at trace time.
    """
    N = signal.shape[0]

    # Need to convert the index of the array to the position correspondence
    k = jnp.arange(N)

    # Check the size of the array to determine if symmetry should be applied or not
    if N <= 32:
        return ft(signal, f_shift, x_shift)

    even = fft(signal[::2], f_shift, x_shift / 2)
    odd = fft(signal[1::2], f_shift, x_shift / 2)

    alpha = jnp.exp(1j * jnp.pi * x_shift)
    beta = jnp.exp(-2j * jnp.pi * (k - f_shift) / N)

    return jnp.concatenate(
        [even + beta[: N // 2] * odd, alpha * (even + beta[N // 2 :] * odd)]
    )


def fft_v(signal):
    """Vectorized bottom-up Cooley-Tukey version of :func:`ft`.

    Uses the centered-DFT shifts implied by the signal length rather than taking
    them as arguments.
    """
    N = signal.shape[0]
    shift = (N - 1) / N  # This only works for the centered DFT
    f_shift = (N - 1) / 2  # The f_shift is constant through-out
    N_min = min(N, 2)

    # Need to convert the index of the array to the position correspondence
    x = jnp.arange(N_min)
    k = x[:, None]

    term = jnp.exp(-2j * jnp.pi * (k - f_shift) * (x - shift) / N_min)
    X = term @ signal.reshape((N_min, -1)).astype(term.dtype)

    # Now build the array up
    while X.shape[0] < N:
        shift *= 2

        even = X[:, : X.shape[1] // 2]
        odd = X[:, X.shape[1] // 2 :]

        beta = jnp.exp(-1j * jnp.pi * (jnp.arange(X.shape[0]) - f_shift) / X.shape[0])[
            :, None
        ]
        alpha = jnp.exp(1j * jnp.pi * shift)

        X = jnp.vstack([even + beta * odd, alpha * (even - beta * odd)])

    return X.ravel()


def ifft(transform, f_shift, x_shift):
    """Recursive inverse of :func:`fft`, on the same principles."""
    N = transform.shape[0]

    # Need to convert the index of the array to the position correspondence
    x = jnp.arange(N)

    # Check the size of the array to determine if symmetry should be applied or not
    if N <= 32:
        return ift(transform, f_shift, x_shift)

    even = ifft(transform[::2], f_shift / 2, x_shift)
    odd = ifft(transform[1::2], f_shift / 2, x_shift)

    alpha = jnp.exp(-1j * jnp.pi * f_shift)
    beta = jnp.exp(2j * jnp.pi * (x - x_shift) / N)

    return (
        jnp.concatenate(
            [even + beta[: N // 2] * odd, alpha * (even + beta[N // 2 :] * odd)]
        )
        / 2
    )


def ifft_v(signal):
    """Vectorized inverse of :func:`fft_v`.

    Deliberately *not* normalized by ``N``: ``Stationary.sample_prior`` applies its
    own normalization.
    """
    N = signal.shape[0]
    shift = (N - 1) / N  # This only works for the centered DFT
    x_shift = (N - 1) / 2  # The x_shift is constant through-out
    N_min = min(N, 2)

    # Need to convert the index of the array to the position correspondence
    k = jnp.arange(N_min)
    x = k[:, None]

    term = jnp.exp(2j * jnp.pi * (x - x_shift) * (k - shift) / N_min)
    X = term @ signal.reshape((N_min, -1)).astype(term.dtype)

    # Now build the array up
    while X.shape[0] < N:
        shift *= 2

        even = X[:, : X.shape[1] // 2]
        odd = X[:, X.shape[1] // 2 :]

        beta = jnp.exp(1j * jnp.pi * (jnp.arange(X.shape[0]) - x_shift) / X.shape[0])[
            :, None
        ]
        alpha = jnp.exp(-1j * jnp.pi * shift)

        X = jnp.vstack([even + beta * odd, alpha * (even - beta * odd)])

    # Normalization removed for the purposes of this project
    return X.ravel()


def ft2(signal, f_shift, x_shift):
    """Brute-force centered 2-D DFT.

    Expressed as a pair of contractions rather than the naive double loop. Note the
    index convention this reproduces: the *row* index of ``signal`` pairs with the
    second output index and ``N``, while the *column* index pairs with the first
    output index and ``M``.

    :param signal: shape ``(M, N)``
    :param f_shift: shape ``(2,)``, frequency shift on each axis
    :param x_shift: shape ``(2,)``, position shift on each axis
    :return: shape ``(M, N)``
    """
    M, N = signal.shape[0], signal.shape[1]
    m, n = jnp.arange(M), jnp.arange(N)

    # A[u, b] and B[v, a] are the two separable phase factors.
    a = jnp.exp(-2j * jnp.pi * (m[:, None] - f_shift[0]) * (n[None, :] - x_shift[0]) / M)
    b = jnp.exp(-2j * jnp.pi * (n[:, None] - f_shift[1]) * (m[None, :] - x_shift[1]) / N)

    return jnp.einsum("ab,ub,va->uv", signal.astype(a.dtype), a, b)


def ift2(transform, f_shift, x_shift):
    """Brute-force inverse of :func:`ft2`, normalized by ``M * N``."""
    M, N = transform.shape[0], transform.shape[1]
    m, n = jnp.arange(M), jnp.arange(N)

    c = jnp.exp(2j * jnp.pi * (n[None, :] - f_shift[0]) * (m[:, None] - x_shift[0]) / M)
    d = jnp.exp(2j * jnp.pi * (m[None, :] - f_shift[1]) * (n[:, None] - x_shift[1]) / N)

    return jnp.einsum("ab,xb,ya->yx", transform.astype(c.dtype), c, d) / M / N


def fft2(signal, f_shift, x_shift):
    """Fast version of :func:`ft2`: a column-wise pass then a row-wise pass."""
    # Column-wise pass, then row-wise pass. vmap replaces the explicit loops.
    first_pass = jax.vmap(
        lambda col: fft(col, f_shift[1], x_shift[1]), in_axes=1, out_axes=1
    )(signal)

    return jax.vmap(lambda row: fft(row, f_shift[0], x_shift[0]))(first_pass)


def fft2_v(signal):
    """Fast vectorized version of :func:`ft2`, using :func:`fft_v` on each axis."""
    first_pass = jax.vmap(fft_v, in_axes=1, out_axes=1)(signal)
    return jax.vmap(fft_v)(first_pass)


def ifft2(transform, f_shift, x_shift):
    """Fast inverse of :func:`fft2`, on the same principles."""
    first_pass = jax.vmap(lambda row: ifft(row, f_shift[0], x_shift[0]))(transform)

    return jax.vmap(
        lambda col: ifft(col, f_shift[1], x_shift[1]), in_axes=1, out_axes=1
    )(first_pass)


def ifft2_v(transform):
    """Fast vectorized inverse of :func:`fft2_v`."""
    first_pass = jax.vmap(ifft_v)(transform)
    return jax.vmap(ifft_v, in_axes=1, out_axes=1)(first_pass)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 32
    space = jnp.linspace(0, 2 * jnp.pi, N * N).reshape(N, N)
    signal = space
    shift = jnp.array([(N - 1) / 2, (N - 1) / 2])

    true = ifft2(signal, shift, shift)
    test = ifft2_v(signal)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(jnp.real(true))
    axes[1].imshow(jnp.real(test))
    plt.show()

    print(jnp.allclose(true, test))

    """
    Correctness of the following pairs has been established (see tests/test_fourier.py):
    1. ft     and ift
    2. fft    and ifft
    3. fft2   and ifft2   (ift2 also inverts fft2)
    4. fft_v  and ifft_v  (up to the N that ifft_v deliberately does not divide by)
    5. fft2_v and ifft2_v (likewise, up to M * N)

    Caveat: ft2 returns the *transpose* of fft2 -- it pairs the row index of its
    input with the second output index. ift2 is transposed in the same way, so the
    two inverses agree with each other, but ift2(ft2(x)) is not x.
    """
