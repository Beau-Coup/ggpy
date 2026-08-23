"""Covariance kernels, as immutable JAX pytrees.

Every kernel is a frozen dataclass registered as a pytree (see :mod:`ggpy.tree`), so
it can be passed through ``jit``, ``grad`` and ``vmap`` directly. Hyperparameters are
traced leaves; ``input_dims`` / ``output_dims`` are static and live in the treedef.

Nothing here mutates: ``optimize`` (in :mod:`ggpy.optimize`) returns a *new* kernel
rather than reassigning hyperparameters in place.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

from . import (
    _config,  # noqa: F401  (enables x64)
    fourier,
)
from .tree import pytree_dataclass, set_frozen, static_field

__all__ = [
    "Kernel",
    "Stationary",
    "WhiteNoise",
    "RBF",
    "MultiKernel",
    "LinearTransformation",
    "DiagKernel",
    "Sum",
    "RevLinear",
    "kernel_eval",
]


def kernel_eval(kernel, x1, x2=None, noise=True):
    """Jitted entry point for ``kernel.eval``.

    ``kernel`` is a pytree whose *class* is part of its treedef, so JAX compiles a
    separate kernel per concrete type and per input shape -- no dispatch table needed.
    """
    return kernel._eval(x1, x2, noise)


class Kernel:
    """Base class for covariance kernels.

    Subclasses are :func:`~ggpy.tree.pytree_dataclass` types; this class only carries
    shared behavior. ``TRAINABLE`` names the fields :func:`ggpy.optimize.optimize`
    will fit.
    """

    TRAINABLE: tuple[str, ...] = ()

    def eval(self, x1, x2=None, noise=True):
        """Evaluate the covariance between ``x1`` and ``x2``.

        Parameters
        ----------
            x1: (n x input_dims) matrix of input points
            x2: (m x input_dims) matrix of input points, or None for ``x1`` against
                itself, in which case the noise term is included.
            noise: whether to add the noise term when ``x2`` is None.

        Returns
        -------
            The covariance. Shape depends on the kernel; scalar-output kernels give
            ``(n, m)``, vector-output kernels ``(n * output_dims, m * output_dims)``,
            and diagonal kernels ``(output_dims, n, m)``.
        """
        return kernel_eval(self, x1, x2, noise)

    def _eval(self, x1, x2, noise):
        raise NotImplementedError

    def gradient(self, x1, x2):
        """Return the gradient of the covariance of (X_1, X_2) with respect to X_1
        d/dx1 (k(x1, x2))

        Parameters
        ----------
            x1: n x input_dims matrix of input points at which to evaluate the gradient
            x2: m x input_dims matrix of points at which to evaluate the covariance

        Returns
        -------
            An output_dims x n x m x input_dims tensor of evaluated gradients.
        """
        raise NotImplementedError

    @property
    def hyper_params(self) -> tuple:
        """The values :func:`ggpy.optimize.optimize` fits, in ``TRAINABLE`` order."""
        return tuple(getattr(self, name) for name in self.TRAINABLE)

    def neg_log_likelihood(self, x, y):
        """Negative log marginal likelihood of input-output pairs under this kernel.

        Parameters
        ----------
            x: (n x input_dims) inputs
            y: (n x output_dims) outputs

        Returns
        -------
            A scalar. Lower is better; this is what ``optimize`` minimizes.
        """
        cov = self.eval(x)
        chol = jnp.linalg.cholesky(cov)
        sol = jsla.cho_solve((chol, True), y)

        # 2 * sum(log(diag(L))) rather than log(prod(diag(L)) ** 2), which underflows.
        ld = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol)))
        prod = jnp.sum(y * sol)
        const = x.shape[0] * jnp.log(2 * jnp.pi)

        return (ld + prod + const) * 0.5

    def optimize(self, x, y, **kwargs):
        """Fit this kernel's ``TRAINABLE`` hyperparameters to ``(x, y)``.

        Returns a *new* kernel; see :func:`ggpy.optimize.optimize` for the options.
        """
        from .optimize import optimize as _optimize

        return _optimize(self, x, y, **kwargs)

    def transform(self, A, noise):
        """Return a new kernel which is the augmented version of this kernel through A.

        Parameters
        ----------
            A: n x output_dims matrix by which to transform the output.

        Returns
        -------
            A kernel corresponding to the GP that results from transforming
            this GP by the linear transformation A.
        """
        return LinearTransformation(self, A, noise)


class Stationary(Kernel):
    """Kernels that depend only on x1 - x2 implement this.

    This allows for drawing random samples more efficiently.
    See the Wiener-Khinchin theorem.
    """

    def power_spectrum(self, k):
        raise NotImplementedError

    def sample_prior(self, key, n_points: int, length: float):
        """Generate a random sample from the prior.

        I am not sure this method actually produces a sample with the right
        properties, but it looks good enough.

        Parameters
        ----------
        key : jax.Array
            A ``jax.random`` PRNG key.
        n_points : int
            The number of points to sample use as the grid.
        length : float
            The width of the sample in spatial coordinates.
            Sample interval on [-size / 2, size /2].

        Returns
        -------
        ndarray ((n_points,) * input_dims + (output_dims,))
            The generated sample.
        """
        k_re, k_im = jax.random.split(key)
        half = int(n_points / 2)
        center = (n_points - 1) / 2.0

        # Handle 1D case
        if self.input_dims == 1:
            x = jnp.arange(n_points)

            # Calculate wave numbers for 1D
            k = jnp.abs(x - center) * (n_points - 1) / length / n_points

            # Apply power spectrum
            grid = self.power_spectrum(k) * length

            # Generate noise in frequency domain
            aj = jnp.zeros(n_points, dtype=jnp.complex128)
            aj = aj.at[:half].set(
                jax.random.normal(k_re, (half,)) + jax.random.normal(k_im, (half,)) * 1j
            )
            # Ensure Hermitian symmetry for real output
            aj = aj.at[half:].set(jnp.conj(aj[half - 1 :: -1]))

            # Scale by power spectrum
            aj = aj * (grid / jnp.sqrt(2))

            # Inverse FFT
            out = fourier.ifft_v(aj) * n_points / length

            assert jnp.allclose(
                jnp.zeros(n_points), jnp.imag(out), atol=1e-10, rtol=1e-08
            )
            return jnp.real(out)

        # Original 2D case
        x = jnp.arange(n_points)
        y = jnp.arange(n_points)

        x, y = jnp.meshgrid(x, y)
        k = (
            jnp.hypot(x - center, y - center)
            * (n_points - 1)
            / length
            / n_points
            / jnp.sqrt(2)
        )

        grid = self.power_spectrum(k) * length * jnp.sqrt(2)

        aj = jnp.zeros((n_points, n_points), dtype=jnp.complex128)
        # We are generating the right half of the noise
        aj = aj.at[:, half:].set(
            jax.random.normal(k_re, (n_points, half))
            + jax.random.normal(k_im, (n_points, half)) * 1j
        )
        # We want [-x, -y] = conj(x, y)
        aj = aj.at[:, :half].set(jnp.conj(aj[-1::-1, -1 : half - 1 : -1]))

        aj = aj * (grid / 2)

        out = fourier.ifft2_v(aj) / length / length  # Unnormalized

        assert jnp.allclose(
            jnp.zeros((n_points, n_points)), jnp.imag(out), atol=1e-10, rtol=1e-08
        )
        return jnp.real(out)


@pytree_dataclass(init=False)
class WhiteNoise(Kernel):
    """Pure diagonal noise; zero covariance between distinct points."""

    noise: jax.Array
    input_noise: jax.Array | None
    input_dims: int = static_field()
    output_dims: int = static_field()

    def __init__(self, input_dims, output_dims, noise, input_noise=None):
        set_frozen(
            self,
            input_dims=input_dims,
            output_dims=output_dims,
            noise=noise,
            input_noise=input_noise,
        )

    def _eval(self, x1, x2, noise):
        n = x1.shape[0] * self.output_dims
        if x2 is None:
            if noise:
                return jnp.eye(n) * self.noise
            return jnp.zeros((n, n))
        return jnp.zeros((n, x2.shape[0] * self.output_dims))

    def gradient(self, x1, x2):
        return jnp.zeros((x1.shape[0], self.input_dims, x2.shape[0]))


@pytree_dataclass(init=False)
class RBF(Stationary):
    """Squared-exponential kernel with a single length scale."""

    TRAINABLE = ("length_scale",)

    noise: jax.Array
    length_scale: jax.Array
    scale: jax.Array
    input_noise: jax.Array | None
    input_dims: int = static_field()
    output_dims: int = static_field()

    def __init__(self, input_dims, noise, length_scale, scale=1.0, input_noise=None):
        set_frozen(
            self,
            input_dims=input_dims,
            output_dims=1,
            noise=noise,
            length_scale=length_scale,
            scale=scale,
            input_noise=input_noise,
        )

    def _eval(self, x1, x2, noise):
        """Evaluate the kernel for x1 and x2

        Parameters:
            x1: (n x self.input_dims) matrix of input points
            x2: (m x self.input_dims) matrix of input points

        Returns:
            k(x1, x2): n x m matrix of pairwise kernel computations, i.e. the covariance
        """
        add = 0.0
        if x2 is None:
            x2 = x1
            if noise:
                add = jnp.eye(x1.shape[0]) * self.noise

        dx = jnp.expand_dims(x1, 1) - x2
        arg = (dx**2).sum(axis=-1) / (self.length_scale**2) * 0.5
        return self.scale * jnp.exp(-arg) + add

    def gradient(self, x1, x2):
        """Return the gradient of the covariance of (X_1, X_2) with respect to X_1
        d/dx1 (k(x1, x2))

        Parameters
        ----------
            x1: n x input_dims matrix of input points at which to evaluate the gradient
            x2: m x input_dims matrix of points at which to evaluate the covariance

        Returns
        -------
            An n x input_dims x m tensor of evaluated gradients.
        """
        dx = jnp.expand_dims(x1, 1) - x2
        if dx.ndim < 3:
            dx = dx.reshape(dx.shape + (self.input_dims,))
        dx = jnp.swapaxes(dx, 1, 2)  # n x input_dims x m
        dk = -dx / self.length_scale**2  # n x input_dims x m
        k = self.eval(x1, x2)  # n x m
        return dk * k[:, None, :]

    def get_length_scale(self):
        """Return the fit length scale of the kernel."""
        return self.length_scale

    def get_noise(self):
        """Return the fit noise value of the kernel."""
        return self.noise

    def power_spectrum(self, k):
        """Return the power spectrum of the kernel at a particular wave number.

        Given Wiener-Khinchin, this is the same as the Fourier Transform of the kernel.

        Parameters
        ----------
        k : float | jax.Array
            The wave number(s) at which to evaluate the power spectrum.

        Returns
        -------
        float | jax.Array
            The power of the given wave number(s).
        """
        return (
            jnp.sqrt(2 * jnp.pi)
            * self.length_scale
            * jnp.exp(-2 * ((jnp.pi * k * self.length_scale) ** 2))
        )


@pytree_dataclass(init=False)
class MultiKernel(Kernel):
    """Class to associate multiple uncorrelated kernels into a single kernel."""

    kernels: tuple
    noise: jax.Array
    input_dims: int = static_field()
    output_dims: int = static_field()

    def __init__(self, *kernels):
        kernels = tuple(kernels)
        set_frozen(
            self,
            kernels=kernels,
            input_dims=sum(k.input_dims for k in kernels),
            output_dims=sum(k.output_dims for k in kernels),
            noise=jsla.block_diag(*[jnp.atleast_2d(k.noise) for k in kernels]),
        )

    def split_inputs(self, inputs):
        """Split an input vector into a list of inputs for each kernel.

        Parameters
        ----------
            inputs: (n x self.input_dims) vector of inputs

        Returns
        -------
            a tuple of inputs for each kernel
        """
        outputs = []
        count = 0
        for k in self.kernels:
            outputs.append(inputs[:, count : count + k.input_dims])
            count += k.input_dims

        return tuple(outputs)

    def _eval(self, x1, x2, noise):
        """Put them all together brother.

        Returns the block-diagonal assembly of the component kernels, interleaved so
        that ``k(x_i, x_j)`` is a contiguous ``output_dims x output_dims`` block.
        """
        if x2 is None:
            x2inp = (None,) * len(self.kernels)
        else:
            x2inp = self.split_inputs(x2)

        x1inp = self.split_inputs(x1)
        # reshape the kernels to have the right form, so k(x1, x1), k(x1, x2)...
        # right now, the kernels each have the form k1(x1, x1), k1(x1, x2)
        # k(xi, xj) = diag(k1(xi, xj), k2(xi, xj), ...)
        m = x1.shape[0]
        n = x2.shape[0] if x2 is not None else x1.shape[0]
        d = self.output_dims

        # Viewed as (m, d, n, d), the result is just each component kernel dropped
        # into its own diagonal band -- a static slice update, not a scatter. The
        # component's own (m * out, n * out) layout reshapes to (m, out, n, out)
        # with no transpose, since both are output-dimension-minor.
        k = jnp.zeros((m, d, n, d))

        offset = 0
        for k_idx, kernel in enumerate(self.kernels):
            kvals = kernel._eval(x1inp[k_idx], x2inp[k_idx], noise)
            out = kernel.output_dims

            k = k.at[:, offset : offset + out, :, offset : offset + out].set(
                kvals.reshape(m, out, n, out)
            )
            offset += out

        return k.reshape(m * d, n * d)

    def gradient(self, x1, x2):
        """Same as above"""
        x1inp = self.split_inputs(x1)
        x2inp = self.split_inputs(x2)

        return jsla.block_diag(
            *[k.gradient(x1inp[i], x2inp[i]) for (i, k) in enumerate(self.kernels)]
        )


@pytree_dataclass(init=False)
class LinearTransformation(Kernel):
    """The kernel of ``A f(x)`` where ``f`` has kernel ``prior`` and ``A`` is linear."""

    prior: Kernel
    transformation: jax.Array
    noise: jax.Array
    input_dims: int = static_field()
    output_dims: int = static_field()

    def __init__(self, kernel, transformation, noise):
        assert transformation.shape[1] == kernel.output_dims, (
            f"Incorrect transformation, expected "
            f"{(transformation.shape[0], kernel.output_dims)}, got {transformation.shape}"
        )
        prior_noise = jnp.asarray(kernel.noise).reshape(
            (kernel.output_dims, kernel.output_dims)
        )
        set_frozen(
            self,
            prior=kernel,
            transformation=transformation,
            input_dims=kernel.input_dims,
            output_dims=transformation.shape[0],
            noise=noise + transformation @ prior_noise @ transformation.T,
        )

    def _eval(self, x1, x2, noise):
        """Evaluate the kernel for x1 and x2

        Returns:
            k(x1, x2): (output_dims * n) x (output_dims * m) covariance
        """
        add = 0.0
        if x2 is None:
            x2 = x1
            if noise:
                add = jnp.kron(self.noise, jnp.eye(x1.shape[0]))

        part = self.semi_left_eval(x1, x2)
        ot2 = jnp.kron(jnp.eye(x2.shape[0]), self.transformation.T)

        return part @ ot2 + add

    def semi_left_eval(self, x1, x2):
        """Evaluate the kernel for x1 and x2 but apply the transformation on the left.

        This is useful for intermediate calculations, and for recovering an estimate
        of the original GP posterior. It doesn't really make sense to have x2 = None
        here so this should not be used to compute the cholesky decomposition directly.

        Parameters
        ----------
            x1: (n x self.input_dims) matrix of input points
            x2: (m x self.input_dims) matrix of input points

        Returns
        -------
            k(x1, x2): (output_dims * n) x (prior.output_dims * m) covariance
        """
        k = self.prior.eval(x1, x2)
        outer_transf = jnp.kron(jnp.eye(x1.shape[0]), self.transformation)

        return outer_transf @ k

    def gradient(self, x1, x2):
        """Return the gradient of the covariance of (X_1, X_2) with respect to X_1
        d/dx1 (k(x1, x2))
        """
        return self.transformation @ self.prior.gradient(x1, x2)


class DiagKernel(Kernel):
    """Kernels whose output dimensions are mutually independent.

    ``eval`` returns a stacked ``(output_dims, n, m)`` array rather than a dense
    block matrix, which lets :class:`~ggpy.gp.MultiGP` factorize each output
    dimension separately.
    """

    def _eval(self, x1, x2, noise):
        """Evaluate the kernel for x1 and x2

        Returns:
            k(x1, x2): (output_dims x n x m) pairwise covariances
        """
        add = 0.0
        if x2 is None:
            x2 = x1
            if noise:
                add = jnp.eye(x1.shape[0])[None, :, :] * jnp.reshape(
                    self.noise, (-1, 1, 1)
                )

        return self._eval_all(x1, x2) + add

    def _eval_all(self, x1, x2):
        """Evaluate every output component, returning ``(output_dims, n, m)``."""
        raise NotImplementedError

    def gradient(self, x1, x2):
        """Return the gradient of the covariance of (X_1, X_2) with respect to X_1
        d/dx1 (k(x1, x2))

        Returns
        -------
            An output_dims x n x m x input_dims tensor of evaluated gradients.
        """
        raise NotImplementedError

    def neg_log_likelihood(self, x, y):
        """Negative log marginal likelihood, summed over independent output dims.

        Parameters
        ----------
            x: (n x input_dims)
            y: (n x output_dims)
        """
        covs = self.eval(x)
        # jnp.linalg.cholesky batches over the leading axis.
        chols = jnp.linalg.cholesky(covs)
        sols = jax.vmap(lambda c, t: jsla.cho_solve((c, True), t))(chols, y.T)

        lds = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chols, axis1=-2, axis2=-1)), axis=-1)
        prod = jnp.sum(y.T * sols)
        const = x.shape[0] * jnp.log(2 * jnp.pi)

        return (jnp.sum(lds) + prod + const) * 0.5


@pytree_dataclass(init=False)
class Sum(Kernel):
    """The kernel of the sum of two independent GPs."""

    k1: Kernel
    k2: Kernel
    noise: jax.Array
    input_noise: jax.Array | None
    input_dims: int = static_field()
    output_dims: int = static_field()

    def __init__(self, k1: Kernel, k2: Kernel, input_noise=None):
        assert k1.input_dims == k2.input_dims
        assert k1.output_dims == k2.output_dims
        set_frozen(
            self,
            k1=k1,
            k2=k2,
            input_dims=k1.input_dims,
            output_dims=k1.output_dims,
            noise=k1.noise + k2.noise,
            input_noise=input_noise,
        )

    def _eval(self, x1, x2, noise):
        return self.k1._eval(x1, x2, noise) + self.k2._eval(x1, x2, noise)

    def gradient(self, x1, x2):
        """Return the gradient of the covariance of (X_1, X_2) with respect to X_1
        d/dx1 (k(x1, x2))

        Returns
        -------
            An output_dims x n x m x input_dims tensor of evaluated gradients.
        """
        return self.k1.gradient(x1, x2) + self.k2.gradient(x1, x2)


@pytree_dataclass(init=False)
class RevLinear(Kernel):
    """Scalar linear kernel ``k(x1, x2) = x1 * x2``."""

    TRAINABLE = ("noise",)

    noise: jax.Array
    input_noise: jax.Array | None
    input_dims: int = static_field()
    output_dims: int = static_field()

    def __init__(self, noise, input_noise=None):
        set_frozen(
            self, input_dims=1, output_dims=1, noise=noise, input_noise=input_noise
        )

    def _eval(self, x1, x2, noise):
        """Evaluate the kernel for x1 and x2

        Returns:
            k(x1, x2): (n x m) matrix of pairwise kernel computations
        """
        add = 0.0
        if x2 is None:
            x2 = x1
            if noise:
                add = jnp.eye(x1.shape[0]) * self.noise

        prod = jnp.expand_dims(x1, 1) * x2
        return prod.squeeze(axis=-1) + add

    def gradient(self, x1, x2):
        """Return the gradient of the covariance of (X_1, X_2) with respect to X_1
        d/dx1 (k(x1, x2))
        """
        return x2[None, :] * jnp.ones(x1.shape[0])
