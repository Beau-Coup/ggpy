"""Gaussian process models over the kernels in :mod:`ggpy.kernel`.

Models are immutable pytrees. Anything that used to mutate now returns a *new*
model, so a fit reads::

    model = GP(kernel)
    model = model.add_points(x, y, fit=True)
    ys, vs = model.predict(xs, var=True)

Because array shapes are static under ``jit``, each distinct training-set size
compiles once. That is inherent to growing a data set point by point; the
incremental ``add_points`` path still avoids the cubic cost of a full refactorization.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

from . import _config  # noqa: F401  (enables x64)
from .kernel import Kernel, LinearTransformation
from .tree import pytree_dataclass, set_frozen

__all__ = ["GP", "MultiGP", "JITTER"]

# Diagonal element added to Cholesky factorization.
JITTER = 1e-10


def _cholesky(k, n_total):
    """Upper-triangular Cholesky factor of ``k``, regularized by :data:`JITTER`."""
    return jsla.cholesky(k + jnp.identity(n_total) * JITTER)


def _extend_cholesky(old_cho, k_old_new, k_new_new):
    """Grow an upper Cholesky factor by a new block, without refactorizing.

    Given ``old_cho`` factorizing the existing block and the two new blocks, returns
    the factor of the enlarged matrix using its Schur complement.
    """
    top_right = jsla.solve_triangular(old_cho.T, k_old_new, lower=True)
    schur = k_new_new - top_right.T @ top_right
    bottom_right = jsla.cholesky(schur)

    old_size = old_cho.shape[0]
    new_size = old_size + k_new_new.shape[0]
    cho = jnp.zeros((new_size, new_size))
    cho = cho.at[:old_size, :old_size].set(old_cho)
    cho = cho.at[:old_size, old_size:].set(top_right)
    return cho.at[old_size:, old_size:].set(bottom_right)


def _extend_kernel(old_k, k_old_new, k_new_new):
    """Assemble the enlarged covariance matrix from its three new blocks."""
    old_size = old_k.shape[0]
    new_size = old_size + k_new_new.shape[0]
    k = jnp.zeros((new_size, new_size))
    k = k.at[:old_size, :old_size].set(old_k)
    k = k.at[:old_size, old_size:].set(k_old_new)
    k = k.at[old_size:, :old_size].set(k_old_new.T)
    return k.at[old_size:, old_size:].set(k_new_new)


@pytree_dataclass(init=False)
class GP:
    """A Gaussian process over a general (possibly correlated-output) kernel."""

    kernel: Kernel
    train_inputs: jax.Array
    train_outputs: jax.Array
    cho: jax.Array | None
    k: jax.Array | None
    input_noise: jax.Array | None

    def __init__(self, kernel, input_noise=None):
        set_frozen(
            self,
            kernel=kernel,
            input_noise=input_noise,
            train_inputs=jnp.zeros((0, kernel.input_dims)),
            train_outputs=jnp.zeros((0, kernel.output_dims)),
            cho=None,
            k=None,
        )

    def reset(self):
        """Return a fresh model with the same kernel and no training data."""
        return type(self)(self.kernel, self.input_noise)

    def _with_data(self, x, y, **changes):
        """Return a copy with ``(x, y)`` appended to the training set."""
        return self.replace(
            train_inputs=jnp.vstack((self.train_inputs, x)),
            train_outputs=jnp.vstack(
                (self.train_outputs, y.reshape((-1, self.kernel.output_dims)))
            ),
            **changes,
        )

    def add_points(self, x, y, fit=False):
        """Add training points, optionally refreshing the factorization.

        When the model is already fitted, the Cholesky factor is *extended* using the
        block structure rather than recomputed from scratch.

        Parameters
        ----------
            x: (n x input_dims) array of input points
            y: (n x output_dims) array of corresponding outputs
            fit: whether to update the factorization as well as the data

        Returns
        -------
            A new model. The receiver is left unchanged.
        """
        new_x = x.reshape((-1, self.kernel.input_dims))

        if not fit:
            return self._with_data(new_x, y)

        model = self
        if model.k is None and model.train_inputs.shape[0] != 0:
            model = model.replace(k=model.kernel.eval(model.train_inputs))

        if model.cho is None:
            return model._with_data(new_x, y).fit()

        # Assume that cho factorizes the kernel with all the current data, and
        # extend it with the blocks contributed by the new points.
        k_old_new = model.kernel.eval(model.train_inputs, new_x)
        k_new_new = model.kernel.eval(new_x) + jnp.identity(new_x.shape[0]) * JITTER

        return model._with_data(
            new_x,
            y,
            k=_extend_kernel(model.k, k_old_new, k_new_new),
            cho=_extend_cholesky(model.cho, k_old_new, k_new_new),
        )

    def fit(self):
        """Return a copy with the covariance factorized over all training data."""
        k = self.k if self.k is not None else self.kernel.eval(self.train_inputs)
        n_total = self.kernel.output_dims * self.train_inputs.shape[0]
        return self.replace(k=k, cho=_cholesky(k, n_total))

    def predict(self, xs, var=False):
        """Return the mean value of the posterior at the provided points.

        Parameters
        ----------
            xs: (n x self.kernel.input_dims) ndarray, points
                at which to evaluate the posterior.
            var: optional bool, whether or not to return the variance/uncertainty at
                each of the evaluated points.

        Returns
        -------
            ys: (n x self.kernel.output_dims) ndarray the evaluated points of the posterior,
            vs: optionally, the (n * self.kernel.output_dims x n * self.kernel.output_dims) ndarray of variances at each point.
        """
        assert xs.shape == (
            xs.shape[0],
            self.kernel.input_dims,
        ), f"Expected xs to be (n, {self.kernel.input_dims}), got {xs.shape}"

        if self.cho is None:
            ys = jnp.zeros((xs.shape[0], self.kernel.output_dims))
            if var:
                return (ys, self.kernel.eval(xs, xs))
            return ys

        cov = self.kernel.eval(self.train_inputs, xs)
        ps = jsla.cho_solve((self.cho.T, True), cov).T
        ys = (ps @ self.train_outputs.flatten()).reshape(
            (xs.shape[0], self.kernel.output_dims)
        )

        if var:
            css = self.kernel.eval(xs, xs)
            return (ys, css - ps @ cov)

        return ys

    def predict_original(self, xs, var=False):
        """Return the mean value of the original posterior of a linearly transformed GP at the provided points.

        Parameters
        ----------
            xs: (n x self.kernel.input_dims) ndarray, points
                at which to evaluate the posterior.
            var: optional bool, whether or not to return the variance/uncertainty at
                each of the evaluated points.

        Returns
        -------
            ys: (n x self.kernel.prior.output_dims) ndarray the evaluated points of the posterior,
            vs: optionally, the ndarray of variances at each point.
        """
        assert isinstance(self.kernel, LinearTransformation), (
            f"Expected a LinearTransformation kernel, not a {type(self.kernel).__name__}"
        )
        assert xs.shape == (
            xs.shape[0],
            self.kernel.input_dims,
        ), f"Expected xs to be (n, {self.kernel.input_dims}), got {xs.shape}"

        if self.cho is None:
            ys = jnp.zeros((xs.shape[0], self.kernel.prior.output_dims))
            if var:
                return (ys, self.kernel.eval(xs, xs))
            return ys

        # Posteriors have the form A k' K^-1 ...

        # Get the left product
        semi_k_star = self.kernel.semi_left_eval(self.train_inputs, xs)
        ps = jsla.cho_solve((self.cho.T, True), semi_k_star).T
        ys = (ps @ self.train_outputs.flatten()).reshape(
            (xs.shape[0], self.kernel.prior.output_dims)
        )

        if var:
            css = self.kernel.prior.eval(xs, xs)
            return (ys, css - (ps @ semi_k_star))

        return ys

    def sample(self, key, xs):
        """Draw a sample from the posterior distribution.

        This is an expensive function for many points, use Stationary.sample_prior if possible.

        Parameters
        ----------
        key : jax.Array
            A ``jax.random`` PRNG key.
        xs : Array (n, input_dims)
            The points at which to sample.

        Returns
        -------
        Array (n, output_dims)
            Values of f(x) of random sample from the posterior evaluated at each x in xs.
        """
        mean, var = self.predict(xs, var=True)
        res = _sample_gaussian(key, mean.flatten(), var)
        return res.reshape((xs.shape[0], self.kernel.output_dims))

    def variance(self, xs):
        """Posterior covariance at ``xs``, ignoring the mean."""
        cov = self.kernel.eval(xs, self.train_inputs)
        ps = jsla.cho_solve((self.cho.T, True), cov.T)
        return cov @ ps

    def optimize(self, **kwargs):
        """Fit the kernel's hyperparameters to the training data and refit.

        Returns a *new* model; see :func:`ggpy.optimize.optimize` for the options.
        """
        kernel = self.kernel.optimize(self.train_inputs, self.train_outputs, **kwargs)
        return self.replace(kernel=kernel, k=None, cho=None).fit()

    def get_line(self):
        """Recover the linear coefficient implied by the posterior."""
        assert self.cho is not None
        ps = jsla.cho_solve((self.cho.T, True), self.train_inputs).T
        return ps @ self.train_outputs

    def get_D(self):
        """Recover the 6x6 D matrix and its per-entry variance.

        Supported for the :class:`~ggpy.kernel.SmallDMatrix` and
        :class:`~ggpy.kernel.NaiveDMatrix` kernels. For the diagonal
        :class:`~ggpy.kernel.DMatrix` kernel use :meth:`MultiGP.get_D`.
        """
        assert self.cho is not None
        n = self.train_inputs.shape[0]

        if isinstance(self.kernel, SmallDMatrix):
            # Be lazy and just project the data this GP thinks belongs to the D term.
            a = self.train_inputs[:, 2::2]
            b = self.train_inputs[:, 1::2]

            k_star = jnp.zeros((6 * n, 14))
            k_star = k_star.at[::6, 0].set(self.train_inputs[:, 0])
            k_star = k_star.at[1::6, 1:4].set(b)
            k_star = k_star.at[2::6, 4:6].set(a)
            k_star = k_star.at[3::6, 6:9].set(b)
            k_star = k_star.at[4::6, 9:11].set(a)
            k_star = k_star.at[5::6, 11:14].set(b)
            k_star = k_star @ self.kernel.prior_unc

            prod = jsla.cho_solve((self.cho.T, True), k_star).T

            y = prod @ self.train_outputs.flatten()
            v = self.kernel.prior_unc - prod @ k_star

            # Turn y into the right form for D
            pattern = jnp.zeros((6, 6), dtype=bool)
            pattern = pattern.at[0, 0].set(True)
            pattern = pattern.at[2::2, 2::2].set(True)
            pattern = pattern.at[1::2, 1::2].set(True)

        elif isinstance(self.kernel, NaiveDMatrix):
            ns = jnp.zeros((2, 6, n))
            ns = ns.at[0, :3].set(self.train_inputs[:, ::2].T)
            ns = ns.at[1, 3:].set(self.train_inputs[:, 1::2].T)
            single = jnp.kron(jnp.eye(3)[:, :, None], ns)

            k_star = jnp.zeros((6 * n, 18))
            for i in range(6):
                k_star = k_star.at[i::6].set(single[i].T)
            k_star = k_star * self.kernel.scale

            prod = jsla.cho_solve((self.cho.T, True), k_star).T

            y = prod @ self.train_outputs.flatten()
            v = jnp.identity(18) * self.kernel.scale - prod @ k_star

            # Turn y into the right form for D
            pattern = jnp.zeros((6, 6), dtype=bool)
            pattern = pattern.at[::2, ::2].set(True)
            pattern = pattern.at[1::2, 1::2].set(True)

        else:
            raise TypeError(f"Expected a DMatrix kernel, got {self.kernel}")

        idx = jnp.nonzero(pattern)
        d = jnp.zeros((6, 6)).at[idx].set(y)
        dv = jnp.zeros((6, 6)).at[idx].set(jnp.diagonal(v))

        return d, dv


@partial(jax.jit, static_argnames=("var",))
def _multi_predict(chos, k_star, kss, outputs, var):
    """Per-output-dimension posterior mean and, optionally, covariance.

    ``chos`` is ``(output_dims, n, n)``; every array carries the output dimension on
    its leading axis, so one ``vmap`` covers what used to be a Python loop. ``var`` is
    static so the second triangular solve is not traced when it is not wanted.
    """

    def inner(cho, ks, kv, y):
        mean = ks @ jsla.cho_solve((cho, False), y)
        if not var:
            return mean, None
        pss = jsla.cho_solve((cho, False), ks.T)
        return mean, kv - ks @ pss

    # kss is None when var is False; don't try to map over it.
    in_axes = (0, 0, 0 if var else None, 0)
    return jax.vmap(inner, in_axes=in_axes)(chos, k_star, kss, outputs.T)


@pytree_dataclass(init=False)
class MultiGP(GP):
    """A Gaussian Process class for handling the special case where the kernel is diagonal.

    This reduces computational overhead by a factor of the number of output dimensions, cubed.
    By treating each output dimension independently, we can perform much more efficient calculations.

    ``cho`` and ``k`` hold ``(output_dims, n, n)`` stacks -- one factorization per
    output dimension -- and every operation is a ``vmap`` over that leading axis.
    """

    @property
    def chos(self):
        """The per-output-dimension Cholesky factors, stacked."""
        return self.cho

    @property
    def ks(self):
        """The per-output-dimension covariance matrices, stacked."""
        return self.k

    def add_points(self, x, y, fit=False):
        """Add new training points to the GP.

        Efficiently updates the Cholesky decomposition when new points are added
        by using the block structure of the decomposition.

        Parameters
        ----------
            x: (n x input_dims) array of input points
            y: (n x output_dims) array of corresponding outputs
            fit: bool, whether to update the model fit after adding points

        Returns
        -------
            A new model. The receiver is left unchanged.
        """
        new_x = x.reshape((-1, self.kernel.input_dims))
        new_y = y.reshape((-1, self.kernel.output_dims))

        if not fit:
            return self._with_data(new_x, new_y)

        model = self
        if model.k is None and model.train_inputs.shape[0] != 0:
            n = model.train_inputs.shape[0]
            model = model.replace(
                k=model.kernel.eval(model.train_inputs) + jnp.eye(n) * JITTER
            )

        if model.cho is None:
            # First fit - just add the points and do a full fit
            return model._with_data(new_x, new_y).fit()

        # Update each dimension's kernel and Cholesky decomposition
        k_old_new = model.kernel.eval(model.train_inputs, new_x)
        k_new_new = model.kernel.eval(new_x) + jnp.eye(new_x.shape[0]) * JITTER

        return model._with_data(
            new_x,
            new_y,
            k=jax.vmap(_extend_kernel)(model.k, k_old_new, k_new_new),
            cho=jax.vmap(_extend_cholesky)(model.cho, k_old_new, k_new_new),
        )

    def fit(self):
        """Fit the GP model using the current training data.

        For a diagonal kernel, we fit each output dimension independently.
        """
        if not self.train_inputs.shape[0]:
            return self

        n = self.train_inputs.shape[0]
        # Add a small nugget for numerical stability
        k = self.kernel.eval(self.train_inputs) + jnp.eye(n) * JITTER

        return self.replace(k=k, cho=jax.vmap(jsla.cholesky)(k))

    def predict(self, xs, var=False):
        """Return the mean value of the posterior at the provided points.

        For a diagonal kernel, we can predict each output dimension independently.

        Parameters
        ----------
            xs: (n x input_dims) ndarray, points at which to evaluate the posterior
            var: bool, whether to return the variance/uncertainty

        Returns
        -------
            ys: (n x output_dims) ndarray of predicted mean values
            vs: (output_dims x n x n) ndarray of predicted variances (if var=True)
        """
        assert xs.shape == (
            xs.shape[0],
            self.kernel.input_dims,
        ), f"Expected xs to be (n, {self.kernel.input_dims}), got {xs.shape}"

        if self.cho is None:
            ys = jnp.zeros((xs.shape[0], self.kernel.output_dims))
            if var:
                return ys, self.kernel.eval(xs)
            return ys, None

        k_star = self.kernel.eval(xs, self.train_inputs)
        kss = self.kernel.eval(xs, xs) if var else None

        ys, vs = _multi_predict(self.cho, k_star, kss, self.train_outputs, var)

        return ys.T, vs

    def sample(self, key, xs):
        """Draw a sample from the posterior distribution.

        For a diagonal kernel, we can sample each output dimension independently.

        Parameters
        ----------
        key : jax.Array
            A ``jax.random`` PRNG key.
        xs : Array (n, input_dims)
            The points at which to sample.

        Returns
        -------
        Array (n, output_dims)
            Values of f(x) of random sample from the posterior evaluated at each x in xs.
        """
        means, variances = self.predict(xs, var=True)
        assert variances is not None

        keys = jax.random.split(key, self.kernel.output_dims)
        samples = jax.vmap(_sample_gaussian)(keys, means.T, variances)

        return samples.T

    def variance(self, xs):
        """Return the variance/uncertainty at the provided points.

        For a diagonal kernel, we compute variance for each output dimension independently.

        Parameters
        ----------
        xs : Array (n, input_dims)
            The points at which to evaluate variance.

        Returns
        -------
        Array (output_dims, n, n)
            Posterior covariance for each output dimension.
        """
        _, var = self.predict(xs, var=True)
        return var
