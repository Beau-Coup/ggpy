import numpy as np
import scipy.linalg as sla
from numpy.typing import NDArray

from .kernel import Kernel


class GP:
    kernel: Kernel
    cho: NDArray | None
    train_inputs: NDArray
    train_outputs: NDArray

    def __init__(self, kernel, input_noise):
        self.kernel = kernel
        self.input_noise = input_noise
        self.train_outputs = np.empty((0, self.kernel.output_dims))
        self.train_inputs = np.empty((0, self.kernel.input_dims))
        self.cho = None

    def add_points(self, x, y):
        self.train_inputs = np.vstack(
            (self.train_inputs, x.reshape((-1, self.kernel.input_dims)))
        )
        self.train_outputs = np.vstack(
            (self.train_outputs, y.reshape((-1, self.kernel.output_dims)))
        )
        # TODO Recompute the cholesky decomposition efficiently

    def fit(self):
        # self.kernel.optimize(self.train_inputs, self.train_outputs)
        # Perform fit steps until convergence
        k = self.kernel.eval(self.train_inputs)
        self.cho = sla.cholesky(k)

    def cho_add_column(self, _):
        raise NotImplementedError

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
            ys, vs: (n x self.kernel.output_dims) ndarray the evaluated points of the posterior,
                and, optionally, the (n x self.kernel.output_dims) ndarray of variances.
        """
        assert xs.shape == (
            xs.shape[0],
            self.kernel.input_dims,
        ), f"Expected xs to be (n, {self.kernel.input_dims}), got {xs.shape}"

        if self.cho is None:
            ys = np.zeros((xs.shape[0], self.kernel.output_dims))
            if var:
                css = self.kernel.eval(xs, xs)
                return (ys, css)
            else:
                return ys

        ps = sla.cho_solve((self.cho, False), self.train_outputs.reshape((-1)))
        cov = self.kernel.eval(xs, self.train_inputs)
        ys = (cov @ ps).reshape((xs.shape[0], self.kernel.output_dims))

        if var:
            css = self.kernel.eval(xs, xs)
            pss = sla.cho_solve((self.cho, False), cov.T)
            return (ys, css - cov @ pss)

        return ys

    def sample(self, xs: NDArray) -> NDArray:
        """Draw a sample from the posterior distribution.

        This is an expensive function for many points, use Stationary.sample_prior if possible.

        Parameters
        ----------
        xs : NDArray (n, input_dims)
            The points at which to sample.

        Returns
        -------
        NDArray (n, output_dims)
            Values of f(x) of random sample from the posterior evaluated at each x in xs.

        """
        mean, var = self.predict(xs, var=True)
        cov_cho = sla.cholesky(
            var + 1e-10 * np.identity(var.shape[0]), lower=True
        )  # Stable my numerical

        entropy = np.random.rand(cov_cho.shape[1])
        return mean + (cov_cho @ entropy).reshape(mean.shape)

    def variance(self, xs):
        cov = self.kernel.eval(xs, self.train_inputs)
        ps = sla.cho_solve(self.cho, cov.T)
        return cov @ ps

    def optimize(self):
        self.kernel.optimize(self.train_inputs, self.train_outputs)
