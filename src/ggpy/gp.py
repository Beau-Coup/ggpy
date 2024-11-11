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
        self.k = None

    def add_points(self, x, y, fit=False):
        # TODO: Docs and assume x y are correct shape. Makes no sense to reshape in here.
        new_x = x.reshape((-1, self.kernel.input_dims))

        if fit:
            if self.k is None:
                self.k = self.kernel.eval(self.train_inputs)

            if self.cho is None:
                self.train_inputs = np.vstack((self.train_inputs, new_x))
                self.train_outputs = np.vstack(
                    (self.train_outputs, y.reshape((-1, self.kernel.output_dims)))
                )
                self.fit()
                return
            else:
                # Assume that self.cho factorizes the kernel with all the current data.
                # Get the new kernel components
                old_size = self.train_inputs.shape[0]
                new_size = old_size + new_x.shape[0]
                old_k = self.k.copy()
                self.k = np.empty((new_size, new_size))
                self.k[:old_size, :old_size] = old_k
                self.k[:old_size, old_size:] = self.kernel.eval(
                    self.train_inputs, new_x
                )
                self.k[old_size:, :old_size] = self.k[:old_size, old_size:].T
                self.k[old_size:, old_size:] = self.kernel.eval(new_x)

                old_cho = self.cho.copy()
                self.cho = np.zeros((new_size, new_size))
                self.cho[:old_size, :old_size] = old_cho

                self.cho[:old_size, old_size:] = sla.solve_triangular(
                    old_cho.T,
                    self.k[:old_size, old_size:],
                    lower=True,
                )
                self.cho[old_size:, old_size:] = sla.cholesky(
                    self.k[old_size:, old_size:]
                    - self.cho[:old_size, old_size:].T @ self.cho[:old_size, old_size:]
                )

        self.train_inputs = np.vstack((self.train_inputs, new_x))
        self.train_outputs = np.vstack(
            (self.train_outputs, y.reshape((-1, self.kernel.output_dims)))
        )

    def fit(self):
        if self.k is None:
            self.k = self.kernel.eval(self.train_inputs)
        self.cho = sla.cholesky(
            self.k + np.identity(self.train_inputs.shape[0]) * 1e-10
        )

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
        return np.random.multivariate_normal(mean.flatten(), var)

    def variance(self, xs):
        cov = self.kernel.eval(xs, self.train_inputs)
        ps = sla.cho_solve(self.cho, cov.T)
        return cov @ ps

    def optimize(self):
        self.kernel.optimize(self.train_inputs, self.train_outputs)
