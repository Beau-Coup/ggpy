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
        if self.cho is None:
            self.fit_step()
        assert self.cho is not None

        count = 0
        while True:
            old_cho = self.cho.copy()
            self.fit_step()
            count += 1
            if ((old_cho - self.cho) ** 2).sum() < 1e-6:
                break

    def fit_step(self):
        k = self.kernel.eval(self.train_inputs)
        # if self.cho is not None:
        #     grad_k = self.kernel.gradient(self.train_inputs, self.train_inputs)
        #     x_nom = sla.cho_solve((self.cho, False), self.train_outputs)
        #     grad = grad_k @ x_nom
        #     g = self.input_noise * grad @ grad.T
        #     k += g

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

        ps = sla.cho_solve((self.cho, False), self.train_outputs.reshape((-1)))
        cov = self.kernel.eval(xs, self.train_inputs)

        ys = (cov @ ps).reshape((xs.shape[0], self.kernel.output_dims))
        if var:
            css = self.kernel.eval(xs, xs)
            pss = sla.cho_solve((self.cho, False), cov.T)
            return (ys, css - cov @ pss)

        return ys

    def variance(self, xs):
        cov = self.kernel.eval(xs, self.train_inputs)
        ps = sla.cho_solve(self.cho, cov.T)
        return cov @ ps

    def optimize(self):
        self.kernel.optimize(self.train_inputs, self.train_outputs)
