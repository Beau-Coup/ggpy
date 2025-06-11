import numpy as np
import scipy.linalg as sla
from numpy.typing import NDArray

from .kernel import *


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

    def reset(self):
        self.train_outputs = np.empty((0, self.kernel.output_dims))
        self.train_inputs = np.empty((0, self.kernel.input_dims))
        self.cho = None
        self.k = None

    def add_points(self, x, y, fit=False):
        # TODO: Docs and assume x y are correct shape. Makes no sense to reshape in here.
        new_x = x.reshape((-1, self.kernel.input_dims))

        if fit:
            if self.k is None and self.train_inputs.shape[0] != 0:
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
                self.k[old_size:, old_size:] = (
                    self.kernel.eval(new_x) + np.identity(new_x.shape[0]) * 1e-10
                )

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

        x = (
            self.k
            + np.identity(self.kernel.output_dims * self.train_inputs.shape[0]) * 1e-10
        )
        self.cho = sla.cholesky(x.squeeze())

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
            ys: (n x self.kernel.output_dims) ndarray the evaluated points of the posterior,
            vs: optionally, the (n * self.kernel.output_dims x n * self.kernel.output_dims) ndarray of variances at each point.
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

        cov = self.kernel.eval(self.train_inputs, xs)
        ps = sla.cho_solve((self.cho.T, True), cov).T
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
            ys: (n x self.kernel.output_dims) ndarray the evaluated points of the posterior,
            vs: optionally, the (n * self.kernel.output_dims x n * self.kernel.output_dims) ndarray of variances at each point.
        """
        assert isinstance(self.kernel, LinearTransformation), (
            f"Expected a LinearTransformation kernel, not a {type(self.kernel).__name__}"
        )
        assert xs.shape == (
            xs.shape[0],
            self.kernel.input_dims,
        ), f"Expected xs to be (n, {self.kernel.input_dims}), got {xs.shape}"

        if self.cho is None:
            ys = np.zeros((xs.shape[0], self.kernel.prior.output_dims))
            if var:
                css = self.kernel.eval(xs, xs)
                return (ys, css)
            else:
                return ys

        # Posteriors have the form A k' K^-1 ...

        # Get the left product
        semi_k_star = self.kernel.semi_left_eval(self.train_inputs, xs)
        ps = sla.cho_solve((self.cho.T, True), semi_k_star).T
        ys = (ps @ self.train_outputs.flatten()).reshape(
            (xs.shape[0], self.kernel.prior.output_dims)
        )

        if var:
            css = self.kernel.prior.eval(xs, xs)
            return (ys, css - (ps @ semi_k_star))

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
        res = np.random.multivariate_normal(mean.flatten(), var)
        return res.reshape((xs.shape[0], self.kernel.output_dims))

    def variance(self, xs):
        cov = self.kernel.eval(xs, self.train_inputs)
        ps = sla.cho_solve(self.cho, cov.T)
        return cov @ ps

    def optimize(self):
        self.kernel.optimize(self.train_inputs, self.train_outputs)

    def get_D(self):
        import matplotlib.pyplot as plt

        if isinstance(self.kernel, LinearTransformation):
            priors, u = self.predict_original(self.train_inputs, var=True)

            x = self.train_inputs[:, 0].copy()
            o = x.argsort()
            x = x[o]

            dp = u.diagonal().reshape(priors.shape)
            y = priors[o, 0]
            v = dp[o, 0]

            plt.plot(x, y, marker=".")
            plt.fill_between(x, y - v, y + v, alpha=0.5)
            plt.plot(x, self.train_outputs[o, 0], marker=".")
            plt.show()

            kern = self.kernel.prior.kernels[0]
            # print(kern)
            gp = GP(kern, None)

            # Add data to the GP
            xs = self.train_inputs[:, :6]
            ys = priors[:, :6]
            # print(xs.shape, ys.shape)
            gp.add_points(xs, ys, fit=True)

            return gp.get_D()

        if isinstance(self.kernel, SmallDMatrix):
            assert self.cho is not None

            # Be lazy and just make another GP to get D but using the data this GP thinks belong to the D term.
            n = self.train_inputs.shape[0]

            a = self.train_inputs[:, 2::2]
            b = self.train_inputs[:, 1::2]

            k_star = np.zeros((6 * n, 14))
            k_star[::6, 0] = self.train_inputs[:, 0]
            k_star[1::6, 1:4] = b
            k_star[2::6, 4:6] = a
            k_star[3::6, 6:9] = b
            k_star[4::6, 9:11] = a
            k_star[5::6, 11:14] = b
            k_star = k_star @ self.kernel.prior_unc

            prod = sla.cho_solve((self.cho.T, True), k_star).T

            y = prod @ self.train_outputs.flatten()
            v = self.kernel.prior_unc - prod @ k_star

            # Turn y into the right form for D
            pattern = np.zeros((6, 6), dtype=bool)
            pattern[0, 0] = True
            pattern[2::2, 2::2] = True
            pattern[1::2, 1::2] = True
            pattern = np.nonzero(pattern)

            d = np.zeros((6, 6))
            d[pattern] = y
            dv = np.zeros_like(d)
            dv[pattern] = v.diagonal()

        elif isinstance(self.kernel, NaiveDMatrix):
            assert self.cho is not None

            # Be lazy and just make another GP to get D but using the data this GP thinks belong to the D term.
            n = self.train_inputs.shape[0]

            ns = np.zeros((2, 6, n))
            ns[0, :3] = self.train_inputs[:, ::2].T
            ns[1, 3:] = self.train_inputs[:, 1::2].T
            single = np.kron(np.eye(3)[:, :, None], ns)

            k_star = np.zeros((6 * n, 18))
            k_star[::6] = single[0].T
            k_star[1::6] = single[1].T
            k_star[2::6] = single[2].T
            k_star[3::6] = single[3].T
            k_star[4::6] = single[4].T
            k_star[5::6] = single[5].T
            k_star *= self.kernel.scale

            prod = sla.cho_solve((self.cho.T, True), k_star).T

            y = prod @ self.train_outputs.flatten()
            v = np.identity(18) * self.kernel.scale - prod @ k_star

            # Turn y into the right form for D
            pattern = np.zeros((6, 6), dtype=bool)
            pattern[::2, ::2] = True
            pattern[1::2, 1::2] = True
            pattern = np.nonzero(pattern)

            d = np.zeros((6, 6))
            d[pattern] = y
            dv = np.zeros_like(d)
            dv[pattern] = v.diagonal()
        else:
            raise Exception(f"Expected a DMatrix kernel, got {self.kernel}")

        return d, dv

        # def get_D(self):
        #     # assert isinstance(self.kernel, NaiveDMatrix), "Not a D Matrix kernel"
        #     # I don't know see the derivation
        #     n = self.train_inputs.shape[0]
        #
        #     ns = np.zeros((2, 6, n))
        #     ns[0, :3] = self.train_inputs[:, ::2].T
        #     ns[1, 3:] = self.train_inputs[:, 1::2].T
        #     single = np.kron(np.eye(3)[:, :, None], ns)
        #
        #     k_star = np.zeros((6 * n, 18))
        #     k_star[::6] = single[0].T
        #     k_star[1::6] = single[1].T
        #     k_star[2::6] = single[2].T
        #     k_star[3::6] = single[3].T
        #     k_star[4::6] = single[4].T
        #     k_star[5::6] = single[5].T
        #
        #
        #     # Cheating see derivation by assuming white noise
        #     noise = np.eye(6 * self.train_inputs.shape[0]) + self.kernel.noise
        #     k = np.eye(18)
        #
        #     m = noise.shape[0] + 18
        #     sig = np.zeros((m, m))
        #     sig[:-18, :-18] = noise
        #     sig[-18:, -18:] = k
        #
        #     # weight = k_star.T @ k_star / (1.0 + self.kernel.noise)
        #     aug_kstar = np.concatenate([k_star, np.eye(18)])
        #     lp = np.linalg.lstsq(sig.T, aug_kstar)[0].T
        #
        #     aug_outputs = np.zeros(lp.shape[1])
        #     aug_outputs[: 6 * n] = self.train_outputs.flatten()
        #     ghat = np.linalg.pinv(lp @ aug_kstar) @ lp @ aug_outputs
        #
        #     pattern = np.zeros((6, 6), dtype=bool)
        #     pattern[::2, ::2] = True
        #     pattern[1::2, 1::2] = True
        #
        #     d = np.zeros(pattern.shape)
        #     d[pattern] = ghat
        #
        #     return d

    def get_line(self):
        assert self.cho is not None
        ps = sla.cho_solve((self.cho.T, True), self.train_inputs).T
        return ps @ self.train_outputs


class MultiGP(GP):
    """A Gaussian Process class for handling the special case where the kernel is diagonal.

    This reduces computational overhead by a factor of the number of output dimensions, cubed.
    By treating each output dimension independently, we can perform much more efficient calculations.
    """

    kernel: DiagKernel
    chos: list  # list of cholesky decomps for each output dimension
    ks: list  # kernels for each output dimension
    train_inputs: NDArray
    train_outputs: NDArray

    def __init__(self, kernel, input_noise):
        super().__init__(kernel, input_noise)
        self.chos = []
        self.ks = []

    def add_points(self, x, y, fit=False):
        """Add new training points to the GP.

        Efficiently updates the Cholesky decomposition when new points are added
        by using the block structure of the decomposition.

        Parameters
        ----------
            x: (n x input_dims) array of input points
            y: (n x output_dims) array of corresponding outputs
            fit: bool, whether to update the model fit after adding points
        """
        new_x = x.reshape((-1, self.kernel.input_dims))
        new_y = y.reshape((-1, self.kernel.output_dims))

        if fit:
            # Initialize kernel matrices if needed
            if not self.ks and self.train_inputs.shape[0] != 0:
                # Get kernel evaluations for each output dimension
                kernel_evals = self.kernel.eval(self.train_inputs)
                n = self.train_inputs.shape[0]
                for i in range(self.kernel.output_dims):
                    self.ks.append(kernel_evals[i] + np.eye(n) * 1e-10)

            if not self.chos:
                # First fit - just add the points and do a full fit
                self.train_inputs = np.vstack((self.train_inputs, new_x))
                self.train_outputs = np.vstack((self.train_outputs, new_y))
                self.fit()
                return
            else:
                # Update each dimension's kernel and Cholesky decomposition
                old_size = self.train_inputs.shape[0]
                new_size = old_size + new_x.shape[0]

                # Get kernel evaluations for the new points
                kernel_evals = self.kernel.eval(
                    self.train_inputs, new_x
                )  # output_dims x old_size x new_points
                kernel_new_new = self.kernel.eval(
                    new_x
                )  # output_dims x new_points x new_points

                # For each output dimension, update kernel matrix and Cholesky factor
                for i in range(self.kernel.output_dims):
                    # Get old kernel matrix and create new larger one
                    old_k = self.ks[i]
                    new_k = np.empty((new_size, new_size))
                    new_k[:old_size, :old_size] = old_k

                    # Fill in new blocks
                    k_old_new = kernel_evals[i]
                    new_k[:old_size, old_size:] = k_old_new
                    new_k[old_size:, :old_size] = k_old_new.T
                    new_k[old_size:, old_size:] = (
                        kernel_new_new[i] + np.eye(new_x.shape[0]) * 1e-10
                    )

                    # Update Cholesky factor using block structure
                    old_cho = self.chos[i]
                    new_cho = np.zeros((new_size, new_size))
                    new_cho[:old_size, :old_size] = old_cho

                    # Compute new components
                    new_cho[:old_size, old_size:] = sla.solve_triangular(
                        old_cho.T, k_old_new, lower=True
                    )

                    # Schur complement
                    schur = (
                        new_k[old_size:, old_size:]
                        - new_cho[:old_size, old_size:].T
                        @ new_cho[:old_size, old_size:]
                    )
                    new_cho[old_size:, old_size:] = sla.cholesky(schur)

                    # Store updated matrices
                    self.ks[i] = new_k
                    self.chos[i] = new_cho

        # Always update the training data
        self.train_inputs = np.vstack((self.train_inputs, new_x))
        self.train_outputs = np.vstack((self.train_outputs, new_y))

    def fit(self):
        """Fit the GP model using the current training data.

        For a diagonal kernel, we fit each output dimension independently.
        """
        if not self.train_inputs.shape[0]:
            return

        # Clear existing kernel matrices and Cholesky factors
        self.ks = []
        self.chos = []

        # Get kernel evaluations for each output dimension
        kernel_evals = self.kernel.eval(self.train_inputs)

        # For each output dimension, compute Cholesky factor
        n = self.train_inputs.shape[0]
        for i in range(self.kernel.output_dims):
            # Extract kernel for this dimension
            k_i = kernel_evals[i]

            # Add small nugget for numerical stability
            k_i = k_i + np.eye(n) * 1e-10

            # Store kernel and compute Cholesky decomposition
            self.ks.append(k_i)
            self.chos.append(sla.cholesky(k_i))

    def predict(self, xs, var=False) -> tuple[NDArray, NDArray | None]:
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

        n_test = xs.shape[0]

        if not self.chos:
            ys = np.zeros((n_test, self.kernel.output_dims))
            if var:
                vs = self.kernel.eval(xs)
                return ys, vs
            return ys, None

        k_star = self.kernel.eval(xs, self.train_inputs)
        kss = self.kernel.eval(xs, xs)

        def _inner(idx):
            ps = sla.cho_solve(
                (self.chos[idx], False), self.train_outputs[:, idx].reshape((-1))
            )
            ys = (k_star[idx] @ ps).reshape((xs.shape[0]))
            if var:
                pss = sla.cho_solve((self.chos[idx], False), k_star[idx].T)
                v = kss[idx] - k_star[idx] @ pss
                return ys, v
            return ys

        res = [_inner(i) for i in range(self.kernel.output_dims)]
        if var:
            ys, vs = tuple(map(np.array, zip(*res)))
        else:
            ys, vs = np.array(res), None

        return ys.T, vs

    def sample(self, xs: NDArray) -> NDArray:
        """Draw a sample from the posterior distribution.

        For a diagonal kernel, we can sample each output dimension independently.

        Parameters
        ----------
        xs : NDArray (n, input_dims)
            The points at which to sample.

        Returns
        -------
        NDArray (n, output_dims)
            Values of f(x) of random sample from the posterior evaluated at each x in xs.
        """
        means, vars = self.predict(xs, var=True)
        assert vars is not None

        n_test = xs.shape[0]
        samples = np.zeros((n_test, self.kernel.output_dims))
        for i in range(self.kernel.output_dims):
            samples[:, i] = np.random.multivariate_normal(
                means[:, i].flatten(), vars[i]
            )

        return samples

    def variance(self, xs):
        """Return the variance/uncertainty at the provided points.

        For a diagonal kernel, we compute variance for each output dimension independently.

        Parameters
        ----------
        xs : NDArray (n, input_dims)
            The points at which to evaluate variance.

        Returns
        -------
        NDArray (n, output_dims)
            Variance of the posterior at each point.
        """
        _, var = self.predict(xs, var=True)
        return var

    def optimize(self):
        """Optimize the kernel hyperparameters using the training data."""
        self.kernel.optimize(self.train_inputs, self.train_outputs)
        # After optimizing hyperparameters, we need to refit the model
        self.fit()

    def get_D(self):
        assert isinstance(self.kernel, DMatrix), "Not a D Matrix kernel"

        n = self.train_inputs.shape[0]

        ns = np.zeros((2, 6, n))
        ns[0, :3] = self.train_inputs[:, ::2].T
        ns[1, 3:] = self.train_inputs[:, 1::2].T
        k_star = np.kron(np.eye(3)[:, :, None], ns)

        def _inner(idx):
            prod = sla.cho_solve((self.chos[idx].T, True), k_star[idx].T).T

            ys = prod @ self.train_outputs[:, idx]

            v = np.identity(18) - prod @ k_star[idx].T
            return ys, v

        res = [_inner(i) for i in range(self.kernel.output_dims)]
        ys, vs = tuple(map(np.array, zip(*res)))
        ys = ys.sum(axis=0)
        vs = vs.mean(axis=0)

        # Turn y into the right form for D
        d = np.zeros(36)
        d[::2] = ys

        return d.reshape((6, 6)), vs.diagonal()
