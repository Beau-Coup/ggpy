import numpy as np
import scipy.linalg as sla
import scipy.optimize as opt
from numpy.typing import NDArray

from . import fourier


class Kernel:
    input_dims: int
    hyper_params: list

    def __init__(self, input_dims, output_dims, noise, input_noise=None):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.noise = noise
        self.input_noise = input_noise

    def eval(self, x1, x2=None):
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
            TODO: Figure this out
        """
        raise NotImplementedError

    def likelihood(self, x, y):
        """Return the likelihood of a set of input-output pairs,
        according to the posterior kernel.

        Parameters
        ----------
            x:
        """
        cov = self.eval(x)
        cov_l = sla.cho_factor(cov)
        sol = sla.cho_solve(cov_l, y)
        det = cov_l[0].diagonal().prod() ** 2

        ld = np.log(det)
        prod = y.T @ sol
        const = x.shape[0] * np.log(2 * np.pi)

        return (ld + prod + const) * 0.5

    def optimize(self, x, y):
        old_h = self.hyper_params.copy()
        # Optimize with the gradients stuff

        def like_inner(theta):
            self.hyper_params = theta
            return self.likelihood(x, y)

        old_params = np.array(old_h)
        bounds = zip(0.01 * old_params, 100 * old_params)
        res = opt.minimize(like_inner, old_h, bounds=(bounds))
        self.hyper_params = res.x
        # print(self.hyper_params)

        return res.x

    def augment(self, A, noise):
        """Return a new kernel which is the augment version of this kernel through A.

        Parameters
        ----------
            A: n x output_dims matrix by which to transform the output.

        Returns
        -------
            A kernel corresponding to the GP that results from transforming
            this GP by the linear transformation A.
        """

        # Utilize the fact that the kernel is already defined, and basically tile it in a bigger matrix.
        # Do optimizations, (which are definitely necessary) later.
        return LinearAugment(self, A, noise)


class Stationary(Kernel):
    """Kernels that depend only on x1 - x2 implement this.

    This allows for drawing random samples more efficiently.
    See the Wiener-Khinchin theorem.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def power_spectrum(self, k: float | NDArray) -> float | NDArray:
        raise NotImplementedError

    def sample_prior(
        self, n_points: int, length: float, seed: int | None = None
    ) -> NDArray:
        """Generate a random sample from the prior.

        I am not sure this method actually produces a sample with the right properties, but it looks good enough.

        Parameters
        ----------
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

        center = (n_points - 1) / 2.0
        x = np.arange(n_points)
        y = np.arange(n_points)

        x, y = np.meshgrid(x, y)
        k = (
            np.hypot(x - center, y - center)
            * (n_points - 1)
            / length
            / n_points
            / np.sqrt(2)
        )

        grid = self.power_spectrum(k) * length * np.sqrt(2)
        half = int(n_points / 2)

        aj = np.zeros((n_points, n_points), dtype=np.complex128)
        # We are generating the right half of the noise
        if seed is not None:
            np.random.seed(seed)

        aj[:, half:] = np.random.randn(n_points, half) + (
            np.random.randn(n_points, half) * 1j
        )
        # We want [-x, -y] = conj(x, y)
        aj[:, :half] = np.conj(aj[-1::-1, -1 : half - 1 : -1])

        aj *= grid / 2

        out = fourier.ifft2_v(aj) / length / length  # Unnormalized

        assert np.allclose(
            np.zeros((n_points, n_points)), np.imag(out), atol=1e-10, rtol=1e-08
        )
        out = np.real(out)
        return out


class RBF(Stationary):
    """Do I want to optimize over the hyperparameters?
    Why not try?
    """

    def __init__(self, input_dims, noise, length_scale, input_noise=None):
        super().__init__(input_dims, 1, noise, input_noise=input_noise)
        self.hyper_params = [length_scale, noise]

    def eval(self, x1, x2=None, noise=True):
        """Evaluate the kernel for x1 and x2

        Parameters:
            x1: (n x self.input_dims) matrix of input points
            x2: (m x self.input_dims) matrix of input points

        Returns:
            k(x1, x2): (out_dims * n) x (out_dims * m) matrix of pairwise kernel computations, i.e. the covariance
            if full_size is false, then that shape is n x m (scalar kernel)
        """
        if x2 is None:
            x2 = x1
            if noise:
                add = np.eye(x1.shape[0]) * self.hyper_params[1]
            else:
                add = 0.0
        else:
            add = 0.0

        dx = np.expand_dims(x1, 1) - x2
        arg = (dx**2).sum(axis=-1) / (self.hyper_params[0] ** 2) * 0.5
        return np.exp(-arg) + add

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
        dx = np.expand_dims(x1, 1) - x2
        if len(dx.shape) < 3:
            dx = dx.reshape(dx.shape + (self.input_dims,))
        dx = np.swapaxes(dx, 1, 2)  # n x input_dims x m
        dk = -dx / self.hyper_params[0] ** 2  # n x input_dims x m
        k = self.eval(x1, x2)  # n x m
        return dk * k[:, None, :]

    def length_scale(self) -> float:
        """Return the fit length scale of the kernel."""
        return self.hyper_params[0]

    def noise(self) -> float:
        """Return the fit noise value of the kernel."""
        return self.hyper_params[1]

    def power_spectrum(self, k: float | NDArray) -> float | NDArray:
        """Return the power spectrum of the kernel at a particular wave number.

        Given Wiener-Khinchin, this is the same as the Fourier Transform of the kernel.

        Parameters
        ----------
        k : float | NDArray
            The wave number(s) at which to evaluate the power spectrum.

        Ruturns
        -------
        float | NDArray
            The power of the given wave number(s).
        """
        return (
            np.sqrt(2 * np.pi)
            * self.hyper_params[0]
            * np.exp(-2 * ((np.pi * k * self.hyper_params[0]) ** 2))
        )


class LinearAugment:
    def __init__(self, prior, transformation, noise):
        assert (
            transformation.shape[1] == prior.output_dims
        ), f"Incorrect transformation, expected {( transformation.shape[0],prior.output_dims )}, got {transformation.shape}"
        self.prior = prior
        self.transformation = transformation
        self.input_dims = self.prior.input_dims
        self.output_dims = transformation.shape[0]
        prior_noise = np.array(self.prior.noise)

        prior_noise = prior_noise.reshape(
            (
                self.prior.output_dims,
                self.prior.output_dims,
            )
        )

        self.noise = noise + transformation @ prior_noise @ transformation.T

    def eval(self, x1, x2=None):
        """Evaluate the kernel for x1 and x2

        Parameters:
            x1: (n x self.input_dims) matrix of input points
            x2: (m x self.input_dims) matrix of input points

        Returns:
            k(x1, x2): (self.output_dims * n) x (self.output_dims * m) matrix of pairwise kernel computations, i.e. the covariance
        """
        if x2 is None:
            x2 = x1.copy()
            add = self.noise
            add = np.kron(add, np.eye(x1.shape[0]))
        else:
            add = 0.0

        k = self.prior.eval(x1, x2)
        outer_transf = np.kron(np.eye(x1.shape[0]), self.transformation)
        ot2 = np.kron(np.eye(x2.shape[0]), self.transformation)

        return outer_transf @ k @ ot2.T + add + 1e-10

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
        return self.transformation @ self.prior.gradient(x1, x2)


class DiagKernel(Kernel):
    def __init__(self, input_dims, output_dims, noise, input_noise=None):
        super().__init__(input_dims, output_dims, noise, input_noise)
        self.noise = np.ones(output_dims) * noise

    def eval(self, x1, x2=None):
        """Evaluate the kernel for x1 and x2

        Parameters:
            x1: (n x self.input_dims) matrix of input points
            x2: (m x self.input_dims) matrix of input points

        Returns:
            k(x1, x2): (self.output_dims x n x m) matrix of pairwise kernel computations, i.e. the covariance
        """
        if x2 is None:
            x2 = x1
            add = np.eye(x1.shape[0])[None, :, :] * self.noise[:, None, None]
        else:
            add = 0.0

        covs = np.array([self._eval_single(x1, x2, i) for i in range(self.output_dims)])
        return covs + add

    def _eval_single(self, x1, x2, idx):
        """Evaluate the kernel for a single output component."""
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
        grads = np.array(
            [self._gradient_single(x1, x2, i) for i in range(self.output_dims)]
        )
        return grads

    def _gradient_single(self, x1, x2, idx):
        """Evaluate the gradient of the covariance for a single output component."""
        raise NotImplementedError

    def likelihood(self, x, y):
        """Return the likelihood of a set of input-output pairs,
        according to the posterior kernel.

        Parameters
        ----------
            x: (n x input_dims)
            y: (n x output_dims)
        """
        covs = self.eval(x)
        cov_ls = [sla.cho_factor(covs[i]) for i in range(self.output_dims)]
        sols = np.stack(
            [sla.cho_solve(cov_ls[i], y[:, i]) for i in range(self.output_dims)], axis=0
        )
        dets = np.array(
            [cov_ls[i][0].diagonal().prod() ** 2 for i in range(self.output_dims)]
        )

        lds = np.log(dets)
        prod = np.sum([y[:, i].T @ sols[i] for i in range(self.output_dims)])
        const = x.shape[0] * np.log(2 * np.pi)

        return (np.sum(lds) + prod + const) * 0.5

    def optimize(self, x, y):
        old_h = self.hyper_params.copy()
        # Optimize with the gradients stuff

        def like_inner(theta):
            self.hyper_params = theta
            return self.likelihood(x, y)

        old_params = np.array(old_h)
        bounds = [(0.01 * p, 100 * p) for p in old_params]
        res = opt.minimize(like_inner, old_h, bounds=bounds)
        self.hyper_params = res.x

        return res.x


class DMatrix(DiagKernel):
    def __init__(self, noise):
        super().__init__(input_dims=6, output_dims=6, noise=noise)
        self.hyper_params = [noise] * 6

    def eval(self, x1, x2=None):
        """Evaluate the kernel for x1 and x2
        In this case, we specialize the evaluation for the DMatrix kernel.

        Parameters:
            x1: (n x self.input_dims) matrix of input points
            x2: (m x self.input_dims) matrix of input points

        Returns:
            k(x1, x2): (output_dims x n x m) matrix of pairwise kernel computations, i.e. the covariance
        """
        return super().eval(x1, x2)

    def _eval_single(self, x1, x2, idx):
        """Evaluate the kernel for a single output component.

        Parameters:
            x1: (n x self.input_dims) matrix of input points
            x2: (m x self.input_dims) matrix of input points
            idx: Index of the output component (0-5)

        Returns:
            (n x m) matrix of covariances for the specified output component
        """
        p = x1[:, None, :] * x2  # Shape (n, m, 6)

        if idx % 2 == 0:
            return p[:, :, 0::2].sum(axis=-1)
        else:
            return p[:, :, 1::2].sum(axis=-1)

    def gradient(self, x1, x2):
        """Return the gradient of the covariance of (X_1, X_2) with respect to X_1
        d/dx1 (k(x1, x2))

        Parameters
        ----------
            x1: n x input_dims matrix of input points at which to evaluate the gradient
            x2: m x input_dims matrix of points at which to evaluate the covariance

        Returns
        -------
            A output_dims x n x m x input_dims tensor of evaluated gradients.
        """
        n = x1.shape[0]
        m = x2.shape[0]

        gradient = np.zeros((self.output_dims, n, m, self.input_dims))

        # Select the right entries that are non-zero
        output_mask = np.zeros((self.output_dims, 1, 1, self.input_dims), dtype=bool)
        output_mask[0::2, :, :, 0::2] = True
        output_mask[1::2, :, :, 1::2] = True

        # Reshape x2 for broadcasting (1, m, 1, input_dims)
        x2_broadcast = x2.reshape(1, m, 1, self.input_dims)

        # Apply the mask to set the gradient values
        gradient = np.where(output_mask, x2_broadcast, gradient)

        return gradient
