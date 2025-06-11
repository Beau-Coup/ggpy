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

    def transform(self, A, noise):
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
        return LinearTransformation(self, A, noise)

    def __repr__(self) -> str:
        """Return a string representation of the kernel.

        Returns a concise representation including class name and key properties.
        """
        class_name = self.__class__.__name__
        props = [
            f"input_dims={self.input_dims}",
            f"output_dims={self.output_dims}",
            f"noise={self.noise!r}",
        ]

        if hasattr(self, "transformation"):
            props.append(f"transformation={self.transformation!r}")

        # Include input_noise only if it's set
        if hasattr(self, "input_noise") and self.input_noise is not None:
            props.append(f"input_noise={self.input_noise!r}")

        # Include hyperparameters if available
        if hasattr(self, "hyper_params"):
            props.append(f"hyper_params={self.hyper_params!r}")

        return f"{class_name}({', '.join(props)})"


class MultiKernel(Kernel):
    """Class to associate mulitple uncorrelated kernels into a single kernel."""

    def __init__(self, *kernels):
        self.kernels = list(kernels)

        self.output_dims = 0
        self.input_dims = 0
        for kernel in self.kernels:
            self.output_dims += kernel.output_dims
            self.input_dims += kernel.input_dims

        self.noise = sla.block_diag(*[k.noise for k in self.kernels])

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

    def eval(self, x1, x2=None):
        """Put them all together brother.
        Parameters:
        ----------
            x1: tuple of inputs to each kernel
            x2: same as above

        Returns:
        -------
            evaluated kernel in block diagonal matrix
        """
        if x2 is None:
            x2inp = (None,) * len(self.kernels)
        else:
            x2inp = self.split_inputs(x2)

        x1inp = self.split_inputs(x1)
        # reshape the kernels to have the right form , so k(x1, x1), k(x1, x2)...
        # right now, the kernels each have the form k1(x1, x1), k1(x1, x2)
        # k(xi, xj) = diag(k1(xi, xj), k2(xi, xj), ...)
        m, n = x1.shape[0], x2.shape[0] if x2 is not None else x1.shape[0]
        k = np.zeros((m * self.output_dims, n * self.output_dims))

        iinds = np.arange(m)
        jinds = np.arange(n)

        offset = 0

        rows = iinds * self.output_dims
        cols = jinds * self.output_dims

        for k_idx, kernel in enumerate(self.kernels):
            kvals = kernel.eval(x1inp[k_idx], x2inp[k_idx])
            out = kernel.output_dims

            # Get the kernel into d x d blocks
            blocks = np.stack(np.split(kvals, m))
            blocks = np.stack(np.split(blocks, n, axis=-1), axis=1)

            # block_idx = np.ix_(np.arange(out), np.arange(out))

            # expand the indices
            # start = k_idx * out + offset
            ksize = np.arange(out)
            row_idx = rows[..., None] + ksize + offset  # m x d
            col_idx = cols[..., None] + ksize + offset  # n x d
            # Want to make an m x n x d x d index for the blocks to fit into
            # print(f"Shapes {row_idx.shape}, {col_idx.shape}, {blocks.shape}")

            k[row_idx[:, None, :, None], col_idx[None, :, None, :]] = blocks
            offset += kernel.output_dims

        return k

    def _inner_eval(self, xi, xj=None):
        """Inner evaluation of the kernel.
        Evaluate only a single pair of inputs.
        """
        return np.diag([k.eval(xi, xj) for k in self.kernels])

    def gradient(self, x1, x2):
        """Same as above"""
        x1inp = self.split_inputs(x1)
        x2inp = self.split_inputs(x2)

        return sla.block_diag(
            *[k.gradient(x1inp[i], x2inp[i]) for (i, k) in enumerate(self.kernels)]
        )


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
        if seed is not None:
            np.random.seed(seed)

        # Handle 1D case
        if self.input_dims == 1:
            center = (n_points - 1) / 2.0
            x = np.arange(n_points)

            # Calculate wave numbers for 1D
            k = np.abs(x - center) * (n_points - 1) / length / n_points

            # Apply power spectrum
            grid = self.power_spectrum(k) * length
            half = int(n_points / 2)

            # Generate noise in frequency domain
            aj = np.zeros(n_points, dtype=np.complex128)
            aj[:half] = np.random.randn(half) + (np.random.randn(half) * 1j)
            # Ensure Hermitian symmetry for real output
            aj[half:] = np.conj(aj[half - 1 :: -1])

            # Scale by power spectrum
            aj *= grid / np.sqrt(2)  # * np.exp(1j / 2 / length)

            # Inverse FFT
            out = fourier.ifft_v(aj) * n_points / length

            assert np.allclose(np.zeros(n_points), np.imag(out), atol=1e-10, rtol=1e-08)
            return np.real(out)

        # Original 2D case
        else:
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


class WhiteNoise(Kernel):
    def __init__(self, input_dims, output_dims, noise, input_noise=None):
        super().__init__(input_dims, output_dims, noise, input_noise=input_noise)

    def eval(self, x1, x2=None):
        if x2 is None:
            x2 = x1
            add = np.eye(x1.shape[0] * self.output_dims) * self.noise
        else:
            add = np.zeros(
                (x1.shape[0] * self.output_dims, x2.shape[0] * self.output_dims)
            )

        return add

    def gradient(self, x1, x2):
        return np.zeros((x1.shape[0], self.input_dims, x2.shape[0]))


class RBF(Stationary):
    """Do I want to optimize over the hyperparameters?
    Why not try?
    """

    def __init__(self, input_dims, noise, length_scale, scale, input_noise=None):
        super().__init__(input_dims, 1, noise, input_noise=input_noise)
        self.hyper_params = [length_scale]
        self.noise = noise
        self.scale = scale

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
                add = np.eye(x1.shape[0]) * self.noise
            else:
                add = 0.0
        else:
            add = 0.0

        dx = np.expand_dims(x1, 1) - x2
        arg = (dx**2).sum(axis=-1) / (self.hyper_params[0] ** 2) * 0.5
        return self.scale * np.exp(-arg) + add

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

    def get_length_scale(self) -> float:
        """Return the fit length scale of the kernel."""
        return self.hyper_params[0]

    def get_noise(self) -> float:
        """Return the fit noise value of the kernel."""
        return self.noise

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


class LinearTransformation(Kernel):
    def __init__(self, kernel, transformation, noise):
        assert transformation.shape[1] == kernel.output_dims, (
            f"Incorrect transformation, expected {(transformation.shape[0], kernel.output_dims)}, got {transformation.shape}"
        )
        super().__init__(kernel.input_dims, transformation.shape[0], noise)
        self.prior = kernel
        self.transformation = transformation
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

        part = self.semi_left_eval(x1, x2)
        ot2 = np.kron(np.eye(x2.shape[0]), self.transformation.T)

        return part @ ot2 + add

    def semi_left_eval(self, x1, x2):
        """Evaluate the kernel for x1 and x2 but apply the transformation on the left.

        This is useful for intermediate calculations, and for recovering an estimate of the original
        GP posterior. It doesn't really make sense to have x2 = None here so this should not be used
        to compute the cholesky decomposition directly.

        Parameters:
        ----------
            x1: (n x self.input_dims) matrix of input points
            x2: (m x self.input_dims) matrix of input points

        Returns:
        --------
            k(x1, x2): (self.output_dims * n) x (self.output_dims * m) matrix of pairwise kernel computations, i.e. the covariance
        """
        k = self.prior.eval(x1, x2)
        outer_transf = np.kron(np.eye(x1.shape[0]), self.transformation)

        return outer_transf @ k

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


class NaiveDMatrix(Kernel):
    def __init__(self, noise, scale=1.0):
        super().__init__(input_dims=6, output_dims=6, noise=noise)
        self.scale = scale  # Essentially account for uninformative priors by making the weight smaller.
        self.noise = noise * np.eye(6)
        self.hyper_params = [noise] * 6

    def eval(self, x1, x2=None):
        """Evaluate the kernel for x1 and x2
        In this case, we specialize the evaluation for the DMatrix kernel.

        Parameters:
            x1: (n x self.input_dims) matrix of input points
            x2: (m x self.input_dims) matrix of input points

        Returns:
            k(x1, x2): (6 * n x 6 * m) matrix of pairwise kernel computations, i.e. the covariance
        """
        if x2 is None:
            x2 = x1
            add = np.kron(np.eye(x1.shape[0]), self.noise)
        else:
            add = 0.0

        p = x1[:, None, :] * x2
        alpha = p[:, :, ::2].sum(axis=-1)
        beta = p[:, :, 1::2].sum(axis=-1)

        res = np.zeros((6 * x1.shape[0], 6 * x2.shape[0]))
        res[::6, ::6] = alpha
        res[2::6, 2::6] = alpha
        res[4::6, 4::6] = alpha
        res[1::6, 1::6] = beta
        res[3::6, 3::6] = beta
        res[5::6, 5::6] = beta

        return res * self.scale + add

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


class SmallDMatrix(Kernel):
    def __init__(self, noise, prior_uncertainty=1.0):
        super().__init__(input_dims=6, output_dims=6, noise=noise)
        self.prior_unc = prior_uncertainty * np.eye(14)
        self.noise = noise * np.eye(6)
        self.hyper_params = [noise] * 6

    def eval(self, x1, x2=None):
        """Evaluate the kernel for x1 and x2
        In this case, we specialize the evaluation for the DMatrix kernel.

        Parameters:
            x1: (n x self.input_dims) matrix of input points
            x2: (m x self.input_dims) matrix of input points

        Returns:
            k(x1, x2): (6 * n x 6 * m) matrix of pairwise kernel computations, i.e. the covariance
        """
        if x2 is None:
            x2 = x1
            add = np.kron(np.eye(x1.shape[0]), self.noise)
        else:
            add = 0.0

        l = np.zeros((6 * x1.shape[0], 14))
        l[::6, 0] = x1[:, 0]
        l[1::6, 1:4] = x1[:, 1::2]
        l[2::6, 4:6] = x1[:, 2::2]
        l[3::6, 6:9] = x1[:, 1::2]
        l[4::6, 9:11] = x1[:, 2::2]
        l[5::6, 11:] = x1[:, 1::2]

        r = np.zeros((6 * x2.shape[0], 14))
        r[::6, 0] = x2[:, 0]
        r[1::6, 1:4] = x2[:, 1::2]
        r[2::6, 4:6] = x2[:, 2::2]
        r[3::6, 6:9] = x2[:, 1::2]
        r[4::6, 9:11] = x2[:, 2::2]
        r[5::6, 11:] = x2[:, 1::2]

        res = l @ self.prior_unc @ r.T

        return res + add

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
        output_mask[0, :, :, 0] = True
        output_mask[2::2, :, :, 2::2] = True
        output_mask[1::2, :, :, 1::2] = True

        # Reshape x2 for broadcasting (1, m, 1, input_dims)
        x2_broadcast = x2.reshape(1, m, 1, self.input_dims)

        # Apply the mask to set the gradient values
        gradient = np.where(output_mask, x2_broadcast, gradient)

        return gradient


class Sum(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel, input_noise=None):
        assert k1.input_dims == k2.input_dims
        assert k1.output_dims == k2.output_dims
        super().__init__(
            k1.input_dims, k1.output_dims, k1.noise + k2.noise, input_noise
        )
        self.k1 = k1
        self.k2 = k2

    def eval(self, x1, x2=None):
        return self.k1.eval(x1, x2) + self.k2.eval(x1, x2)

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
        raise self.k1.gradient(x1, x2) + self.k2.gradient(x1, x2)


class RevLinear(Kernel):
    def __init__(self, noise):
        super().__init__(1, 1, noise, None)

    def eval(self, x1, x2=None):
        """Evaluate the kernel for x1 and x2
        In this case, we specialize the evaluation for a reversed linear kernel.

        Parameters:
            x1: (n x self.input_dims) matrix of input points
            x2: (m x self.input_dims) matrix of input points

        Returns:
            k(x1, x2): (n x m) matrix of pairwise kernel computations, i.e. the covariance
        """
        if x2 is None:
            x2 = x1
            add = np.eye(x1.shape[0]) * self.noise
        else:
            add = 0.0
        prod = np.expand_dims(x1, 1) * x2
        return prod.squeeze() + add

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

        return x2[None, :] * np.ones(x1.shape[0])
