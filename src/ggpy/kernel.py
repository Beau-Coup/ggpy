import numpy as np
import scipy.optimize as opt
import scipy.linalg as sla


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
        bounds = zip(0.1 * old_params, 10 * old_params)
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


class RBF(Kernel):
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
            full_size: bool, whether or not to return the whole covariance, or only the non-zeros

        Returns:
            k(x1, x2): (out_dims * n) x (out_dims * m) matrix of pairwise kernel computations, i.e. the covariance
            if full_size is false, then that shape is n x m (scalar kernel)
        """
        if x2 is None:
            x2 = x1.copy()
            if noise:
                add = np.eye(x1.shape[0]) * self.hyper_params[1]
            else:
                add = 0.0
        else:
            add = 0.0

        dx = np.expand_dims(x1, 1) - x2
        if len(dx.shape) < 3:
            dx = dx.reshape(dx.shape + (self.input_dims,))

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

        print(transformation.shape, prior_noise.shape)
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

        return outer_transf @ k @ ot2.T + add

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