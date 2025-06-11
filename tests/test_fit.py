import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ggpy import GP, RevLinear


def test_linear_kernel():
    """Test that the DMatrix kernel correctly implements a matrix with alternating zeros.

    This test:
    1. Creates a DMatrix kernel
    2. Generates random input points
    3. Evaluates the kernel
    4. Uses least squares to recover the 6x6 matrix structure
    5. Checks that the matrix has zeros in the right places
    """
    np.random.seed(43)

    # Create the GP to sample from
    noise = 1e-2
    kernel = RevLinear(noise)
    num_samples = 100
    x_samples = np.random.randn(num_samples, 1)

    model = GP(kernel, None)
    samp = model.sample(x_samples)

    plt.scatter(x_samples, samp)
    plt.show()

    # Define a random true dmatrix
    true_a = 1.3

    y_samples = true_a * x_samples
    # add noise
    # y_noise = y_samples + np.random.randn(*y_samples.shape) * np.sqrt(noise)

    # Fit the GP
    model.add_points(x_samples, y_samples, fit=True)

    xs = np.expand_dims(np.linspace(-100, 100, 2000), 1)
    y_pred, vs = model.predict(xs, var=True)
    vs = vs.diagonal()
    print(vs)

    plt.plot(xs, y_pred)
    plt.fill_between(
        xs.squeeze(), y_pred.squeeze() - vs, y_pred.squeeze() + vs, alpha=0.1
    )
    plt.scatter(x_samples, y_samples)
    plt.show()

    # Now try to recover the matrix using least squares
    print(xs.shape, y_pred.shape)
    recovered_a = np.linalg.lstsq(xs, y_pred, rcond=None)[0].T

    print(f"True a: {true_a} vs {recovered_a}")

    imputed_a = model.get_line()
    print(f"True a: {true_a} vs imputed {imputed_a}")

    # check that the imputed coefficient is actually close to the real one
    assert np.allclose(
        true_a, imputed_a
    ), f"Imputed coefficient {imputed_a} is not equal to the data-generating coefficient {true_a}"

    # check that the recovered matrix is actually close to the real one
    assert np.allclose(
        true_a, recovered_a
    ), f"Recovered coefficient {recovered_a} is not equal to the data-generating coefficient {true_a}"

    # Now use the sample from the prior distribution and back out the matrix, checking that it has the right form
    generated_a = np.linalg.lstsq(x_samples, np.expand_dims(samp, 1), rcond=None)[0].T
