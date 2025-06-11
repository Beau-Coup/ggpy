import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ggpy import GP, RBF, MultiKernel


def test_linear_transform():
    np.random.seed(43)
    transform = np.concatenate([np.eye(2), np.eye(2)]).T

    noise1 = 1e-2
    noise2 = 1e-3
    rbf1 = RBF(1, noise1, 2.0)
    rbf2 = RBF(1, noise2, 1.0)

    k1 = MultiKernel(rbf1, rbf1)
    k2 = MultiKernel(rbf2, rbf2)

    kernel = MultiKernel(k1, k2)
    original_model = GP(kernel, None)
    transformed = kernel.transform(transform, 0.01)
    model = GP(transformed, None)

    num_samples = 100
    # x_test = np.random.randn(num_samples)
    x_test = np.linspace(-1, 1, num_samples)
    x_test = np.stack(
        [
            x_test,
        ]
        * 4
    ).T

    s_o = original_model.sample(x_test)
    original_model.add_points(x_test, s_o, fit=True)
    s = (transform @ s_o.T).T
    # print(f"Samples {s_o} at {x_test}")
    # print(f"Added: {s}")

    model.add_points(x_test, s, fit=True)

    xt = np.linspace(-1, 1, 2 * num_samples)
    xt = np.stack([xt] * 4, axis=1)

    reco, _ = model.predict(xt, var=True)  # Should be identical to s
    sor, _ = model.predict_original(xt, var=True)  # Hopefully close to s_o

    orig, _ = original_model.predict(xt, var=True)  # Should be almost identical to s_o

    f, a = plt.subplots(2)

    for i in range(4):
        col = "C" + str(i)
        # Plot original components
        a[0].plot(xt, orig[:, i], color=col, alpha=0.6)
        # Plot recovered components
        a[0].plot(xt, sor[:, i], color=col, linestyle="--", alpha=0.6)

    # a[0].scatter(x_test[:, :], s_o[:, :], color="C5")

    for i in range(2):
        col = "C" + str(i)
        # Plot original components
        a[1].plot(xt, orig[:, i] + orig[:, i + 2], color=col, alpha=0.4)
        # Plot recovered sum
        a[1].plot(xt, reco[:, i], color=col, linestyle="--", alpha=0.4)

        # Plot sum of recoverd components
        a[1].plot(
            xt,
            sor[:, i] + sor[:, i + 2],
            color="C" + str(i + 2),
            linestyle=":",
            alpha=0.4,
        )

    a[0].set_title("Original 4 components and recovered 4")
    a[0].grid()
    a[1].set_title("Original sum components and recovered sum")
    a[1].grid()
    plt.show()

    # TODO: get the original prediction and the recovered and compare


def test_multi_kernel():
    """Test that the multi kernel works as expected."""

    np.random.seed(43)

    k1 = RBF(2, 0.1, 1.0)
    k2 = RBF(2, 0.1, 1.0)
    kernel = MultiKernel(k1, k2)

    model = GP(kernel, None)

    num_samples = 32
    # x_samples = np.random.randn(num_samples, 2)
    # x_samples = np.concatenate([x_samples, x_samples], axis=1)
    xs = np.linspace(-1, 1, num_samples)
    X, Y = np.meshgrid(xs, xs)

    x_samples = np.stack((X.flatten(), Y.flatten()))
    x_samples = np.concatenate([x_samples, x_samples]).T

    samp = model.sample(x_samples)

    s1 = samp[:, 0].reshape((num_samples, num_samples))
    s2 = samp[:, 1].reshape((num_samples, num_samples))
    # Plot it to make sure it works
    f, a = plt.subplots(1, 2)
    a[0].imshow(s1)
    a[1].imshow(s2)
    plt.show()
