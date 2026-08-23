import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ggpy import GP, RBF, MultiKernel


def test_linear_transform():
    """A GP through a linear transform should recover the untransformed components."""
    keys = jax.random.split(jax.random.key(43), 2)
    transform = jnp.concatenate([jnp.eye(2), jnp.eye(2)]).T

    noise1 = 1e-2
    noise2 = 1e-3
    rbf1 = RBF(1, noise1, 2.0)
    rbf2 = RBF(1, noise2, 1.0)

    k1 = MultiKernel(rbf1, rbf1)
    k2 = MultiKernel(rbf2, rbf2)

    kernel = MultiKernel(k1, k2)
    original_model = GP(kernel)
    transformed = kernel.transform(transform, 0.01)
    model = GP(transformed)

    num_samples = 100
    x_test = jnp.linspace(-1, 1, num_samples)
    x_test = jnp.stack([x_test] * 4).T

    s_o = original_model.sample(keys[0], x_test)
    original_model = original_model.add_points(x_test, s_o, fit=True)
    s = (transform @ s_o.T).T

    model = model.add_points(x_test, s, fit=True)

    xt = jnp.linspace(-1, 1, 2 * num_samples)
    xt = jnp.stack([xt] * 4, axis=1)

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

    for i in range(2):
        col = "C" + str(i)
        # Plot original components
        a[1].plot(xt, orig[:, i] + orig[:, i + 2], color=col, alpha=0.4)
        # Plot recovered sum
        a[1].plot(xt, reco[:, i], color=col, linestyle="--", alpha=0.4)

        # Plot sum of recovered components
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

    # The transformed model must track the data it was fitted on. It does not
    # interpolate it exactly: the transformed kernel carries an observation noise
    # variance of 0.021, so the posterior is shrunk by up to ~1 noise sd.
    reco_train, _ = model.predict(x_test, var=True)
    noise_sd = float(np.sqrt(np.max(np.diag(np.asarray(transformed.noise)))))
    assert np.allclose(reco_train, s, atol=noise_sd), (
        "Transformed posterior does not track its own training data"
    )

    # Recovering the components and re-applying the transform is an exact identity.
    sor_train, _ = model.predict_original(x_test, var=True)
    summed = (transform @ sor_train.T).T
    assert np.allclose(summed, reco_train, atol=1e-9), (
        "Recovered components do not sum to the transformed posterior"
    )


def test_multi_kernel():
    """Test that the multi kernel works as expected."""
    key = jax.random.key(43)

    k1 = RBF(2, 0.1, 1.0)
    k2 = RBF(2, 0.1, 1.0)
    kernel = MultiKernel(k1, k2)

    model = GP(kernel)

    num_samples = 32
    xs = jnp.linspace(-1, 1, num_samples)
    X, Y = jnp.meshgrid(xs, xs)

    x_samples = jnp.stack((X.flatten(), Y.flatten()))
    x_samples = jnp.concatenate([x_samples, x_samples]).T

    samp = model.sample(key, x_samples)

    assert samp.shape == (num_samples * num_samples, 2)
    assert np.all(np.isfinite(samp)), "Prior sample contains non-finite values"

    s1 = samp[:, 0].reshape((num_samples, num_samples))
    s2 = samp[:, 1].reshape((num_samples, num_samples))
    # Plot it to make sure it works
    f, a = plt.subplots(1, 2)
    a[0].imshow(s1)
    a[1].imshow(s2)
    plt.show()
