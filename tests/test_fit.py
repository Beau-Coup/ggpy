import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ggpy import GP, RevLinear


def test_linear_kernel():
    """Fitting a linear GP to y = a*x must recover the coefficient a.

    Checks both the least-squares estimate taken from the posterior and the
    closed-form estimate from ``GP.get_line``.
    """
    keys = jax.random.split(jax.random.key(43), 2)

    # Create the GP to sample from
    noise = 1e-2
    kernel = RevLinear(noise)
    num_samples = 100
    x_samples = jax.random.normal(keys[0], (num_samples, 1))

    model = GP(kernel)
    samp = model.sample(keys[1], x_samples)

    plt.scatter(x_samples, samp)
    plt.show()

    # Define a random true coefficient
    true_a = 1.3

    y_samples = true_a * x_samples

    # Fit the GP
    model = model.add_points(x_samples, y_samples, fit=True)

    xs = jnp.expand_dims(jnp.linspace(-100, 100, 2000), 1)
    y_pred, vs = model.predict(xs, var=True)
    vs = jnp.diagonal(vs)

    plt.plot(xs, y_pred)
    plt.fill_between(
        xs.squeeze(), y_pred.squeeze() - vs, y_pred.squeeze() + vs, alpha=0.1
    )
    plt.scatter(x_samples, y_samples)
    plt.show()

    # Now try to recover the coefficient using least squares
    recovered_a = jnp.linalg.lstsq(xs, y_pred, rcond=None)[0].T

    print(f"True a: {true_a} vs {recovered_a}")

    imputed_a = model.get_line()
    print(f"True a: {true_a} vs imputed {imputed_a}")

    # Both estimates are shrunk toward zero by the 1e-2 observation noise over 100
    # samples, which costs about 1e-4 of accuracy -- hence atol rather than the
    # default rtol=1e-5.
    assert np.allclose(true_a, imputed_a, atol=1e-3), (
        f"Imputed coefficient {imputed_a} is not equal to the data-generating coefficient {true_a}"
    )

    assert np.allclose(true_a, recovered_a, atol=1e-3), (
        f"Recovered coefficient {recovered_a} is not equal to the data-generating coefficient {true_a}"
    )

    # A prior sample is linear in x too, so least squares on it recovers a slope.
    generated_a = jnp.linalg.lstsq(x_samples, samp, rcond=None)[0].T
    assert np.all(np.isfinite(generated_a)), (
        f"Slope recovered from the prior sample is not finite: {generated_a}"
    )
