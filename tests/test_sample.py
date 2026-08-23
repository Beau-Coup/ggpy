import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
from scipy import stats

from ggpy import GP, RBF


def test_sampling():
    """The spectral prior sampler and the posterior sampler must agree in distribution.

    Whitens a ``sample_prior`` draw and a ``GP.sample`` draw by the same posterior
    covariance and checks the two whitened sets are indistinguishable under a
    two-sample KS test.
    """
    L_SCALE = 1
    N = 32
    L = 20.0
    keys = jax.random.split(jax.random.key(0), 2)

    rbf_kern = RBF(2, 0.0, L_SCALE)
    s1 = rbf_kern.sample_prior(keys[0], N, L)

    model = GP(rbf_kern)
    xs = jnp.linspace(-L / 2, L / 2, N)
    x, y = jnp.meshgrid(xs, xs)
    xs = jnp.stack((x, y), axis=-1).reshape((-1, 2))

    # Back out the random vector from s1
    mu, cov = model.predict(xs, True)
    cov = np.asarray(cov)
    cinv = np.linalg.pinv(cov + np.identity(cov.shape[0]) * 1e-10)
    # scipy: jax.scipy.linalg has no sqrtm
    cf = np.real(sla.sqrtm(cinv))
    noise = cf @ (np.asarray(s1).reshape((-1)) - np.asarray(mu).flatten())

    # Now, the noise should be standard normal, which we check statistically.
    s2 = model.sample(keys[1], xs)
    n2 = cf @ np.asarray(s2)
    pv = stats.ks_2samp(noise.ravel(), n2.ravel()).pvalue
    print(f"Got std {np.std(noise)} compared to {np.std(n2)}")
    print(f"Got mean {noise.mean()} compared to {n2.mean()}")

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    s2im = np.asarray(s2).reshape((N, N))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Set the shared color scale limits
    vmin = min(s1.min(), s2.min())
    vmax = max(s1.max(), s2.max())

    # Display the first s
    im1 = axes[0].imshow(s1, vmin=vmin, vmax=vmax)
    axes[0].set_title("s1")

    # Display the second s
    im2 = axes[1].imshow(s2im, vmin=vmin, vmax=vmax)
    axes[1].set_title("s2")

    # Create a shared colorbar
    divider = make_axes_locatable(axes[1])  # Use the second axis for placement
    cax = divider.append_axes("right", size="5%", pad=0.1)

    fig.colorbar(im1, cax=cax)

    plt.tight_layout()
    plt.show()

    assert pv > 0.05
