import numpy as np
import scipy.linalg as sla
from scipy import stats

from ggpy import GP, RBF


def test_sampling():
    L_SCALE = 0.1
    N = 32
    L = 2.0
    SEED = None

    rbf_kern = RBF(2, 0.0, L_SCALE)
    s1 = rbf_kern.sample_prior(N, L, SEED)

    model = GP(rbf_kern, None)
    xs = np.linspace(-L / 2, L / 2, N)
    x, y = np.meshgrid(xs, xs)
    xs = np.stack((x, y), axis=-1)
    xs = xs.reshape((-1, 2))

    # Back out the random vector from s1
    mu, cov = model.predict(xs, True)
    cho = sla.cholesky(cov + np.identity(cov.shape[0]) * 1e-10, lower=True)
    noise = np.linalg.pinv(cho) @ (s1.reshape((-1)) - mu.flatten())
    # print(np.std(noise), noise.mean())

    # Now, the noise should be standard normal, which we check statistically.
    pv = stats.kstest(noise, stats.norm.cdf).pvalue

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.axes_grid1 import make_axes_locatable

    # s2 = model.sample(xs).reshape((N, N))
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #
    # # Set the shared color scale limits
    # vmin = min(s1.min(), s2.min())
    # vmax = max(s1.max(), s2.max())
    #
    # # Display the first s
    # im1 = axes[0].imshow(s1, vmin=vmin, vmax=vmax)
    # axes[0].set_title("s1")
    #
    # # Display the second s
    # im2 = axes[1].imshow(s2, vmin=vmin, vmax=vmax)
    # axes[1].set_title("s2")
    #
    # # Create a shared colorbar
    # divider = make_axes_locatable(axes[1])  # Use the second axis for placement
    # cax = divider.append_axes("right", size="5%", pad=0.1)
    #
    # # Create a shared colorbar
    # fig.colorbar(im1, cax=cax)
    #
    # plt.tight_layout()
    # plt.show()
    assert pv > 0.05
