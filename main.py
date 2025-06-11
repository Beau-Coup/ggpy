import matplotlib.pyplot as plt
import numpy as np

from ggpy import GP, RBF

if __name__ == "__main__":
    # Make a kernel with hetero noise and fit lol
    noise = 0.05
    kern = RBF(1, noise, 2.0, 1.0)
    model = GP(kern, None)

    n_points = 300
    xs = np.linspace(0, 10, n_points)
    ys = model.sample(xs[:, None])

    # Now fit on a random sample
    n_train = 6
    train_idx = np.random.choice(n_points, n_train)
    x_train = xs[train_idx]
    y_train = ys[train_idx] + noise * np.random.randn(n_train)[:, None]

    model.add_points(x_train, y_train, fit=True)

    # Predict posterior mean and variance
    y_pred, y_var = model.predict(xs[:, None], var=True)
    y_pred = y_pred.squeeze(1)
    ys = ys.squeeze(1)
    y_std = np.sqrt(y_var.diagonal())

    # Plot the true data, posterior fit, and uncertainty
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, color="C0")
    plt.plot(xs, y_pred, color="C0", linestyle="--")
    plt.fill_between(xs, y_pred - y_std, y_pred + y_std, color="C0", alpha=0.2)
    plt.errorbar(x_train, y_train.squeeze(), yerr=np.sqrt(noise), fmt="o")

    # Remove grid lines and show only curves
    plt.gca().grid(False)
    plt.axis("off")  # Remove all axis lines, ticks, and labels
    plt.subplots_adjust(
        left=0, right=1, top=1, bottom=0
    )  # Remove all padding around the plot
    plt.savefig("ggpy.jpg", dpi=300)
    plt.show()
