"""Shared test configuration.

Several tests call ``plt.show()`` for visual inspection. Force a non-interactive
backend so the suite runs headless instead of blocking on a window.
"""

import matplotlib

matplotlib.use("Agg")
