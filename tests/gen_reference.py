"""How ``tests/reference.npz`` was produced -- kept for provenance, NOT runnable.

The fixture holds golden outputs captured from the **pre-JAX NumPy/SciPy**
implementation, and this script is the code that captured them. It deliberately calls
the old API (mutating ``add_points``, ``GP(kernel, input_noise)``, ``kernel.likelihood``)
and therefore does not run against the current JAX tree -- that is the point. The
values are a frozen baseline; regenerating them from the JAX code would be circular.

``tests/test_parity.py`` asserts the JAX port still reproduces every array in the
fixture. The inputs are stored alongside the outputs so both implementations saw
byte-identical data.

The D-matrix kernels this originally also covered have since been removed from the
library, and their arrays were dropped from the fixture.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ggpy import GP, RBF, MultiKernel, RevLinear
from ggpy import fourier


def build():
    rng = np.random.default_rng(20240821)
    out = {}

    def rec(name, value):
        out[name] = np.asarray(value)

    # ---- shared inputs -------------------------------------------------
    x1_1d = rng.standard_normal((7, 1))
    x2_1d = rng.standard_normal((5, 1))
    x1_2d = rng.standard_normal((6, 2))
    x2_2d = rng.standard_normal((4, 2))
    x1_4d = rng.standard_normal((6, 4))
    x2_4d = rng.standard_normal((4, 4))
    for n, v in [("x1_1d", x1_1d), ("x2_1d", x2_1d), ("x1_2d", x1_2d), ("x2_2d", x2_2d),
                 ("x1_4d", x1_4d), ("x2_4d", x2_4d)]:
        rec("in_" + n, v)

    # ---- RBF -----------------------------------------------------------
    rbf = RBF(2, 0.1, 2.0, 1.5)
    rec("rbf_cross", rbf.eval(x1_2d, x2_2d))
    rec("rbf_self", rbf.eval(x1_2d))
    rec("rbf_self_nonoise", rbf.eval(x1_2d, noise=False))
    rec("rbf_grad", rbf.gradient(x1_2d, x2_2d))
    rec("rbf_power", rbf.power_spectrum(np.linspace(0.0, 3.0, 11)))

    # ---- RevLinear -----------------------------------------------------
    rev = RevLinear(0.01)
    rec("rev_cross", rev.eval(x1_1d, x2_1d))
    rec("rev_self", rev.eval(x1_1d))
    rec("rev_grad", rev.gradient(x1_1d, x2_1d))

    # ---- MultiKernel + LinearTransformation ------------------------------
    k1 = RBF(2, 1e-2, 2.0, 1.0)
    k2 = RBF(2, 1e-3, 1.0, 0.8)
    mk = MultiKernel(k1, k2)
    rec("multi_cross", mk.eval(x1_4d, x2_4d))
    rec("multi_self", mk.eval(x1_4d))
    rec("multi_noise", mk.noise)

    transform = np.array([[1.0, 0.5], [0.25, -1.0], [0.0, 2.0]])
    lt = mk.transform(transform, 0.01)
    rec("lt_noise", lt.noise)
    rec("lt_cross", lt.eval(x1_4d, x2_4d))
    rec("lt_self", lt.eval(x1_4d))
    rec("lt_semi_left", lt.semi_left_eval(x1_4d, x2_4d))

    # ---- likelihood ------------------------------------------------------
    y_rbf = rng.standard_normal((x1_2d.shape[0], 1))
    rec("in_y_rbf", y_rbf)
    rec("rbf_likelihood", rbf.likelihood(x1_2d, y_rbf))

    # ---- GP fit / predict -------------------------------------------------
    gp = GP(RBF(2, 0.1, 2.0, 1.5), None)
    y_train = rng.standard_normal((x1_2d.shape[0], 1))
    rec("in_y_train", y_train)
    gp.add_points(x1_2d, y_train, fit=True)
    rec("gp_cho", gp.cho)
    mean, var = gp.predict(x2_2d, var=True)
    rec("gp_mean", mean)
    rec("gp_var", var)

    # incremental add_points reproduces a full refit
    gp_inc = GP(RBF(2, 0.1, 2.0, 1.5), None)
    gp_inc.add_points(x1_2d[:-2], y_train[:-2], fit=True)
    gp_inc.add_points(x1_2d[-2:], y_train[-2:], fit=True)
    rec("gp_cho_incremental", gp_inc.cho)

    # unfitted GP
    gp_empty = GP(RBF(2, 0.1, 2.0, 1.5), None)
    m0, v0 = gp_empty.predict(x2_2d, var=True)
    rec("gp_empty_mean", m0)
    rec("gp_empty_var", v0)

    # ---- GP with LinearTransformation -------------------------------------
    gp_lt = GP(lt, None)
    y_lt = rng.standard_normal((x1_4d.shape[0], 3))
    rec("in_y_lt", y_lt)
    gp_lt.add_points(x1_4d, y_lt, fit=True)
    rec("gp_lt_cho", gp_lt.cho)
    m_lt, v_lt = gp_lt.predict(x2_4d, var=True)
    rec("gp_lt_mean", m_lt)
    rec("gp_lt_var", v_lt)
    m_o, v_o = gp_lt.predict_original(x2_4d, var=True)
    rec("gp_lt_orig_mean", m_o)
    rec("gp_lt_orig_var", v_o)

    # ---- GP.get_line -------------------------------------------------------
    gp_rev = GP(RevLinear(1e-2), None)
    y_rev = 1.3 * x1_1d
    gp_rev.add_points(x1_1d, y_rev, fit=True)
    rec("gp_rev_cho", gp_rev.cho)
    rec("gp_rev_line", gp_rev.get_line())

    # ---- fourier -----------------------------------------------------------
    sig = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    sig2 = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    rec("in_fourier_sig", sig)
    rec("in_fourier_sig2", sig2)
    shift = np.array([(16 - 1) / 2, (16 - 1) / 2])
    rec("fourier_fft_v", fourier.fft_v(sig))
    rec("fourier_ifft_v", fourier.ifft_v(sig))
    rec("fourier_fft2_v", fourier.fft2_v(sig2))
    rec("fourier_ifft2_v", fourier.ifft2_v(sig2))
    rec("fourier_ft", fourier.ft(sig, 31.5, 31.5))
    rec("fourier_ift", fourier.ift(sig, 31.5, 31.5))
    rec("fourier_fft", fourier.fft(sig, 31.5, 31.5))
    rec("fourier_ifft", fourier.ifft(sig, 31.5, 31.5))
    rec("fourier_ft2", fourier.ft2(sig2, shift, shift))
    rec("fourier_ift2", fourier.ift2(sig2, shift, shift))
    rec("fourier_fft2", fourier.fft2(sig2, shift, shift))
    rec("fourier_ifft2", fourier.ifft2(sig2, shift, shift))

    return out


if __name__ == "__main__":
    data = build()
    dest = Path(__file__).parent / "reference.npz"
    np.savez_compressed(dest, **data)
    print(f"wrote {dest} with {len(data)} arrays")
