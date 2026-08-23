"""Generate ``tests/reference.npz``: golden outputs from the pre-JAX NumPy implementation.

Run with the *NumPy* implementation checked out to (re)generate the fixture:

    python tests/gen_reference.py

``tests/test_parity.py`` asserts the JAX port reproduces every array in here. The
inputs are stored alongside the outputs so both backends see byte-identical data.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ggpy import GP, RBF, DMatrix, MultiGP, MultiKernel, NaiveDMatrix, RevLinear, SmallDMatrix
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
    x1_6d = rng.standard_normal((8, 6))
    x2_6d = rng.standard_normal((5, 6))
    for n, v in [("x1_1d", x1_1d), ("x2_1d", x2_1d), ("x1_2d", x1_2d), ("x2_2d", x2_2d),
                 ("x1_4d", x1_4d), ("x2_4d", x2_4d), ("x1_6d", x1_6d), ("x2_6d", x2_6d)]:
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

    # ---- DMatrix (diagonal) --------------------------------------------
    dm = DMatrix(1e-3)
    rec("dmatrix_cross", dm.eval(x1_6d, x2_6d))
    rec("dmatrix_self", dm.eval(x1_6d))
    rec("dmatrix_grad", dm.gradient(x1_6d, x2_6d))

    # ---- NaiveDMatrix ---------------------------------------------------
    ndm = NaiveDMatrix(1e-3, scale=0.7)
    rec("naive_cross", ndm.eval(x1_6d, x2_6d))
    rec("naive_self", ndm.eval(x1_6d))
    rec("naive_grad", ndm.gradient(x1_6d, x2_6d))

    # ---- SmallDMatrix ---------------------------------------------------
    sdm = SmallDMatrix(1e-3, 2.0)
    rec("small_cross", sdm.eval(x1_6d, x2_6d))
    rec("small_self", sdm.eval(x1_6d))
    rec("small_grad", sdm.gradient(x1_6d, x2_6d))

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

    y_diag = rng.standard_normal((x1_6d.shape[0], 6))
    rec("in_y_diag", y_diag)
    rec("dmatrix_likelihood", dm.likelihood(x1_6d, y_diag))

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

    # ---- GP.get_D for the non-diagonal D kernels ---------------------------
    gp_naive = GP(NaiveDMatrix(1e-6), None)
    y_naive = rng.standard_normal((x1_6d.shape[0], 6))
    rec("in_y_naive", y_naive)
    gp_naive.add_points(x1_6d, y_naive, fit=True)
    d, dv = gp_naive.get_D()
    rec("gp_naive_D", d)
    rec("gp_naive_Dvar", dv)

    gp_small = GP(SmallDMatrix(1e-6, 1.0), None)
    gp_small.add_points(x1_6d, y_naive, fit=True)
    d, dv = gp_small.get_D()
    rec("gp_small_D", d)
    rec("gp_small_Dvar", dv)

    # ---- MultiGP -------------------------------------------------------------
    mgp = MultiGP(DMatrix(1e-6), None)
    mgp.add_points(x1_6d, y_diag, fit=True)
    rec("mgp_chos", np.asarray(mgp.chos))
    m_m, v_m = mgp.predict(x2_6d, var=True)
    rec("mgp_mean", m_m)
    rec("mgp_var", v_m)
    d, dv = mgp.get_D()
    rec("mgp_D", d)
    rec("mgp_Dvar", dv)

    mgp_inc = MultiGP(DMatrix(1e-6), None)
    mgp_inc.add_points(x1_6d[:-3], y_diag[:-3], fit=True)
    mgp_inc.add_points(x1_6d[-3:], y_diag[-3:], fit=True)
    rec("mgp_chos_incremental", np.asarray(mgp_inc.chos))

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
