"""Minimal IDL-parity smoke tests for the Python SFQ package."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from sfq import sfq_disambig
from sfq.disambig import sfq_clean
from sfq.utils import median_2d, smooth_2d


def _synthetic_magnetogram(nx: int = 64, ny: int = 64, seed: int = 42):
    """Potential-like bipolar AR with ~50% random azimuth flips."""
    np.random.seed(seed)
    x = np.linspace(-100, 100, nx)
    y = np.linspace(-100, 100, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    r1 = np.sqrt((xx + 30) ** 2 + (yy - 20) ** 2)
    r2 = np.sqrt((xx - 30) ** 2 + (yy + 20) ** 2)
    bz = 500 * np.exp(-(r1**2) / 800) - 400 * np.exp(-(r2**2) / 800)
    bz = bz + 30 * np.random.randn(ny, nx)

    bz_pad = np.zeros((2 * ny, 2 * nx), dtype=float)
    bz_pad[:ny, :nx] = bz
    kx = 2 * np.pi * np.fft.fftfreq(2 * nx)
    ky = 2 * np.pi * np.fft.fftfreq(2 * ny)
    kx, ky = np.meshgrid(kx, ky)
    q = np.sqrt(kx**2 + ky**2)
    q_safe = np.where(q > 1e-10, q, 1.0)
    bz_hat = np.fft.fft2(bz_pad)
    bx_hat = -1j * kx / q_safe * bz_hat
    by_hat = -1j * ky / q_safe * bz_hat
    bx_hat[0, 0] = 0
    by_hat[0, 0] = 0
    bx_true = np.fft.ifft2(bx_hat).real[:ny, :nx]
    by_true = np.fft.ifft2(by_hat).real[:ny, :nx]
    bx_true = gaussian_filter(bx_true, sigma=2)
    by_true = gaussian_filter(by_true, sigma=2)

    flip = np.random.rand(ny, nx) > 0.5
    bx_ambig = bx_true.copy()
    by_ambig = by_true.copy()
    bx_ambig[flip] = -bx_ambig[flip]
    by_ambig[flip] = -by_ambig[flip]
    bx_ambig = bx_ambig + 20 * np.random.randn(ny, nx)
    by_ambig = by_ambig + 20 * np.random.randn(ny, nx)

    pos = np.array([-100.0, -100.0, 100.0, 100.0])
    rsun = 960.0
    return bx_ambig, by_ambig, bz, pos, rsun, bx_true, by_true


def test_package_imports():
    import sfq
    from sfq.field import pot_vmag
    from sfq.utils import sol_crd, u_grid

    assert callable(sfq.sfq_disambig)
    assert callable(pot_vmag)
    assert callable(sol_crd)
    assert callable(u_grid)


def test_sfq_disambig_recovers_transverse_sign():
    bx_ambig, by_ambig, bz, pos, rsun, bx_true, by_true = _synthetic_magnetogram()

    mask = (bx_true**2 + by_true**2) > 100
    assert np.any(mask)
    ambig_dot = bx_true * bx_ambig + by_true * by_ambig
    ambig_agree = float(np.mean(ambig_dot[mask] > 0))
    assert ambig_agree < 0.6

    bx_out, by_out = sfq_disambig(
        bx_ambig.copy(),
        by_ambig.copy(),
        bz,
        pos,
        rsun,
        silent=True,
    )

    err1 = np.sum((bx_out - bx_true) ** 2 + (by_out - by_true) ** 2)
    err2 = np.sum((bx_out + bx_true) ** 2 + (by_out + by_true) ** 2)
    if err2 < err1:
        bx_out = -bx_out
        by_out = -by_out

    recovered_dot = bx_true * bx_out + by_true * by_out
    sign_recovery = float(np.mean(recovered_dot[mask] > 0))
    assert sign_recovery > 0.90, f"SFQ transverse-sign recovery too low: {sign_recovery:.1%}"


def test_sfq_clean_filter_width_is_idl_size_not_radius():
    """IDL ``median/smooth(arr, s)`` uses neighborhood size ``s``, not ``2*s+1``."""
    arr = np.zeros((9, 9), dtype=float)
    arr[4, 4] = 100.0
    sm3 = smooth_2d(arr, 3)
    sm7 = smooth_2d(arr, 7)
    assert sm3[4, 4] == pytest.approx(100.0 / 9.0)
    assert sm7[4, 4] == pytest.approx(100.0 / 49.0)
    assert sm3[4, 4] != sm7[4, 4]
    assert median_2d(arr, 1)[4, 4] == arr[4, 4]


def test_sfq_disambig_rejects_unsupported_large_fov():
    bx = np.ones((1024, 1024), dtype=float)
    by = np.ones((1024, 1024), dtype=float)
    bz = np.ones((1024, 1024), dtype=float)
    pos = np.array([-500.0, -500.0, 500.0, 500.0])
    with pytest.raises(ValueError, match="large-FOV"):
        sfq_disambig(bx, by, bz, pos, rsun=960.0, silent=True)


def test_sfq_clean_does_not_mutate_caller_arrays():
    bx = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=float)
    by = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=float)
    bx0, by0 = bx.copy(), by.copy()
    sfq_clean(bx, by, silent=True)
    assert np.array_equal(bx, bx0)
    assert np.array_equal(by, by0)


def test_sfq_disambig_does_not_mutate_caller_arrays():
    bx_ambig, by_ambig, bz, pos, rsun, *_ = _synthetic_magnetogram(nx=32, ny=32)
    bx0 = bx_ambig.copy()
    by0 = by_ambig.copy()
    sfq_disambig(bx_ambig, by_ambig, bz, pos, rsun, silent=True)
    assert np.array_equal(bx_ambig, bx0)
    assert np.array_equal(by_ambig, by0)
