"""Certification of the Frank copula implementation.

Mirrors the thesis's footnote: densities are computed in log space and
"certified against arbitrary-precision reference values" — here via mpmath.
Run directly (``python tests/test_frank_copula.py``) or under pytest.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.frank_copula import (  # noqa: E402
    frank_logpdf, frank_logpdf_bivariate, log_eulerian, log_polylog_neg, fit_theta_mle,
)


def test_eulerian_numbers():
    # A(3,.) = (1, 4, 1); A(4,.) = (1, 11, 11, 1)
    assert np.allclose(np.exp(log_eulerian(3)), [1, 4, 1])
    assert np.allclose(np.exp(log_eulerian(4)), [1, 11, 11, 1])


def test_polylog_against_mpmath():
    import mpmath
    for n in (1, 2, 5, 12):
        for w in (0.05, 0.5, 0.95):
            ours = log_polylog_neg(n, np.log(np.array([w])))[0]
            ref = float(mpmath.log(mpmath.polylog(-n, w)))
            assert abs(ours - ref) < 1e-9, (n, w, ours, ref)


def test_polylog_at_large_order():
    """The SIC sector blocks of Section 3.7 reach ~100 bonds, so the density
    needs the 100-th derivative of the generator's pseudo-inverse. Check the
    orders actually used, not just the small ones."""
    import mpmath
    mpmath.mp.dps = 60
    for n in (20, 50, 103, 150):
        for w in (0.05, 0.5, 0.95, 0.999):
            ours = log_polylog_neg(n, np.log(np.array([w])))[0]
            ref = float(mpmath.log(mpmath.polylog(-n, mpmath.mpf(w))))
            assert abs(ours - ref) < 1e-9, (n, w, ours, ref)


def test_density_finite_in_high_dimension():
    rng = np.random.default_rng(0)
    u = rng.uniform(0.01, 0.99, size=(4, 100))
    for theta in (0.5, 3.0, 12.0):
        v = frank_logpdf(u, theta)
        assert np.all(np.isfinite(v)), (theta, v)


def test_general_matches_bivariate():
    rng = np.random.default_rng(0)
    U = rng.uniform(0.02, 0.98, size=(200, 2))
    for theta in (0.3, 2.0, 8.0):
        a = frank_logpdf(U, theta)
        b = frank_logpdf_bivariate(U[:, 0], U[:, 1], theta)
        assert np.allclose(a, b, atol=1e-10), theta


def test_density_integrates_to_one():
    # midpoint rule on a fine grid, d = 2 (both signs) and d = 3 (theta > 0)
    m = 400
    g = (np.arange(m) + 0.5) / m
    for theta in (-4.0, 4.0):
        u, v = np.meshgrid(g, g)
        val = np.exp(frank_logpdf_bivariate(u.ravel(), v.ravel(), theta)).mean()
        assert abs(val - 1.0) < 5e-3, (theta, val)
    m3 = 60
    g3 = (np.arange(m3) + 0.5) / m3
    U = np.stack(np.meshgrid(g3, g3, g3), axis=-1).reshape(-1, 3)
    val = np.exp(frank_logpdf(U, 3.0)).mean()
    assert abs(val - 1.0) < 2e-2, val


def test_density_against_mpmath_highdim():
    """Full d-dimensional density vs arbitrary-precision finite differences of C."""
    import mpmath
    mpmath.mp.dps = 40
    theta = 2.5
    d = 5
    u = [0.3, 0.55, 0.7, 0.2, 0.85]

    def C(*uu):
        th = mpmath.mpf(theta)
        s = sum(-mpmath.log((1 - mpmath.e**(-th * x)) / (1 - mpmath.e**(-th))) for x in uu)
        return -(1 / th) * mpmath.log(1 - (1 - mpmath.e**(-th)) * mpmath.e**(-s))

    def density(uu):
        return mpmath.diff(C, tuple(uu), (1,) * d)

    ours = frank_logpdf(np.array(u)[None, :], theta)[0]
    ref_val = float(mpmath.log(density(u)))
    assert abs(ours - ref_val) < 1e-6, (ours, ref_val)


def test_theta_mle_recovers_dependence_ordering():
    rng = np.random.default_rng(1)
    # strongly dependent months via a common factor -> large theta;
    # independent months -> theta near 0
    dep_months, ind_months = [], []
    for _ in range(12):
        z = rng.normal()
        dep = 0.9 * z + 0.45 * rng.normal(size=6)
        from scipy.stats import norm
        dep_months.append(norm.cdf(dep / np.sqrt(0.9**2 + 0.45**2)))
        ind_months.append(rng.uniform(0.01, 0.99, size=6))
    th_dep = fit_theta_mle(dep_months)
    th_ind = fit_theta_mle(ind_months)
    assert th_dep > 2.0, th_dep
    assert th_ind < 1.0, th_ind


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all frank_copula tests passed")
