"""Sanity tests for the Longstaff-Schwartz convertible bond pricer.

Run with:  pytest benchmarks/test_lsm_pricer.py -q
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from benchmarks.lsm_pricer import (
    ConvertibleBond,
    MarketData,
    _coupon_step_amounts,
    price_lsm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bond(
    *,
    face: float = 100.0,
    coupon: float = 0.05,
    freq: int = 2,
    years: float = 5.0,
    ratio: float = 1.0,
) -> ConvertibleBond:
    valuation = date(2025, 1, 1)
    days = int(round(years * 365.0))
    return ConvertibleBond(
        face_value=face,
        coupon_rate=coupon,
        coupon_frequency=freq,
        maturity_date=date.fromordinal(valuation.toordinal() + days),
        valuation_date=valuation,
        conversion_ratio=ratio,
    )


def _market(
    *,
    spot: float,
    vol: float = 0.25,
    r: float = 0.04,
    q: float = 0.02,
    cs_bps: float = 200.0,
) -> MarketData:
    return MarketData(
        spot_price=spot,
        volatility=vol,
        risk_free_rate=r,
        dividend_yield=q,
        credit_spread_bps=cs_bps,
    )


def _straight_bond_pv_per_100(bond: ConvertibleBond, market: MarketData) -> float:
    """Closed-form PV of the bond cashflows alone, discounted at r+s.

    Uses the same step grid + coupon-snap convention as the LSM pricer so the
    comparison is apples-to-apples.
    """
    T = (bond.maturity_date - bond.valuation_date).days / 365.0
    n_steps = max(1, int(round(T * 52)))  # match default steps_per_year
    dt = T / n_steps
    coupons = _coupon_step_amounts(bond, n_steps, dt, T)
    risky = market.risk_free_rate + market.credit_spread_bps / 10_000.0

    pv = 0.0
    for k in range(n_steps + 1):
        if coupons[k] != 0.0:
            pv += coupons[k] * math.exp(-risky * k * dt)
    pv += bond.face_value * math.exp(-risky * T)
    return pv * 100.0 / bond.face_value


# ---------------------------------------------------------------------------
# Zero-volatility sanity checks
# ---------------------------------------------------------------------------


def test_zero_vol_deep_otm_matches_straight_bond():
    """With sigma=0 and deep-OTM spot, LSM must equal the straight-bond PV."""
    bond = _bond(years=5.0, coupon=0.05, ratio=1.0)
    # Forward S_T = S_0 * exp((r-q)*T) — keep it well below F/kappa = 100.
    market = _market(spot=10.0, vol=0.0, r=0.04, q=0.02, cs_bps=200.0)

    res = price_lsm(bond, market, basis="hinge", n_paths=2_000, seed=1)
    expected = _straight_bond_pv_per_100(bond, market)

    # Zero vol => all paths identical => SE should be (numerically) zero.
    assert res.standard_error < 1e-6
    assert res.price == pytest.approx(expected, rel=1e-4)
    assert res.n_paths_itm == 0


def test_zero_vol_deep_itm_matches_immediate_conversion():
    """With sigma=0, deep-ITM spot, q > coupon yield: convert immediately."""
    bond = _bond(years=5.0, coupon=0.01, ratio=1.0)  # tiny coupon
    market = _market(spot=500.0, vol=0.0, r=0.03, q=0.05, cs_bps=200.0)
    # q > coupon => optimal to convert ASAP and capture dividends.

    res = price_lsm(bond, market, basis="hinge", n_paths=2_000, seed=1)
    # Conversion at the first available step (k=1) ≈ kappa * S_0 * exp((r-q)*dt) * exp(-r*dt)
    # = kappa * S_0 * exp(-q*dt) — very close to kappa * S_0 for small dt.
    expected = bond.conversion_ratio * market.spot_price * 100.0 / bond.face_value
    assert res.standard_error < 1e-6
    assert res.price == pytest.approx(expected, rel=2e-3)
    assert res.n_paths_itm == res.n_paths


# ---------------------------------------------------------------------------
# Asymptotic limits with positive vol
# ---------------------------------------------------------------------------


def test_deep_otm_approaches_straight_bond():
    """Deep-OTM spot with modest vol: price within ~3% of straight-bond floor."""
    bond = _bond(years=5.0, coupon=0.04, ratio=1.0)
    market = _market(spot=20.0, vol=0.20, r=0.04, q=0.02, cs_bps=300.0)

    res = price_lsm(bond, market, basis="hinge", n_paths=20_000, seed=42)
    floor = _straight_bond_pv_per_100(bond, market)

    # Convertible value >= straight bond value (option non-negative).
    assert res.price >= floor - 5 * res.standard_error
    # And, deep OTM, only a small premium above the floor.
    assert res.price <= floor * 1.05


def test_deep_itm_approaches_conversion_value():
    """Deep-ITM spot: price within a couple % of immediate conversion value."""
    bond = _bond(years=3.0, coupon=0.01, ratio=1.0)
    market = _market(spot=400.0, vol=0.25, r=0.03, q=0.05, cs_bps=200.0)

    res = price_lsm(bond, market, basis="hinge", n_paths=20_000, seed=7)
    conv_value = bond.conversion_ratio * market.spot_price * 100.0 / bond.face_value

    # Should be very close to conversion value; allow modest dividend-drag.
    assert res.price == pytest.approx(conv_value, rel=0.03)
    assert res.n_paths_itm == res.n_paths


# ---------------------------------------------------------------------------
# Basis consistency
# ---------------------------------------------------------------------------


def test_hinge_and_laguerre_agree_mid_range():
    """On a mid-range bond the two bases should price within a few SEs."""
    bond = _bond(years=5.0, coupon=0.04, ratio=1.0)
    market = _market(spot=80.0, vol=0.30, r=0.04, q=0.02, cs_bps=250.0)

    res_h = price_lsm(bond, market, basis="hinge",    n_paths=40_000, seed=11)
    res_l = price_lsm(bond, market, basis="laguerre", n_paths=40_000, seed=23)

    # Combined SE for an unpaired-difference test (independent seeds).
    combined_se = math.sqrt(res_h.standard_error**2 + res_l.standard_error**2)
    diff = abs(res_h.price - res_l.price)
    assert diff <= 3.0 * combined_se + 0.10, (
        f"basis disagreement: hinge={res_h.price:.4f} ± {res_h.standard_error:.4f}, "
        f"laguerre={res_l.price:.4f} ± {res_l.standard_error:.4f}, "
        f"diff={diff:.4f}, 3*combined_se={3*combined_se:.4f}"
    )


# ---------------------------------------------------------------------------
# Output-shape / interface smoke test
# ---------------------------------------------------------------------------


def test_result_shape_and_seed_reproducibility():
    bond = _bond(years=4.0, coupon=0.045, ratio=1.0)
    market = _market(spot=70.0, vol=0.30)

    a = price_lsm(bond, market, basis="hinge", n_paths=4_000, seed=99)
    b = price_lsm(bond, market, basis="hinge", n_paths=4_000, seed=99)
    c = price_lsm(bond, market, basis="hinge", n_paths=4_000, seed=100)

    # Identical seed => identical price (and SE).
    assert a.price == b.price
    assert a.standard_error == b.standard_error
    assert a.n_paths_itm == b.n_paths_itm
    # Different seed => different draw (almost surely different price).
    assert a.price != c.price
    # Antithetic doubling: odd n_paths rounds up to even.
    odd = price_lsm(bond, market, basis="laguerre", n_paths=1001, seed=1)
    assert odd.n_paths == 1002


def test_unknown_basis_raises():
    bond = _bond()
    market = _market(spot=80.0)
    with pytest.raises(ValueError):
        price_lsm(bond, market, basis="poly", n_paths=200, seed=0)  # type: ignore[arg-type]
