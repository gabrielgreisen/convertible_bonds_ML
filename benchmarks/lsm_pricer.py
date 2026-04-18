"""Longstaff-Schwartz Monte Carlo pricer for vanilla convertible bonds.

Vanilla = American conversion only; no calls, puts, mandatory conversion,
contingent conversion, resets, or variable conversion ratios. Anything else
must be filtered upstream — this module does not check.

Single-instrument interface. A separate batch wrapper handles looping,
timing and aggregation across many bonds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Literal

import numpy as np


# ---------------------------------------------------------------------------
# Instrument / market dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvertibleBond:
    """Vanilla convertible bond contract terms."""

    face_value: float
    coupon_rate: float
    maturity_date: date
    valuation_date: date
    conversion_ratio: float
    coupon_frequency: int = 2  # payments per year (1 = annual, 2 = semi, 4 = quarterly)


@dataclass(frozen=True)
class MarketData:
    """Market state for the underlying and credit."""

    spot_price: float
    volatility: float
    risk_free_rate: float
    dividend_yield: float
    credit_spread_bps: float


@dataclass(frozen=True)
class LSMResult:
    """Outputs of an LSM pricing run."""

    price: float                # mean discounted cashflow, per 100 face
    standard_error: float       # sample SE of the price estimate, per 100 face
    n_paths_itm: int            # paths that hit m > 1 at any exercise step
    n_paths: int                # total simulated paths (post antithetic doubling)
    n_steps: int                # number of timesteps used


# ---------------------------------------------------------------------------
# Basis-function strategies
# ---------------------------------------------------------------------------

BasisFn = Callable[[np.ndarray], np.ndarray]


def _hinge_basis(m: np.ndarray) -> np.ndarray:
    """Hinge basis (5 terms): [1, m, m^2, max(m-1, 0), max(1-m, 0)]."""
    out = np.empty((m.size, 5), dtype=np.float64)
    out[:, 0] = 1.0
    out[:, 1] = m
    out[:, 2] = m * m
    out[:, 3] = np.maximum(m - 1.0, 0.0)
    out[:, 4] = np.maximum(1.0 - m, 0.0)
    return out


def _laguerre_basis(m: np.ndarray) -> np.ndarray:
    """Unweighted Laguerre basis through degree 3 (4 terms)."""
    out = np.empty((m.size, 4), dtype=np.float64)
    m2 = m * m
    m3 = m2 * m
    out[:, 0] = 1.0
    out[:, 1] = 1.0 - m
    out[:, 2] = 1.0 - 2.0 * m + 0.5 * m2
    out[:, 3] = 1.0 - 3.0 * m + 1.5 * m2 - m3 / 6.0
    return out


_BASES: dict[str, BasisFn] = {
    "hinge": _hinge_basis,
    "laguerre": _laguerre_basis,
}


# ---------------------------------------------------------------------------
# Coupon schedule helper
# ---------------------------------------------------------------------------


def _coupon_step_amounts(
    bond: ConvertibleBond,
    n_steps: int,
    dt: float,
    T: float,
) -> np.ndarray:
    """Lump coupons onto the discrete step grid.

    Coupon dates are generated backward from maturity at the contract
    frequency. Each coupon is snapped to its nearest timestep on the path
    grid. The maturity coupon always lands on the final step.
    """
    coupon_amount = bond.face_value * bond.coupon_rate / bond.coupon_frequency
    period = 1.0 / bond.coupon_frequency

    coupons = np.zeros(n_steps + 1, dtype=np.float64)
    # Generate coupon times working backward from maturity in years from valuation.
    t_c = T
    while t_c > 1e-12:
        k = int(round(t_c / dt))
        if 0 <= k <= n_steps:
            coupons[k] += coupon_amount
        t_c -= period
    return coupons


# ---------------------------------------------------------------------------
# Path generation
# ---------------------------------------------------------------------------


def _simulate_paths(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pre-allocated GBM path matrix using log-Euler with antithetic draws."""
    half = n_paths // 2
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float32)
    paths[:, 0] = np.float32(S0)

    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol_step = sigma * np.sqrt(dt)

    for k in range(n_steps):
        z_half = rng.standard_normal(half).astype(np.float32)
        z = np.concatenate([z_half, -z_half])
        log_step = np.float32(drift) + np.float32(vol_step) * z
        paths[:, k + 1] = paths[:, k] * np.exp(log_step)

    return paths


# ---------------------------------------------------------------------------
# Main pricer
# ---------------------------------------------------------------------------


def price_lsm(
    bond: ConvertibleBond,
    market: MarketData,
    *,
    basis: Literal["hinge", "laguerre"],
    n_paths: int = 50_000,
    steps_per_year: int = 52,
    seed: int | None = None,
) -> LSMResult:
    """Price a vanilla convertible bond by Longstaff-Schwartz Monte Carlo.

    Tsiveriotis-Fernandes credit split (LSM adaptation)
    --------------------------------------------------
    The TF model splits convertible value into an equity component (discount
    at the risk-free rate ``r``) and a debt component (discount at the risky
    rate ``r + s``, where ``s`` is the credit spread). In an LSM setting the
    optimal regime is determined per path by the regression-based exercise
    rule:

      * While a path holds the bond (continuation > conversion at every step
        seen so far), it sits in the "debt" regime: the carried value is
        rolled back at ``exp(-(r+s)*dt)`` and any coupon falling on the step
        is added to it.
      * As soon as a path's regression-implied continuation value falls below
        its conversion value, it switches to the "equity" regime: the carried
        value is replaced by ``kappa * S_k`` and rolled back at ``exp(-r*dt)``
        from that point onward, with no further coupons.

    The regime is therefore sticky once a path converts. Maturity payoff
    ``max(kappa*S_T, F + final_coupon)`` initializes the regime per path
    accordingly.

    Notes
    -----
    * Antithetic variates are mandatory; ``n_paths`` is rounded up to the
      nearest even number.
    * Standard error reported is the raw MC SE of the per-path discounted
      cashflow. The Longstaff-Schwartz regression itself introduces a
      (typically downward) finite-sample bias on the option value; that bias
      is a separate concern and is **not** quantified here.
    """
    if basis not in _BASES:
        raise ValueError(f"basis must be one of {list(_BASES)}, got {basis!r}")
    basis_fn = _BASES[basis]
    basis_dim = basis_fn(np.array([1.0])).shape[1]

    # ---- time grid ------------------------------------------------------
    T = (bond.maturity_date - bond.valuation_date).days / 365.0
    if T <= 0:
        raise ValueError("maturity_date must be after valuation_date")
    n_steps = max(1, int(round(T * steps_per_year)))
    dt = T / n_steps

    # ---- antithetic-friendly path count ---------------------------------
    if n_paths < 2:
        raise ValueError("n_paths must be >= 2 for antithetic sampling")
    if n_paths % 2 != 0:
        n_paths += 1

    # ---- market parameters ---------------------------------------------
    sigma = market.volatility
    r = market.risk_free_rate
    q = market.dividend_yield
    s = market.credit_spread_bps / 10_000.0
    F = bond.face_value
    kappa = bond.conversion_ratio

    disc_rf = float(np.exp(-r * dt))
    disc_risky = float(np.exp(-(r + s) * dt))

    # ---- coupon schedule on the grid -----------------------------------
    coupons_at_step = _coupon_step_amounts(bond, n_steps, dt, T)

    # ---- simulate paths ------------------------------------------------
    rng = np.random.default_rng(0 if seed is None else seed)
    paths = _simulate_paths(
        S0=market.spot_price, r=r, q=q, sigma=sigma,
        n_paths=n_paths, n_steps=n_steps, dt=dt, rng=rng,
    )

    # ---- backward induction --------------------------------------------
    # State at each backward step: per-path carried value V (float64 for
    # numerical stability of the regression), and a sticky boolean
    # indicating whether the path has switched to the equity regime.
    final_bond = F + coupons_at_step[-1]
    S_T = paths[:, -1].astype(np.float64)
    conv_T = kappa * S_T

    V = np.maximum(conv_T, final_bond).astype(np.float64)
    is_equity = conv_T > final_bond  # converted at maturity

    # Track per-path "ever ITM" using moneyness m = kappa*S/F = S/S* > 1
    itm_ever = conv_T > F

    for k in range(n_steps - 1, 0, -1):
        # 1. roll V back from k+1 to k at the path's regime rate
        V *= np.where(is_equity, disc_rf, disc_risky)

        # 2. coupon paid at step k accrues only to debt-regime paths
        c_k = coupons_at_step[k]
        if c_k != 0.0:
            V += np.where(is_equity, 0.0, c_k)

        # 3. exercise decision on every ITM path. The is_equity flag only
        # controls discounting / coupon accrual — a path whose maturity-side
        # optimum was conversion may still want to convert *earlier*, so we
        # do not exclude it from the regression.
        S_k = paths[:, k].astype(np.float64)
        conv_val = kappa * S_k
        m = conv_val / F  # moneyness; m > 1 ⇔ ITM for conversion
        itm_now = m > 1.0
        itm_ever |= itm_now

        n_cand = int(itm_now.sum())
        if n_cand > basis_dim:
            X = basis_fn(m[itm_now])
            y = V[itm_now]
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            cont_est = X @ coeffs

            exercise_local = conv_val[itm_now] > cont_est
            if exercise_local.any():
                cand_idx = np.flatnonzero(itm_now)
                ex_idx = cand_idx[exercise_local]
                V[ex_idx] = conv_val[ex_idx]
                is_equity[ex_idx] = True
        # If too few ITM paths to fit the basis, all of them trivially
        # continue this step — same outcome as a degenerate regression.

    # 4. final roll from step 1 to step 0 (no exercise opportunity here:
    # t=0 conversion is captured by the user comparing price vs kappa*S0).
    V *= np.where(is_equity, disc_rf, disc_risky)
    if coupons_at_step[0] != 0.0:
        V += np.where(is_equity, 0.0, coupons_at_step[0])

    # ---- aggregate to price-per-100 convention -------------------------
    scale = 100.0 / F
    V_scaled = V * scale
    price = float(V_scaled.mean())
    se = float(V_scaled.std(ddof=1) / np.sqrt(n_paths))

    return LSMResult(
        price=price,
        standard_error=se,
        n_paths_itm=int(itm_ever.sum()),
        n_paths=n_paths,
        n_steps=n_steps,
    )
