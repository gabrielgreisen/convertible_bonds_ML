"""Feature engineering for convertible bond pricing datasets."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tier 1 features
# ---------------------------------------------------------------------------

def log_moneyness(df: pd.DataFrame) -> pd.Series:
    """Log of parity over redemption: log(S * conversion_ratio / redemption).

    Measures how deep in- or out-of-the-money the conversion option is.
    """
    return np.log(df["S"] * df["conversion_ratio"] / df["redemption"])


def total_vol(df: pd.DataFrame) -> pd.Series:
    """Annualised volatility scaled to maturity: bs_vol * sqrt(maturity_years).

    Represents the expected total diffusion of the underlying over the bond's life.
    """
    return df["bs_vol"] * np.sqrt(df["maturity_years"])


def income_advantage(df: pd.DataFrame) -> pd.Series:
    """Coupon rate minus dividend yield: coupon_rate - q.

    Positive values indicate the bond holder earns more income than the
    equity holder, reducing the incentive to convert early.
    """
    return df["coupon_rate"] - df["q"]


def risky_discount_rate(df: pd.DataFrame) -> pd.Series:
    """Risk-free rate plus credit spread: r + credit_spread.

    The rate used to discount the bond's cash flows under the issuer's
    credit risk.
    """
    return df["r"] + df["credit_spread"]


def conversion_premium(df: pd.DataFrame) -> pd.Series:
    """Premium of face value over parity: (redemption / (S * conversion_ratio)) - 1.

    Measures how much the bond's face value exceeds its conversion value,
    expressed as a fraction of parity.
    """
    return (df["redemption"] / (df["S"] * df["conversion_ratio"])) - 1


# ---------------------------------------------------------------------------
# Tier 2 features
# ---------------------------------------------------------------------------

def spread_to_vol_ratio(df: pd.DataFrame) -> pd.Series:
    """Credit spread divided by implied volatility: credit_spread / bs_vol.

    Captures the relative importance of credit risk versus equity optionality.
    """
    return df["credit_spread"] / df["bs_vol"]


def sqrt_maturity(df: pd.DataFrame) -> pd.Series:
    """Square root of time to maturity: sqrt(maturity_years).

    Appears naturally in option-pricing formulas and normalises time effects.
    """
    return np.sqrt(df["maturity_years"])


def parity(df: pd.DataFrame) -> pd.Series:
    """Conversion parity: S * conversion_ratio.

    The market value of the shares received upon conversion.
    """
    return df["S"] * df["conversion_ratio"]


def total_remaining_income(df: pd.DataFrame) -> pd.Series:
    """Undiscounted total coupon income: maturity_years * coupon_rate * redemption.

    Rough measure of the total income the bond holder forgoes upon conversion.
    """
    return df["maturity_years"] * df["coupon_rate"] * df["redemption"]


def real_rate(df: pd.DataFrame) -> pd.Series:
    """Risk-free rate net of dividend yield: r - q.

    Proxy for the cost-of-carry of the underlying equity.
    """
    return df["r"] - df["q"]


# ---------------------------------------------------------------------------
# Tier 3 features
# ---------------------------------------------------------------------------

def credit_vol_product(df: pd.DataFrame) -> pd.Series:
    """Product of credit spread and implied volatility: credit_spread * bs_vol.

    Interaction term capturing joint credit-equity risk.
    """
    return df["credit_spread"] * df["bs_vol"]


def income_to_optionality(df: pd.DataFrame) -> pd.Series:
    """Ratio of total remaining income to scaled optionality value.

    total_remaining_income / (bs_vol * sqrt(maturity_years) * S * conversion_ratio + 0.001)

    Measures the trade-off between holding the bond for income versus
    converting for equity upside.  The small epsilon (0.001) prevents
    division by zero.
    """
    tri = df["maturity_years"] * df["coupon_rate"] * df["redemption"]
    optionality = (
        df["bs_vol"]
        * np.sqrt(df["maturity_years"])
        * df["S"]
        * df["conversion_ratio"]
        + 0.001
    )
    return tri / optionality


def rate_spread_ratio(df: pd.DataFrame) -> pd.Series:
    """Risk-free rate as a share of the risky rate: r / (r + credit_spread).

    Values near 1 indicate negligible credit risk; values near 0 indicate
    credit spread dominates the discount rate.
    """
    return df["r"] / (df["r"] + df["credit_spread"])


_FREQUENCY_MAP = {1: 1, 2: 2, 4: 4}


def frequency_per_year(df: pd.DataFrame) -> pd.Series:
    """Map QuantLib frequency enum integers to numeric payments per year.

    QuantLib encoding: 1 = Annual, 2 = Semiannual, 4 = Quarterly.
    """
    return df["frequency"].map(_FREQUENCY_MAP)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_TIERS: dict[int, list[tuple[str, callable]]] = {
    1: [
        ("log_moneyness", log_moneyness),
        ("total_vol", total_vol),
        ("income_advantage", income_advantage),
        ("risky_discount_rate", risky_discount_rate),
        ("conversion_premium", conversion_premium),
    ],
    2: [
        ("spread_to_vol_ratio", spread_to_vol_ratio),
        ("sqrt_maturity", sqrt_maturity),
        ("parity", parity),
        ("total_remaining_income", total_remaining_income),
        ("real_rate", real_rate),
    ],
    3: [
        ("credit_vol_product", credit_vol_product),
        ("income_to_optionality", income_to_optionality),
        ("rate_spread_ratio", rate_spread_ratio),
        ("frequency_per_year", frequency_per_year),
    ],
}


def engineer_features(
    df: pd.DataFrame,
    tiers: list[int] | None = None,
) -> pd.DataFrame:
    """Add engineered features to a convertible-bond pricing DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with columns: S, conversion_ratio, redemption, bs_vol,
        maturity_years, coupon_rate, q, r, credit_spread, frequency.
    tiers : list[int] | None
        Which feature tiers to compute.  ``[1]``, ``[1, 2]``, or
        ``[1, 2, 3]`` (default).

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the selected engineered features appended.
    """
    if tiers is None:
        tiers = [1, 2, 3]

    result = df.copy()
    for tier in tiers:
        for name, func in _TIERS[tier]:
            result[name] = func(df)
    return result
