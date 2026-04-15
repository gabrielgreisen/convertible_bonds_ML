from __future__ import annotations
from datetime import date
import QuantLib as ql
import pandas as pd
import numpy as np
from urllib.error import URLError



def fetch_treasury_yield_curve_latest():
    """
    Returns (as_of_date_iso, curve_series), where curve_series maps:
    maturity in YEARS (float) -> decimal yield (e.g., 0.045 = 4.5%)

    Pulls the official 'Daily Treasury Par Yield Curve Rates' CSV for the
    current year, falling back to the previous year if needed.
    """
    def _read_year(y):
        url = (
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            f"daily-treasury-rates.csv/{y}/all?_format=csv&field_tdr_date_value={y}"
            "&type=daily_treasury_yield_curve"
        )
        try:
            df = pd.read_csv(url)
            # Normalize
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date")
            # force numeric for all yield columns (coerce N/A)
            for c in df.columns:
                if c != "Date":
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except (URLError, OSError, pd.errors.ParserError):
            return None

    year_now = date.today().year
    df = _read_year(year_now)
    if df is None or df.empty:
        df = _read_year(year_now - 1)
    if df is None or df.empty:
        raise RuntimeError("Could not download Treasury par yield CSV.")

    # Pick the most recent row with at least one yield present
    last = df.dropna(how="all", axis=1).iloc[-1]

    # Map ALL known tenors that might appear. Some years include "1.5 Mo" (≈6 weeks).
    col_to_years = {
        "1 Mo": 1/12,  "1.5 Mo": 1.5/12, "2 Mo": 2/12,  "3 Mo": 3/12,  "4 Mo": 4/12,  "6 Mo": 6/12,
        "1 Yr": 1.0,   "2 Yr": 2.0,      "3 Yr": 3.0,   "5 Yr": 5.0,   "7 Yr": 7.0,
        "10 Yr": 10.0, "20 Yr": 20.0,    "30 Yr": 30.0,
    }

    data = {}
    for col, yrs in col_to_years.items():
        if col in last and pd.notna(last[col]):
            data[yrs] = float(last[col]) / 100.0

    curve_series = pd.Series(data, dtype=float).sort_index()
    if curve_series.empty:
        raise RuntimeError("No yields found in the most recent Treasury row.")
    return last["Date"].date().isoformat(), curve_series


def interest_rate_interpolation(curve_series, x_eval):
    """
    Build a QuantLib monotone cubic natural spline from curve_series
    and return interpolated/extrapolated values at x_eval.
    """
    xs = [float(x) for x in curve_series.index]
    ys = [float(y) for y in curve_series.values]

    if len(xs) == 0:
        raise ValueError("Empty curve")

    xs, ys = zip(*sorted(zip(xs, ys)))
    xs = list(xs)
    ys = list(ys)

    if len(xs) == 1:
        return np.full_like(np.array(x_eval, dtype=float), ys[0], dtype=float)

    spline = ql.MonotonicCubicNaturalSpline(xs, ys)
    x_min, x_max = xs[0], xs[-1]
    results = []

    for t in np.atleast_1d(x_eval):
        t = float(t)

        if x_min <= t <= x_max:
            y = float(spline(t, True))
        else:
            if t < x_min:
                x1, x2, y1, y2 = xs[0], xs[1], ys[0], ys[1]
            else:
                x1, x2, y1, y2 = xs[-2], xs[-1], ys[-2], ys[-1]
            slope = (y2 - y1) / (x2 - x1)
            y = float(y1 + slope * (t - x1))

        results.append(y)

    return np.array(results, dtype=float)

