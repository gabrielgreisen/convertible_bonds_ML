import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import QuantLib as ql
import time
from convertible_pricer_class import Tsiveriotis_Fernandes_Pricer

FREQ_MAP = {1: ql.Annual, 2: ql.Semiannual, 4: ql.Quarterly}

DS_FRAC = 0.01   # 1% relative spot bump
DVOL    = 0.01   # 1 vol point absolute bump


def compute_greeks(rows_df, chunk_size=15_000, out_dir="greeks_output", worker_id=0):
    """
    Compute delta, gamma, vega for each row in rows_df using
    the Tsiveriotis-Fernandes QuantLib pricer with finite differences.
    """
    os.makedirs(out_dir, exist_ok=True)

    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    todays_date = calendar.adjust(ql.Date.todaysDate())
    ql.Settings.instance().evaluationDate = todays_date
    day_count = ql.Actual365Fixed()

    pricer = Tsiveriotis_Fernandes_Pricer(
        todays_date=todays_date,
        calendar=calendar,
        day_count=day_count,
        steps_binomial=10_000,
    )

    N = len(rows_df)
    data = []
    chunk = 0
    start_time = time.perf_counter()

    for i in range(N):

        if i % chunk_size == 0 and i != 0:
            pd.DataFrame(data).to_csv(
                f"{out_dir}/w{worker_id}_greeks_chunk{chunk}.csv", index=False
            )
            chunk += 1
            elapsed = time.perf_counter() - start_time
            print(f"Worker {worker_id} chunk {chunk-1} time {elapsed:.1f}s")
            start_time = time.perf_counter()
            data.clear()

        row = rows_df.iloc[i]

        S             = float(row["S"])
        r             = float(row["r"])
        q             = float(row["q"])
        bs_vol        = float(row["bs_vol"])
        credit_spread = float(row["credit_spread"])
        redemption    = float(row["redemption"])
        coupon_rate   = float(row["coupon_rate"])
        frequency     = FREQ_MAP.get(int(row["frequency"]), ql.Semiannual)
        maturity_years = float(row["maturity_years"])
        conversion_ratio = float(row["conversion_ratio"])

        maturity_days = int(round(maturity_years * 365))
        issue_date    = todays_date
        maturity_date = calendar.advance(todays_date, ql.Period(maturity_days, ql.Days))
        settlement_days = 2

        try:
            g = pricer.greeks(
                redemption=redemption,
                spot_price=S,
                conversion_ratio=conversion_ratio,
                issue_date=issue_date,
                maturity_date=maturity_date,
                coupon_rate=coupon_rate,
                frequency=frequency,
                settlement_days=settlement_days,
                r=r,
                q=q,
                bs_volatility=bs_vol,
                credit_spread_rate=credit_spread,
                ds_frac=DS_FRAC,
                dvol=DVOL,
            )
            data.append({
                "delta": g["delta"],
                "gamma": g["gamma"],
                "vega":  g["vega"],
            })
        except Exception:
            data.append({
                "delta": np.nan,
                "gamma": np.nan,
                "vega":  np.nan,
            })

    # Write final chunk
    pd.DataFrame(data).to_csv(
        f"{out_dir}/w{worker_id}_greeks_chunk{chunk}.csv", index=False
    )
    elapsed = time.perf_counter() - start_time
    print(f"Worker {worker_id} chunk {chunk} time {elapsed:.1f}s (final)")
