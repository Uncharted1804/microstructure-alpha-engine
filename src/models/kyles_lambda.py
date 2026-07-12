"""
Kyle's Lambda (price impact coefficient) estimation.

Reference: Kyle, A.S. (1985). "Continuous Auctions and Insider Trading."
           Econometrica, 53(6), 1315-1335.

Model: ΔP = α + λ * signed_order_flow + ε
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger


def estimate_kyles_lambda_ols(
    price_changes: pd.Series,
    signed_flow: pd.Series,
    min_obs: int = 50,
) -> dict:
    """
    Estimate Kyle's lambda via OLS with HAC (Newey-West) standard errors.

    HAC = Heteroskedasticity and Autocorrelation Consistent. We use this
    instead of plain OLS standard errors because financial time series
    have autocorrelated residuals — plain OLS would understate uncertainty.

    Returns dict with: lambda, alpha, r_squared, p_value_lambda, n_obs
    """
    df = pd.DataFrame({"dp": price_changes, "flow": signed_flow}).dropna()

    if len(df) < min_obs:
        logger.warning(f"Only {len(df)} obs — below minimum {min_obs}")
        return {"lambda": np.nan, "r_squared": np.nan, "n_obs": len(df)}

    X = sm.add_constant(df["flow"])
    y = df["dp"]

    try:
        model = sm.OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 10},
        )
        return {
            "lambda": model.params["flow"],
            "alpha": model.params["const"],
            "r_squared": model.rsquared,
            "p_value_lambda": model.pvalues["flow"],
            "t_stat_lambda": model.tvalues["flow"],
            "ci_lower": model.conf_int().loc["flow", 0],
            "ci_upper": model.conf_int().loc["flow", 1],
            "n_obs": int(model.nobs),
        }
    except Exception as e:
        logger.error(f"OLS failed: {e}")
        return {"lambda": np.nan, "n_obs": len(df)}


def rolling_lambda(
    df: pd.DataFrame,
    window: int = 500,
    step: int = 50,
    price_col: str = "mid_return",
    flow_col: str = "signed_volume",
) -> pd.DataFrame:
    """
    Compute Kyle's lambda over rolling windows.

    window: number of snapshots per estimate
    step:   how many snapshots to advance between estimates

    Returns a DataFrame with one row per window containing
    timestamp, lambda, r_squared, n_obs.
    """
    results = []
    n = len(df)

    for start in range(0, n - window, step):
        end = start + window
        chunk = df.iloc[start:end]
        est = estimate_kyles_lambda_ols(chunk[price_col], chunk[flow_col])
        est["timestamp_ms"] = df.iloc[start + window // 2]["timestamp_ms"]
        results.append(est)

    result_df = pd.DataFrame(results)
    logger.info(
        f"Rolling lambda: {len(result_df)} windows | "
        f"Mean λ = {result_df['lambda'].mean():.6f}"
    )
    return result_df


def decompose_price_impact(
    df: pd.DataFrame,
    short_horizon: int = 5,
    long_horizon: int = 50,
) -> dict:
    """
    Decompose price impact into permanent and temporary components.

    Method:
      Short-horizon lambda ≈ total impact  (before mean reversion)
      Long-horizon lambda  ≈ permanent impact (after mean reversion)
      Temporary = total - permanent

    The reversion_ratio tells you what fraction of the price move reverses.
    High reversion = mostly temporary (liquidity) impact.
    Low reversion  = mostly permanent (information) impact.
    """
    short_dp = df["mid_price"].diff(short_horizon).shift(-short_horizon)
    long_dp = df["mid_price"].diff(long_horizon).shift(-long_horizon)
    flow = df["signed_volume"]

    total_est = estimate_kyles_lambda_ols(short_dp, flow)
    perm_est = estimate_kyles_lambda_ols(long_dp, flow)

    total = total_est.get("lambda", np.nan)
    perm = perm_est.get("lambda", np.nan)
    temp = total - perm if not np.isnan(total) else np.nan
    reversion = temp / total if total and total != 0 else np.nan

    logger.info(
        f"\nPrice impact decomposition:"
        f"\n  Total λ:      {total:.6f}"
        f"\n  Permanent λ:  {perm:.6f}  ({100 * (1 - reversion):.1f}% of total)"
        f"\n  Temporary λ:  {temp:.6f}  ({100 * reversion:.1f}% of total — reverses)"
    )

    return {
        "total_lambda": total,
        "permanent_lambda": perm,
        "temporary_lambda": temp,
        "reversion_ratio": reversion,
        "total_r2": total_est.get("r_squared"),
        "permanent_r2": perm_est.get("r_squared"),
    }
