"""
Order Flow Imbalance (OFI) feature computation.

Based on Cont, Kukanov & Stoikov (2014):
"The Price Impact of Order Book Events"
Journal of Financial Econometrics, 12(1), 47-88.
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_ofi_single_level(df: pd.DataFrame, level: int = 1) -> pd.Series:
    """
    Compute OFI at a single price level.

    For each snapshot, compare the current LOB state to the previous one:

    BID side contribution:
        If bid price rose or held  →  +current_bid_qty  (buying pressure held/grew)
        If bid price fell          →  -previous_bid_qty  (buyers retreated)

    ASK side contribution:
        If ask price fell or held  →  +current_ask_qty  (selling pressure held/grew)
        If ask price rose          →  -previous_ask_qty  (sellers retreated)

    OFI = bid_contribution - ask_contribution
    Positive = buying pressure > selling pressure → expect price to rise
    """
    bid_p = df[f"bid_price_{level}"]
    bid_q = df[f"bid_qty_{level}"]
    ask_p = df[f"ask_price_{level}"]
    ask_q = df[f"ask_qty_{level}"]

    prev_bid_p = bid_p.shift(1)
    prev_bid_q = bid_q.shift(1)
    prev_ask_p = ask_p.shift(1)
    prev_ask_q = ask_q.shift(1)

    bid_contrib = np.where(
        prev_bid_p.isna(), np.nan, np.where(bid_p >= prev_bid_p, bid_q, -prev_bid_q)
    )

    ask_contrib = np.where(
        prev_ask_p.isna(), np.nan, np.where(ask_p <= prev_ask_p, ask_q, -prev_ask_q)
    )

    return pd.Series(bid_contrib - ask_contrib, index=df.index)


def compute_multi_level_ofi(
    df: pd.DataFrame,
    n_levels: int = 5,
    weights: str = "uniform",
) -> pd.DataFrame:
    """
    Compute OFI at each level and combine into one signal.

    weights='uniform'     → equal weight to all levels
    weights='exponential' → best level gets most weight, decays deeper
    """
    result = {}

    for lvl in range(1, n_levels + 1):
        result[f"ofi_{lvl}"] = compute_ofi_single_level(df, level=lvl)

    ofi_df = pd.DataFrame(result, index=df.index)

    if weights == "uniform":
        w = np.ones(n_levels) / n_levels
    elif weights == "exponential":
        w = np.array([0.5 ** (lvl - 1) for lvl in range(1, n_levels + 1)])
        w = w / w.sum()
    else:
        raise ValueError(f"Unknown weights: {weights}")

    cols = [f"ofi_{lvl}" for lvl in range(1, n_levels + 1)]
    ofi_df["ofi_combined"] = ofi_df[cols].values @ w

    return ofi_df


def compute_all_features(
    depth_df: pd.DataFrame,
    n_levels: int = 5,
    ofi_rolling_window: int = 20,
    forward_horizon: int = 10,
) -> pd.DataFrame:
    """
    Full feature pipeline. Returns the enriched DataFrame with:

    OFI features:
        ofi_1 … ofi_5      raw OFI at each level
        ofi_combined        weighted combination
        ofi_exp_combined    exponentially-weighted combination
        ofi_rolling         rolling mean of ofi_combined (smoothed)
        ofi_zscore          z-score: how unusual is the current OFI?

    Target variable:
        target_return       forward log return over next `forward_horizon` snapshots
        target_direction    +1 (price rises), -1 (price falls), 0 (flat)

    Lag features:
        spread_lag1, spread_change
        qi_lag1, qi_change      (queue imbalance lags)
        signed_vol_roll         rolling signed volume
    """
    df = depth_df.copy()

    # --- Uniform-weighted OFI ---
    ofi_df = compute_multi_level_ofi(df, n_levels=n_levels, weights="uniform")

    # --- Exponential-weighted OFI ---
    ofi_exp = compute_multi_level_ofi(df, n_levels=n_levels, weights="exponential")
    ofi_df["ofi_exp_combined"] = ofi_exp["ofi_combined"]

    df = pd.concat([df, ofi_df], axis=1)

    # --- Rolling OFI and z-score ---
    df["ofi_rolling"] = (
        df["ofi_combined"].rolling(ofi_rolling_window, min_periods=1).mean()
    )
    df["ofi_rolling_std"] = (
        df["ofi_combined"].rolling(ofi_rolling_window, min_periods=1).std()
    )
    df["ofi_zscore"] = (df["ofi_combined"] - df["ofi_rolling"]) / df[
        "ofi_rolling_std"
    ].replace(0, np.nan)

    # --- Target: forward mid-price return ---
    # This is the thing you're predicting.
    # shift(-forward_horizon) looks forward in time.
    # rolling(forward_horizon).sum() accumulates the return over that window.
    df["target_return"] = (
        df["mid_return"].shift(-forward_horizon).rolling(forward_horizon).sum()
    )
    df["target_direction"] = np.sign(df["target_return"])

    # --- Lag features ---
    if "spread" in df.columns:
        df["spread_lag1"] = df["spread"].shift(1)
        df["spread_change"] = df["spread"] - df["spread_lag1"]

    if "queue_imbalance" in df.columns:
        df["qi_lag1"] = df["queue_imbalance"].shift(1)
        df["qi_change"] = df["queue_imbalance"] - df["qi_lag1"]

    if "signed_volume" in df.columns:
        df["signed_vol_roll"] = df["signed_volume"].rolling(ofi_rolling_window).mean()

    logger.info(
        f"Features done. Shape: {df.shape} | "
        f"Target non-null: {df['target_return'].notna().sum():,}"
    )

    return df
