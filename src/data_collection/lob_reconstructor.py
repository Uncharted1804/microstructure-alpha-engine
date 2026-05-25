"""
LOB state reconstructor and data loader.

Loads raw Parquet files, computes derived columns, and merges
depth snapshots with trade aggregates.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


def load_depth_data(run_id: str, data_dir: Path = Path("./data")) -> pd.DataFrame:
    """Load all depth Parquet files for a run_id into one sorted DataFrame."""
    depth_path = data_dir / "raw" / "depth" / run_id

    if not depth_path.exists():
        raise FileNotFoundError(f"No depth data at {depth_path}")

    files = sorted(depth_path.glob("*.parquet"))
    if not files:
        raise ValueError(f"No Parquet files in {depth_path}")

    logger.info(f"Loading {len(files)} depth file(s) from {depth_path}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = (
        df.sort_values("timestamp_ms")
        .drop_duplicates("timestamp_ms")
        .reset_index(drop=True)
    )
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    logger.info(
        f"Loaded {len(df):,} depth snapshots | "
        f"{df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}"
    )
    return df


def load_trade_data(run_id: str, data_dir: Path = Path("./data")) -> pd.DataFrame:
    """Load all trade Parquet files for a run_id into one sorted DataFrame."""
    trade_path = data_dir / "raw" / "trades" / run_id

    if not trade_path.exists():
        raise FileNotFoundError(f"No trade data at {trade_path}")

    files = sorted(trade_path.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("timestamp_ms").drop_duplicates().reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    logger.info(f"Loaded {len(df):,} trades")
    return df


def add_derived_columns(depth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived LOB metrics to the depth DataFrame.

    New columns:
        mid_price            (best_bid + best_ask) / 2
        spread               best_ask - best_bid
        relative_spread_bps  spread as basis points of mid_price
        bid_depth            total bid volume across all levels
        ask_depth            total ask volume across all levels
        queue_imbalance      (bid_depth - ask_depth) / total  →  range [-1, +1]
        micro_price          volume-weighted mid using top 3 levels
        mid_return           log return between consecutive snapshots
    """
    df = depth_df.copy()

    df["best_bid"] = df["bid_price_1"]
    df["best_ask"] = df["ask_price_1"]
    df["mid_price"] = (df["best_bid"] + df["best_ask"]) / 2.0
    df["spread"] = df["best_ask"] - df["best_bid"]
    df["relative_spread_bps"] = (df["spread"] / df["mid_price"]) * 10_000

    # Count how many levels exist
    n_levels = sum(1 for c in df.columns if c.startswith("bid_qty_"))

    df["bid_depth"] = sum(df[f"bid_qty_{i}"] for i in range(1, n_levels + 1))
    df["ask_depth"] = sum(df[f"ask_qty_{i}"] for i in range(1, n_levels + 1))

    total_depth = df["bid_depth"] + df["ask_depth"]
    df["queue_imbalance"] = (df["bid_depth"] - df["ask_depth"]) / total_depth.replace(
        0, np.nan
    )

    # Micro-price: volume-weighted mid using top 3 levels
    # Anticipates near-term price direction better than plain mid-price
    top = min(3, n_levels)
    bid_num = sum(df[f"bid_price_{i}"] * df[f"bid_qty_{i}"] for i in range(1, top + 1))
    ask_num = sum(df[f"ask_price_{i}"] * df[f"ask_qty_{i}"] for i in range(1, top + 1))
    bid_vol = sum(df[f"bid_qty_{i}"] for i in range(1, top + 1))
    ask_vol = sum(df[f"ask_qty_{i}"] for i in range(1, top + 1))
    df["micro_price"] = (bid_num + ask_num) / (bid_vol + ask_vol)

    # Log return between consecutive snapshots
    df["mid_return"] = np.log(df["mid_price"] / df["mid_price"].shift(1))

    return df


def merge_depth_and_trades(
    depth_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    window_ms: int = 1000,
) -> pd.DataFrame:
    """
    For each depth snapshot, aggregate all trades in the preceding
    window_ms milliseconds and attach them as extra columns.

    New columns added:
        n_trades        number of trades in the window
        trade_volume    total quantity traded
        signed_volume   net signed quantity (buy vol - sell vol)
        buy_volume      aggressive buy volume
        sell_volume     aggressive sell volume
    """
    depth_df = depth_df.sort_values("timestamp_ms").reset_index(drop=True)
    trade_df = trade_df.sort_values("timestamp_ms").reset_index(drop=True)

    results = []
    for _, row in depth_df.iterrows():
        t = row["timestamp_ms"]
        mask = (trade_df["timestamp_ms"] >= t - window_ms) & (
            trade_df["timestamp_ms"] < t
        )
        w = trade_df[mask]
        results.append(
            {
                "timestamp_ms": t,
                "n_trades": len(w),
                "trade_volume": w["quantity"].sum(),
                "signed_volume": w["signed_qty"].sum(),
                "buy_volume": w.loc[~w["is_buyer_maker"], "quantity"].sum(),
                "sell_volume": w.loc[w["is_buyer_maker"], "quantity"].sum(),
            }
        )

    trade_agg = pd.DataFrame(results)
    merged = depth_df.merge(trade_agg, on="timestamp_ms", how="left")
    fill_cols = [
        "n_trades",
        "trade_volume",
        "signed_volume",
        "buy_volume",
        "sell_volume",
    ]
    merged[fill_cols] = merged[fill_cols].fillna(0)

    return merged
