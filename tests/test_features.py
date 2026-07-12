"""Unit tests for OFI feature computation."""

import numpy as np
import pandas as pd
import pytest
from src.features.ofi_features import compute_ofi_single_level, compute_multi_level_ofi


def make_simple_lob_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base_price = 30_000.0
    rows = []
    for i in range(n):
        mid = base_price + rng.normal(0, 10)
        half_spread = 0.5 + rng.uniform(0, 0.5)
        row = {"timestamp_ms": i * 100, "symbol": "BTCUSDT"}
        for lvl in range(1, 6):
            row[f"bid_price_{lvl}"] = mid - half_spread * lvl
            row[f"bid_qty_{lvl}"] = abs(rng.normal(1, 0.3))
            row[f"ask_price_{lvl}"] = mid + half_spread * lvl
            row[f"ask_qty_{lvl}"] = abs(rng.normal(1, 0.3))
        rows.append(row)
    return pd.DataFrame(rows)


def test_ofi_single_level_shape():
    df = make_simple_lob_df(50)
    ofi = compute_ofi_single_level(df, level=1)
    assert len(ofi) == 50


def test_ofi_first_value_is_nan():
    df = make_simple_lob_df(20)
    ofi = compute_ofi_single_level(df, level=1)
    assert pd.isna(ofi.iloc[0])


def test_multi_level_ofi_columns():
    df = make_simple_lob_df(50)
    result = compute_multi_level_ofi(df, n_levels=5)
    for lvl in range(1, 6):
        assert f"ofi_{lvl}" in result.columns
    assert "ofi_combined" in result.columns


def test_ofi_pure_buy_pressure():
    df = pd.DataFrame(
        {
            "timestamp_ms": [0, 1, 2],
            "bid_price_1": [100.0, 101.0, 102.0],
            "bid_qty_1": [5.0, 6.0, 7.0],
            "ask_price_1": [101.0, 101.0, 101.0],
            "ask_qty_1": [5.0, 5.0, 5.0],
        }
    )
    ofi = compute_ofi_single_level(df, level=1)
    assert ofi.iloc[1] > 0
    assert ofi.iloc[2] > 0
