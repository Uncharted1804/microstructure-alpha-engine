"""Unit tests for model components."""

import numpy as np
import pandas as pd
import pytest
from src.backtest.signal_backtest import compute_transaction_cost_bps, OFIBacktester


def test_tc_is_positive():
    tc = compute_transaction_cost_bps(mid_price=30_000, spread=1.0)
    assert tc > 0


def test_tc_scales_with_fee():
    tc_low = compute_transaction_cost_bps(30_000, 1.0, taker_fee_bps=2.0)
    tc_high = compute_transaction_cost_bps(30_000, 1.0, taker_fee_bps=8.0)
    assert tc_high > tc_low


def test_tc_is_8bps_for_standard_params():
    # 4 bps fee x 2 sides = 8 bps, spread on $76k asset is negligible
    tc = compute_transaction_cost_bps(mid_price=76_000, spread=0.01, taker_fee_bps=4.0)
    assert abs(tc - 8.0) < 0.01


def test_signals_are_valid_values():
    df = pd.DataFrame(
        {
            "timestamp_ms": range(100),
            "mid_price": [30_000.0] * 100,
            "ofi_zscore": np.random.normal(0, 1, 100),
            "spread": [1.0] * 100,
        }
    )
    bt = OFIBacktester(entry_threshold=1.0, holding_period=5)
    signals = bt.generate_signals(df)
    assert set(signals.unique()).issubset({-1, 0, 1})


def test_requires_ofi_zscore_column():
    df = pd.DataFrame({"mid_price": [1, 2, 3], "timestamp_ms": [0, 1, 2]})
    bt = OFIBacktester()
    with pytest.raises(ValueError, match="ofi_zscore"):
        bt.generate_signals(df)
