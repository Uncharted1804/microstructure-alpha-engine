"""
TC-adjusted OFI signal backtest.

Transaction costs applied:
  1. Taker fee: 0.04% per side on Binance Futures (4 bps) — round trip = 8 bps
  2. Spread cost: you cross the full spread on a round trip
  3. Kyle impact (optional): λ * quantity
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_transaction_cost(
    mid_price: float,
    spread: float,
    quantity: float = 1.0,
    taker_fee_bps: float = 4.0,
    lambda_impact: float = 0.0,
) -> float:
    """
    Total round-trip transaction cost in price units.

    fee_cost    = (bps/10000) * price * 2   (entry + exit)
    spread_cost = spread                     (full spread for round trip)
    impact_cost = lambda * quantity * 2      (round trip)
    """
    fee_cost = (taker_fee_bps / 10_000) * mid_price * 2
    spread_cost = spread
    impact_cost = lambda_impact * quantity * 2
    return fee_cost + spread_cost + impact_cost


class OFIBacktester:
    """
    Simple OFI signal backtest — no overlapping trades.

    entry_threshold : OFI z-score magnitude to enter a trade
    holding_period  : how many snapshots to hold before exiting
    taker_fee_bps   : exchange fee in basis points (default 4 = 0.04%)
    """

    def __init__(
        self,
        entry_threshold: float = 1.5,
        holding_period: int = 10,
        taker_fee_bps: float = 4.0,
        kyle_lambda: float = 0.0,
    ):
        self.entry_threshold = entry_threshold
        self.holding_period = holding_period
        self.taker_fee_bps = taker_fee_bps
        self.kyle_lambda = kyle_lambda

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return +1 (long), -1 (short), 0 (no trade) for each row."""
        if "ofi_zscore" not in df.columns:
            raise ValueError(
                "ofi_zscore column required — run compute_all_features first"
            )

        signal = pd.Series(0, index=df.index)
        signal[df["ofi_zscore"] > self.entry_threshold] = 1
        signal[df["ofi_zscore"] < -self.entry_threshold] = -1
        return signal

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the backtest. Returns a DataFrame of trades with columns:
          entry_idx, exit_idx, direction, entry_price, exit_price,
          tc_cost, gross_pnl, net_pnl, profitable,
          cumulative_gross_pnl, cumulative_net_pnl
        """
        df = df.reset_index(drop=True).copy()
        signals = self.generate_signals(df)

        trades = []
        last_exit = -1

        for i in range(len(df) - self.holding_period - 1):
            if i <= last_exit:
                continue

            if signals.iloc[i] == 0:
                continue

            direction = signals.iloc[i]
            entry_price = df["mid_price"].iloc[i]
            exit_idx = i + self.holding_period
            exit_price = df["mid_price"].iloc[exit_idx]
            spread = (
                df["spread"].iloc[i] if "spread" in df.columns else entry_price * 0.0001
            )

            tc = compute_transaction_cost(
                mid_price=entry_price,
                spread=spread,
                taker_fee_bps=self.taker_fee_bps,
                lambda_impact=self.kyle_lambda,
            )

            gross_pnl = (exit_price - entry_price) * direction
            net_pnl = gross_pnl - tc

            trades.append(
                {
                    "entry_idx": i,
                    "exit_idx": exit_idx,
                    "entry_time_ms": df["timestamp_ms"].iloc[i],
                    "exit_time_ms": df["timestamp_ms"].iloc[exit_idx],
                    "direction": direction,
                    "ofi_zscore": df["ofi_zscore"].iloc[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "spread": spread,
                    "tc_cost": tc,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "profitable": net_pnl > 0,
                }
            )

            last_exit = exit_idx

        if not trades:
            logger.warning("No trades generated — try lowering entry_threshold")
            return pd.DataFrame()

        results = pd.DataFrame(trades)
        results["cumulative_gross_pnl"] = results["gross_pnl"].cumsum()
        results["cumulative_net_pnl"] = results["net_pnl"].cumsum()
        results["cumulative_tc"] = results["tc_cost"].cumsum()

        self._print_summary(results)
        return results

    def _print_summary(self, r: pd.DataFrame) -> None:
        gross = r["gross_pnl"].sum()
        tc = r["tc_cost"].sum()
        net = r["net_pnl"].sum()
        sharpe = (
            r["net_pnl"].mean() / r["net_pnl"].std() if r["net_pnl"].std() > 0 else 0
        )
        logger.info(
            f"\n{'=' * 45}"
            f"\nBACKTEST RESULTS"
            f"\n{'=' * 45}"
            f"\n  Trades:       {len(r)}"
            f"\n  Win rate:     {r['profitable'].mean():.1%}"
            f"\n  Gross PnL:    {gross:.6f}"
            f"\n  Total TC:     {tc:.6f}  ({tc / abs(gross) * 100:.1f}% of gross)"
            f"\n  Net PnL:      {net:.6f}"
            f"\n  Sharpe:       {sharpe:.2f}"
            f"\n  Survived TC:  {'YES ✓' if net > 0 else 'NO ✗'}"
            f"\n{'=' * 45}"
        )


def parameter_sweep(
    df: pd.DataFrame,
    thresholds: list = [0.5, 1.0, 1.5, 2.0, 2.5],
    holding_periods: list = [5, 10, 20, 30, 50],
    taker_fee_bps: float = 4.0,
) -> pd.DataFrame:
    """
    Grid search over entry thresholds and holding periods.
    Returns a DataFrame you can pivot into a heatmap.
    """
    results = []

    for threshold in thresholds:
        for holding in holding_periods:
            bt = OFIBacktester(threshold, holding, taker_fee_bps)
            trades = bt.run(df)

            if trades.empty or len(trades) < 5:
                results.append(
                    {
                        "threshold": threshold,
                        "holding_period": holding,
                        "n_trades": 0,
                        "net_pnl": 0,
                        "win_rate": np.nan,
                        "sharpe": np.nan,
                    }
                )
                continue

            sharpe = (
                trades["net_pnl"].mean() / trades["net_pnl"].std()
                if trades["net_pnl"].std() > 0
                else 0
            )
            results.append(
                {
                    "threshold": threshold,
                    "holding_period": holding,
                    "n_trades": len(trades),
                    "net_pnl": trades["net_pnl"].sum(),
                    "win_rate": trades["profitable"].mean(),
                    "sharpe": sharpe,
                }
            )

    sweep_df = pd.DataFrame(results)
    best = sweep_df.loc[sweep_df["sharpe"].idxmax()]
    logger.info(
        f"Best: threshold={best['threshold']}, "
        f"holding={best['holding_period']}, "
        f"Sharpe={best['sharpe']:.2f}, "
        f"trades={best['n_trades']}"
    )
    return sweep_df
