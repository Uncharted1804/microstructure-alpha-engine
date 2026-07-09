"""
TC-adjusted OFI signal backtest.

PnL and TC are expressed in BASIS POINTS of the entry price.
This makes the signal strength directly comparable to the cost
regardless of the absolute price level of the asset.

8 bps to break even (4 bps taker fee each way + spread).
OFI signal needs to generate > 8 bps per trade to be profitable.

Transaction costs applied:
  1. Taker fee: 4 bps per side = 8 bps round trip
  2. Spread cost: spread / mid_price * 10000 bps
"""

import numpy as np
import pandas as pd
from loguru import logger


def compute_transaction_cost_bps(
    mid_price: float,
    spread: float,
    taker_fee_bps: float = 4.0,
) -> float:
    """
    Total round-trip transaction cost in BASIS POINTS.

    fee_cost_bps    = taker_fee_bps * 2   (entry + exit)
    spread_cost_bps = (spread / mid_price) * 10000
    """
    fee_cost_bps = taker_fee_bps * 2
    spread_cost_bps = (spread / mid_price) * 10_000
    return fee_cost_bps + spread_cost_bps


class OFIBacktester:
    """
    OFI signal backtest. All PnL figures are in basis points.

    entry_threshold : OFI z-score magnitude to enter a trade
    holding_period  : how many snapshots to hold before exiting
    taker_fee_bps   : exchange fee per side in basis points (default 4)
    """

    def __init__(
        self,
        entry_threshold: float = 1.5,
        holding_period: int = 10,
        taker_fee_bps: float = 4.0,
    ):
        self.entry_threshold = entry_threshold
        self.holding_period = holding_period
        self.taker_fee_bps = taker_fee_bps

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
        Run the backtest. All PnL in basis points.

        Returns DataFrame with columns:
          entry_idx, exit_idx, direction,
          entry_price, exit_price,
          gross_pnl_bps, tc_cost_bps, net_pnl_bps,
          profitable, cumulative columns
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

            # PnL in basis points
            price_move_bps = ((exit_price - entry_price) / entry_price) * 10_000
            gross_pnl_bps = price_move_bps * direction

            # TC in basis points
            tc_bps = compute_transaction_cost_bps(
                entry_price, spread, self.taker_fee_bps
            )
            net_bps = gross_pnl_bps - tc_bps

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
                    "tc_cost_bps": tc_bps,
                    "gross_pnl_bps": gross_pnl_bps,
                    "net_pnl_bps": net_bps,
                    "profitable": net_bps > 0,
                }
            )

            last_exit = exit_idx

        if not trades:
            logger.warning("No trades generated — try lowering entry_threshold")
            return pd.DataFrame()

        results = pd.DataFrame(trades)
        results["cumulative_gross_bps"] = results["gross_pnl_bps"].cumsum()
        results["cumulative_net_bps"] = results["net_pnl_bps"].cumsum()
        results["cumulative_tc_bps"] = results["tc_cost_bps"].cumsum()

        self._print_summary(results)
        return results

    def _print_summary(self, r: pd.DataFrame) -> None:
        gross = r["gross_pnl_bps"].sum()
        tc = r["tc_cost_bps"].sum()
        net = r["net_pnl_bps"].sum()
        sharpe = (
            r["net_pnl_bps"].mean() / r["net_pnl_bps"].std()
            if r["net_pnl_bps"].std() > 0
            else 0
        )
        tc_drag = tc / abs(gross) * 100 if gross != 0 else float("inf")

        logger.info(
            f"\n{'=' * 45}"
            f"\nBACKTEST RESULTS  (all figures in basis points)"
            f"\n{'=' * 45}"
            f"\n  Trades:          {len(r)}"
            f"\n  Win rate:        {r['profitable'].mean():.1%}"
            f"\n  Gross PnL:       {gross:.2f} bps"
            f"\n  Total TC:        {tc:.2f} bps  ({tc_drag:.1f}% of gross)"
            f"\n  Net PnL:         {net:.2f} bps"
            f"\n  Mean per trade:  {r['net_pnl_bps'].mean():.2f} bps"
            f"\n  Break-even TC:   {r['tc_cost_bps'].mean():.2f} bps/trade"
            f"\n  Sharpe:          {sharpe:.2f}"
            f"\n  Survived TC:     {'YES ✓' if net > 0 else 'NO ✗'}"
            f"\n{'=' * 45}"
        )


def parameter_sweep(
    df: pd.DataFrame,
    thresholds: list = [0.5, 1.0, 1.5, 2.0, 2.5],
    holding_periods: list = [5, 10, 20, 50, 100],
    taker_fee_bps: float = 4.0,
) -> pd.DataFrame:
    """
    Grid search over entry thresholds and holding periods.
    Returns a DataFrame you can pivot into a heatmap.
    All PnL figures in basis points.
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
                        "net_pnl_bps": np.nan,
                        "win_rate": np.nan,
                        "sharpe": np.nan,
                        "mean_net_bps": np.nan,
                    }
                )
                continue

            sharpe = (
                trades["net_pnl_bps"].mean() / trades["net_pnl_bps"].std()
                if trades["net_pnl_bps"].std() > 0
                else 0
            )
            results.append(
                {
                    "threshold": threshold,
                    "holding_period": holding,
                    "n_trades": len(trades),
                    "net_pnl_bps": trades["net_pnl_bps"].sum(),
                    "win_rate": trades["profitable"].mean(),
                    "sharpe": sharpe,
                    "mean_net_bps": trades["net_pnl_bps"].mean(),
                }
            )

    sweep_df = pd.DataFrame(results)
    valid = sweep_df.dropna(subset=["sharpe"])

    if not valid.empty:
        best = valid.loc[valid["sharpe"].idxmax()]
        logger.info(
            f"Best params: threshold={best['threshold']}, "
            f"holding={best['holding_period']}, "
            f"Sharpe={best['sharpe']:.2f}, "
            f"mean net={best['mean_net_bps']:.2f} bps/trade"
        )

    return sweep_df
