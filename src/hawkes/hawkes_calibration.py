"""
Hawkes process calibration on trade arrival times.

Pure numpy/scipy implementation — no external Hawkes library required.

The Hawkes process intensity at time t:
    λ(t) = μ + Σ_{t_i < t} α * exp(-β * (t - t_i))

We calibrate μ, α, β by maximising the log-likelihood using scipy.optimize.

References:
    Hawkes (1971). "Spectra of some self-exciting and mutually exciting
    point processes." Biometrika, 58(1), 83-90.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger


def extract_trade_timestamps(
    trade_df: pd.DataFrame,
    direction: Optional[str] = None,
) -> np.ndarray:
    """
    Extract trade timestamps in seconds, zero-indexed from first trade.

    direction: 'buy' = aggressive buys only
               'sell' = aggressive sells only
               None = all trades
    """
    df = trade_df.copy()

    if direction == "buy":
        df = df[~df["is_buyer_maker"]]
    elif direction == "sell":
        df = df[df["is_buyer_maker"]]

    t = df["timestamp_ms"].values.astype(float) / 1000.0
    t = np.sort(t)
    t = t - t[0]
    return t


def calibrate_hawkes_mle_manual(
    timestamps: np.ndarray,
    initial_params: tuple = (1.0, 0.5, 2.0),
    max_iter: int = 1000,
) -> dict:
    """
    Calibrate Hawkes process via maximum likelihood estimation.

    The log-likelihood of a Hawkes process with exponential kernel is:

      LL = -μT  -  (α/β) * Σ(1 - exp(-β*(T-t_i)))
           + Σ log(μ + α * R_i)

    where R_i = Σ_{j<i} exp(-β*(t_i - t_j)) is computed recursively:
      R_i = exp(-β*(t_i - t_{i-1})) * (1 + R_{i-1})

    This recursion is the key computational trick — instead of an O(n²)
    double sum, you compute R in O(n) using the previous value.

    Parameters
    ----------
    timestamps : sorted array of event times in seconds, starting at 0
    initial_params : starting guess for (μ, α, β)
    """
    T = timestamps[-1]
    n = len(timestamps)

    def neg_log_likelihood(params):
        mu, alpha, beta = params

        # Reject invalid parameter regions
        if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
            return 1e10

        # Recursive computation of R_i
        R = np.zeros(n)
        for i in range(1, n):
            dt = timestamps[i] - timestamps[i - 1]
            R[i] = np.exp(-beta * dt) * (1 + R[i - 1])

        intensities = mu + alpha * R
        if np.any(intensities <= 0):
            return 1e10

        ll = (
            -mu * T
            - (alpha / beta) * np.sum(1 - np.exp(-beta * (T - timestamps)))
            + np.sum(np.log(intensities))
        )
        return -ll  # minimise negative log-likelihood

    from scipy.optimize import minimize, Bounds

    result = minimize(
        neg_log_likelihood,
        x0=initial_params,
        method="L-BFGS-B",
        bounds=Bounds(lb=[1e-6, 1e-6, 1e-6], ub=[100, 0.9999, 1000]),
        options={"maxiter": max_iter, "ftol": 1e-10, "gtol": 1e-8},
    )

    mu, alpha, beta = result.x
    branching_ratio = alpha / beta

    logger.info(
        f"Hawkes MLE {'converged' if result.success else 'DID NOT CONVERGE'}:"
        f"\n  μ (baseline rate):   {mu:.4f} events/sec"
        f"\n  α (excitation):      {alpha:.4f}"
        f"\n  β (decay rate):      {beta:.4f}"
        f"\n  n = α/β (branching): {branching_ratio:.4f} "
        f"({'STABLE' if branching_ratio < 1 else 'UNSTABLE — n >= 1'})"
        f"\n  Log-likelihood:      {-result.fun:.2f}"
    )

    return {
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": branching_ratio,
        "converged": result.success,
        "log_likelihood": -result.fun,
    }


def compute_hawkes_intensity(
    timestamps: np.ndarray,
    params: dict,
    eval_times: Optional[np.ndarray] = None,
) -> tuple:
    """
    Compute the conditional intensity λ(t) at evaluation times.
    Used for plotting — shows how the arrival rate spikes after each trade.
    """
    mu, alpha, beta = params["mu"], params["alpha"], params["beta"]
    T = timestamps[-1]

    if eval_times is None:
        eval_times = np.linspace(0, T, 2000)

    intensities = np.full_like(eval_times, mu, dtype=float)
    for t_i in timestamps:
        mask = eval_times > t_i
        intensities[mask] += alpha * np.exp(-beta * (eval_times[mask] - t_i))

    return eval_times, intensities


def fit_rolling_hawkes(
    timestamps: np.ndarray,
    window_duration: float = 600.0,
    step_duration: float = 60.0,
) -> pd.DataFrame:
    """
    Fit Hawkes process in rolling windows to see how market activity
    changes over time. Default: 10-minute windows, advancing 1 minute.

    Returns a DataFrame with one row per window.
    """
    T = timestamps[-1]
    results = []
    t_start = 0.0

    while t_start + window_duration <= T:
        t_end = t_start + window_duration
        mask = (timestamps >= t_start) & (timestamps < t_end)
        window_ts = timestamps[mask] - t_start

        if len(window_ts) >= 20:
            try:
                params = calibrate_hawkes_mle_manual(window_ts)
                params.update(
                    {
                        "window_start": t_start,
                        "window_end": t_end,
                        "n_events": len(window_ts),
                        "empirical_rate": len(window_ts) / window_duration,
                    }
                )
                results.append(params)
            except Exception as e:
                logger.warning(f"Window {t_start:.0f}s failed: {e}")

        t_start += step_duration

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        logger.info(
            f"Rolling Hawkes: {len(result_df)} windows | "
            f"Mean branching ratio: {result_df['branching_ratio'].mean():.3f}"
        )
    return result_df
