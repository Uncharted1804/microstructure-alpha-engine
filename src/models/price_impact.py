"""
OFI → forward price return regression.

Two models:
1. OLS via statsmodels — interpretable coefficients and significance tests
2. Ridge via sklearn — regularised, with time-series cross-validation

Critical: always use TimeSeriesSplit for CV, never random splits.
Random splits on time series leak future information into training.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from loguru import logger


def fit_ofi_regression(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "target_return",
    min_obs: int = 200,
) -> dict:
    """
    OLS regression: target = α + Σ β_i * feature_i + ε

    Uses HAC standard errors (Newey-West) to account for
    autocorrelation in the residuals.
    """
    data = df[feature_cols + [target_col]].dropna()

    if len(data) < min_obs:
        raise ValueError(f"Only {len(data)} obs — need {min_obs}")

    X = sm.add_constant(data[feature_cols])
    y = data[target_col]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})

    coefs = dict(zip(feature_cols, model.params[1:]))
    pvals = dict(zip(feature_cols, model.pvalues[1:]))

    logger.info(f"\n{model.summary()}")

    return {
        "model": model,
        "coefficients": coefs,
        "p_values": pvals,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_obs": int(model.nobs),
        "aic": model.aic,
        "bic": model.bic,
    }


def fit_ridge_model(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "target_return",
    n_cv_splits: int = 5,
) -> dict:
    """
    Ridge regression with TimeSeriesSplit cross-validation.

    StandardScaler normalises features before fitting —
    Ridge is sensitive to feature scale.

    Returns the best alpha, OOS R² per fold, and fitted model.
    """
    data = df[feature_cols + [target_col]].dropna()
    X = data[feature_cols].values
    y = data[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    alphas = np.logspace(-4, 4, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=tscv, scoring="r2")
    ridge_cv.fit(X_scaled, y)

    best_alpha = ridge_cv.alpha_
    logger.info(f"Ridge CV selected alpha = {best_alpha:.4f}")

    oos_r2 = []
    for train_idx, test_idx in tscv.split(X_scaled):
        m = Ridge(alpha=best_alpha)
        m.fit(X_scaled[train_idx], y[train_idx])
        oos_r2.append(r2_score(y[test_idx], m.predict(X_scaled[test_idx])))

    logger.info(
        f"Ridge OOS R² across {n_cv_splits} folds: "
        f"{[f'{x:.4f}' for x in oos_r2]} | Mean: {np.mean(oos_r2):.4f}"
    )

    ridge = Ridge(alpha=best_alpha).fit(X_scaled, y)

    return {
        "model": ridge,
        "scaler": scaler,
        "coefficients": dict(zip(feature_cols, ridge.coef_)),
        "best_alpha": best_alpha,
        "oos_r2_mean": float(np.mean(oos_r2)),
        "oos_r2_by_fold": oos_r2,
        "n_obs": len(y),
    }
