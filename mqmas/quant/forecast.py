"""Econometric short-horizon forecast: OLS polynomial trend + 95% prediction band.

For y = X·β + ε with design matrix X = [1, t, t², ...]:

    β̂        = (XᵀX)⁻¹ Xᵀy
    s²        = RSS / (n - p)
    ŷ₀        = x₀ᵀβ̂
    se_pred   = s · sqrt(1 + x₀ᵀ (XᵀX)⁻¹ x₀)
    PI₉₅      = ŷ₀ ± t_{0.975, n-p} · se_pred

The polynomial degree (1 or 2) is selected by the corrected Akaike criterion
(AICc) so the model only bends when the data justify the extra parameter.
Time is centred/scaled to keep XᵀX well conditioned.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from mqmas.models import Forecast


def _fit(t: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    X = np.vander(t, degree + 1, increasing=True)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    n, p = X.shape
    aicc = n * np.log(max(rss, 1e-12) / n) + 2 * p + (2 * p * (p + 1)) / max(n - p - 1, 1)
    return X, beta, rss, aicc


def polynomial_forecast(
    close: np.ndarray, horizon: int = 4, max_degree: int = 2, confidence: float = 0.95
) -> Forecast:
    y = np.asarray(close, dtype=float)
    n = y.size
    if n < 8:
        raise ValueError("need at least 8 observations for the regression")
    t_raw = np.arange(n, dtype=float)
    mu, sd = t_raw.mean(), t_raw.std()
    t = (t_raw - mu) / sd

    best: tuple[int, np.ndarray, np.ndarray, float, float] | None = None
    for d in range(1, max_degree + 1):
        X, beta, rss, aicc = _fit(t, y, d)
        if best is None or aicc < best[4]:
            best = (d, X, beta, rss, aicc)
    assert best is not None
    degree, X, beta, rss, _ = best
    p = degree + 1
    dof = n - p
    s2 = rss / dof
    xtx_inv = np.linalg.pinv(X.T @ X)

    t0 = (n - 1 + horizon - mu) / sd
    x0 = np.vander(np.array([t0]), p, increasing=True)[0]
    y_hat = float(x0 @ beta)
    se_pred = float(np.sqrt(s2 * (1.0 + x0 @ xtx_inv @ x0)))
    t_crit = float(stats.t.ppf(0.5 + confidence / 2.0, dof))

    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    # Instantaneous slope at the last bar, converted back to price units / bar.
    t_last = (n - 1 - mu) / sd
    dy_dt = sum(k * beta[k] * t_last ** (k - 1) for k in range(1, p))
    last = float(y[-1])
    return Forecast(
        degree=degree,
        horizon=horizon,
        r_squared=r2,
        slope_per_bar=float(dy_dt / sd),
        expected_price=y_hat,
        lower_95=y_hat - t_crit * se_pred,
        upper_95=y_hat + t_crit * se_pred,
        expected_return_pct=(y_hat - last) / last,
    )
