"""Alpha estimation and event-study significance testing."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def minimum_detectable_effect(standard_error, alpha=0.05, power=0.80):
    """Minimum detectable effect (MDE) for a two-sided test with the given standard error.

    This is descriptive of a design's power -- given how noisy an estimator already is, how
    large would a true effect need to be for a test at this significance level and power to
    reliably distinguish it from zero? It is not itself a hypothesis test: it takes a
    standard error already produced by a real fit (or derived algebraically from one via
    SE = estimate / t-stat) and asks what effect size that SE could resolve.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    return (z_alpha + z_power) * standard_error


def car_mde(sigma, n, alpha=0.05, power=0.80):
    """MDE for a CAR test: the standard error of a CAR (sum of N i.i.d. abnormal returns with
    per-period std `sigma`) is `sigma * sqrt(N)` -- see `test_car_significance`."""
    return minimum_detectable_effect(sigma * np.sqrt(n), alpha=alpha, power=power)


def test_car_significance(ar_series, n_boot=1000, seed=42):
    """Parametric + bootstrap significance test for a cumulative abnormal return (CAR).

    CAR is the sum of N abnormal returns. Under the standard event-study assumption that
    abnormal returns are i.i.d. with per-period std `sigma`, the CAR's standard error is
    `sigma * sqrt(N)` (variance of a sum of N i.i.d. draws), not `sigma / sqrt(N)` (which
    is the standard error of a *mean*).
    """
    if len(ar_series) < 5:
        return np.nan, np.nan, np.nan

    car = ar_series.sum()
    sigma = ar_series.std(ddof=1)
    t_stat = car / (sigma * np.sqrt(len(ar_series)))

    p_param = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    rng = np.random.default_rng(seed)
    boot_cars = np.array([
        rng.choice(ar_series, size=len(ar_series), replace=True).sum()
        for _ in range(n_boot)
    ])
    p_boot = np.mean(np.abs(boot_cars) >= abs(car))

    return t_stat, p_param, p_boot


def ff30_car_event(FF30, FF3M, FF5M, industry, base_start, base_end, event_start, event_end, model="FF3"):
    """CAR for one FF30 industry: fit a factor model on the baseline window, apply it to the
    event window, return (CAR, AR series). Single implementation used everywhere Report 1
    needs an FF30 event-study CAR (previously duplicated three times with diverging minimum-N
    guards across the notebook)."""
    col = industry
    if col not in FF30.columns:
        if industry == "Util" and "Utils" in FF30.columns:
            col = "Utils"
        elif industry == "Utils" and "Util" in FF30.columns:
            col = "Util"

    if model == "FF3":
        factors = FF3M[["Mkt-RF", "SMB", "HML"]]
    elif model == "FF5":
        factors = FF3M[["Mkt-RF", "SMB", "HML"]].join(FF5M[["RMW", "CMA"]], how="inner")
    else:
        raise ValueError(f"Unknown model: {model}")

    rf = FF3M["RF"]
    panel = FF30[[col]].join(factors, how="inner").join(rf, how="inner")

    base = panel.loc[(panel.index >= base_start) & (panel.index <= base_end)]
    if len(base) < 24:
        return np.nan, pd.Series(dtype=float)
    yb = base[col] - base["RF"]
    Xb = sm.add_constant(base[factors.columns])
    fit = sm.OLS(yb, Xb).fit()

    event = panel.loc[(panel.index >= event_start) & (panel.index <= event_end)]
    if len(event) < 3:
        return np.nan, pd.Series(dtype=float)
    ye = event[col] - event["RF"]
    Xe = sm.add_constant(event[factors.columns])
    ar = (ye - fit.predict(Xe)).dropna()

    return float(ar.sum()), ar


def capm_alpha(port_ret, mkt_ret, rf_daily):
    """CAPM alpha via Newey-West (HAC) standard errors."""
    y = port_ret - rf_daily
    excess_mkt = (mkt_ret - rf_daily).rename("Mkt_RF")
    X = sm.add_constant(excess_mkt)
    model = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return model


def ff6_alpha(port_ret, factors_df, rf_daily):
    """Fama-French 5 + Momentum alpha via Newey-West (HAC) standard errors.
    `factors_df` must contain columns Mkt_RF, SMB, HML, RMW, CMA, Mom."""
    y = port_ret - rf_daily
    X = sm.add_constant(factors_df[["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]])
    model = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return model


if __name__ == "__main__":
    # Sanity check against a synthetic series with known properties before wiring this
    # back into the notebooks (per plan). A t-stat this size should sit in single digits
    # for a null-ish series, not the 100+ the old (inverted) formula produced.
    rng = np.random.default_rng(0)
    N = 60
    true_mean, true_sigma = 0.0, 0.01
    synthetic = pd.Series(rng.normal(true_mean, true_sigma, N))

    car = synthetic.sum()
    sigma = synthetic.std(ddof=1)
    expected_t = car / (sigma * np.sqrt(N))

    t_stat, p_param, p_boot = test_car_significance(synthetic)
    assert abs(t_stat - expected_t) < 1e-9, f"t-stat mismatch: {t_stat} vs {expected_t}"
    assert abs(t_stat) < 5, f"t-stat implausibly large for a null series: {t_stat}"

    # A series with a real, large mean shift should register as significant.
    shifted = pd.Series(rng.normal(0.02, true_sigma, N))
    t_stat_shifted, p_param_shifted, _ = test_car_significance(shifted)
    assert abs(t_stat_shifted) > 5 and p_param_shifted < 0.001, (
        f"Expected an obviously significant t-stat for a shifted series, got t={t_stat_shifted}"
    )

    print(f"Null series   (N={N}, mean=0.00):   t={t_stat:.3f}  p_param={p_param:.3f}  p_boot={p_boot:.3f}")
    print(f"Shifted series(N={N}, mean=0.02):   t={t_stat_shifted:.3f}  p_param={p_param_shifted:.6f}")
    print("OK: test_car_significance formula verified against synthetic series.")
