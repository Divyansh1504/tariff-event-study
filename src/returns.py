"""Return, risk, and portfolio-construction helpers shared by both notebooks."""

import numpy as np
import pandas as pd


def cumulative_return(x):
    """Total return actually earned over the sample -- NOT annualized."""
    return float((1 + x).prod() - 1)


def cagr(x, periods_per_year=252):
    """Annualized growth rate implied by the sample. For a sample shorter than a year this
    is an extrapolation, not a return earned -- always report alongside `cumulative_return`
    and label it explicitly as annualized."""
    n = len(x)
    return float((1 + x).prod() ** (periods_per_year / n) - 1)


def annualized_vol(x, periods_per_year=252):
    return float(x.std() * np.sqrt(periods_per_year))


def sharpe_ratio(x, rf_annual=0.05, periods_per_year=252):
    rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1
    return float((x - rf_daily).mean() / x.std() * np.sqrt(periods_per_year))


def equal_weighted_return(ret, tickers):
    w = np.repeat(1 / len(tickers), len(tickers))
    return (ret[tickers] * w).sum(axis=1)


def value_weighted_return(ret, tickers, market_caps):
    """`market_caps` is a Series indexed by ticker (e.g. cap at the start of the window)."""
    w = (market_caps.reindex(tickers) / market_caps.reindex(tickers).sum()).values
    return (ret[tickers] * w).sum(axis=1)


def performance_summary(returns_by_label: dict, rf_annual=0.05, periods_per_year=252):
    """Build a comparison table with BOTH the actual cumulative return and the annualized
    (CAGR) figure, explicitly labeled, so the two are never conflated (Bug D)."""
    rows = []
    for label, x in returns_by_label.items():
        rows.append({
            "Cumulative Return (actual, sample period)": cumulative_return(x),
            "CAGR (annualized from sample)": cagr(x, periods_per_year),
            "Ann_Vol": annualized_vol(x, periods_per_year),
            "Sharpe": sharpe_ratio(x, rf_annual, periods_per_year),
        })
    return pd.DataFrame(rows, index=list(returns_by_label.keys()))
