"""Data loaders: ETF/stock prices via yfinance, Fama-French factors and FF30 industry
portfolios from Ken French's data library."""

import io
import zipfile

import numpy as np
import pandas as pd
import requests
import yfinance as yf

FF30_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_CSV.zip"
FF3_MONTHLY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
FF5_MONTHLY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
FF5_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
FF_MOM_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"


def _to_monthly_frame(lines):
    """Parse a block of 'YYYYMM,val,val,...' lines into a decimal-scaled monthly DataFrame."""
    df = pd.read_csv(io.StringIO("\n".join(lines)))
    df = df.rename(columns={df.columns[0]: "Date"})
    df = df[df["Date"].astype(str).str.match(r"^\d{6}$")]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.set_index("Date").astype(float) / 100


def load_ff30_monthly():
    """Load the FF30 'Average Value Weighted Returns -- Monthly' table.

    The source CSV stacks four tables that all share the same YYYYMM date format (value-weighted
    returns, equal-weighted returns, number of firms, average firm size). A naive
    `pd.read_csv(..., skiprows=11)` followed by a `^\\d{6}$` regex filter can't tell these apart
    and silently concatenates firm counts and firm sizes onto the return series as if they were
    more months of returns -- this was the root cause of the impossible (300%+) CAR figures in
    the original notebook. Fix: read only the first table's own line range, stopping at the next
    section header rather than the end of the file.
    """
    r = requests.get(FF30_URL, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode("latin1").splitlines()

    start = next(i for i, line in enumerate(raw) if "Average Value Weighted Returns" in line) + 2
    end = next(i for i in range(start, len(raw)) if raw[i].strip() == "" and "Equal Weighted" in raw[i + 1])
    header = raw[start - 1]

    return _to_monthly_frame([header] + raw[start:end])


def load_ff_factors_monthly(url):
    """Load a monthly Fama-French factor file (FF3 or FF5), stopping before the 'Annual
    Factors' footer table so annual figures never leak into the monthly frame."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode("latin1").splitlines()

    end = next((i for i, line in enumerate(raw) if line.strip().startswith("Annual Factors")), len(raw))
    return _to_monthly_frame(raw[3:end])


def load_ff_factors_daily(url, skiprows=3):
    """Load a daily Fama-French factor file (FF5 or Momentum)."""
    df = pd.read_csv(url, skiprows=skiprows)
    df = df.rename(columns={df.columns[0]: "Date"})
    df = df[df["Date"].astype(str).str.len() == 8]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").astype(float) / 100
    df.columns = [c.strip().replace(" ", "").replace("-", "_") for c in df.columns]
    return df


def load_ff6_daily():
    """Fama-French 5 factors + Momentum, merged on common trading days."""
    ff5 = load_ff_factors_daily(FF5_DAILY_URL, skiprows=3)
    mom = load_ff_factors_daily(FF_MOM_DAILY_URL, skiprows=13)
    return ff5.join(mom, how="inner")


def fetch_prices(tickers, start, end):
    """Daily adjusted close prices for a list of tickers (plus any benchmark included in the
    list) over [start, end]. No retry/backoff loop -- yfinance's own batch download handles
    this in one call."""
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return px


def daily_returns(prices):
    return prices.pct_change().dropna()
