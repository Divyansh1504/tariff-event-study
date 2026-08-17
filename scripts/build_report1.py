"""Builds notebooks/report1_industry_tariff_impact.ipynb from scratch, importing the
corrected logic from src/. Run this, then execute the notebook separately with nbclient."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


md("""
# Report 1 — Industry Tariff Impact (FF30 + ETF Event Study)

**Question:** did the 2025 US tariff announcements produce a measurable, statistically
distinguishable effect on returns for tariff-exposed industries (Autos, Steel, Semiconductors,
Health, Finance, Utilities)?

**Event window:** 2025-04-01 to 2025-09-30 (six months around the "Liberation Day" tariff
announcement and subsequent policy actions).
**Baseline window:** 2023-01-01 to 2025-03-31.

**Method:** market-model / Fama-French abnormal returns (AR), summed into a cumulative
abnormal return (CAR) per industry, tested for significance with both a parametric t-test and a
bootstrap. All CAR and significance logic lives in `src/regression.py`; this notebook drives it.
""")

code("""
import sys
sys.path.insert(0, "../src")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model

from data_fetch import (
    load_ff30_monthly, load_ff_factors_monthly, fetch_prices, daily_returns,
    FF3_MONTHLY_URL, FF5_MONTHLY_URL,
)
from regression import ff30_car_event, test_car_significance
from plotting import plot_car_curve, plot_industry_heatmap

plt.rcParams["figure.figsize"] = (9, 5)

START_DATE = "2023-01-01"
BASE_START, BASE_END = pd.Timestamp(START_DATE), pd.Timestamp("2025-03-31")
RECENT_START, RECENT_END = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-09-30")
BENCH = "SPY"

ETF_MAP = {
    "Autos": "CARZ", "Steel": "XME", "TechSemis": "SOXX",
    "Health": "XLV", "Finance": "XLF", "Utilities": "XLU",
}
FF30_TO_ETF = {
    "Autos": ETF_MAP["Autos"], "Steel": ETF_MAP["Steel"], "BusEq": ETF_MAP["TechSemis"],
    "Hlth": ETF_MAP["Health"], "Fin": ETF_MAP["Finance"], "Util": ETF_MAP["Utilities"],
}
INDUSTRY_LABELS = {"Autos": "Autos", "Steel": "Steel", "BusEq": "Semis",
                    "Hlth": "Health", "Fin": "Finance", "Util": "Utilities"}
""")

md("""
## Data

Daily ETF prices come from Yahoo Finance (`yfinance`); FF30 industry portfolios and Fama-French
3/5-factor data come from Ken French's data library. `load_ff30_monthly` (see `src/data_fetch.py`)
reads only the "Average Value Weighted Returns -- Monthly" table's own line range from the source
CSV — the original notebook's loader read to end-of-file and silently absorbed three other stacked
tables (equal-weighted returns, firm counts, firm size) under the same `YYYYMM` date format, which
was the root cause of Bug B (see `../CORRECTIONS.md`).
""")

code("""
FF30 = load_ff30_monthly()
FF3M = load_ff_factors_monthly(FF3_MONTHLY_URL)
FF5M = load_ff_factors_monthly(FF5_MONTHLY_URL)

print("FF30 shape:", FF30.shape, "| columns:", len(FF30.columns))
print("FF3M shape:", FF3M.shape, "| FF5M shape:", FF5M.shape)

event_window_check = FF30.loc[(FF30.index >= RECENT_START) & (FF30.index <= RECENT_END)]
print("Rows in the Apr-Sep 2025 event window (should be 6):", len(event_window_check))
assert len(event_window_check) == 6, "FF30 event window is not 6 monthly rows -- loader regressed"
""")

code("""
etf_tickers = list(ETF_MAP.values())
px = fetch_prices(etf_tickers + [BENCH], START_DATE, "2025-09-30")
ret = daily_returns(px)
vol = px.copy()  # placeholder overwritten below with actual volume
logret = np.log(px / px.shift(1)).dropna()

# Volume needs a separate yfinance pull (Close-only download above doesn't include Volume)
import yfinance as yf
vol_raw = yf.download(etf_tickers + [BENCH], start=START_DATE, end="2025-09-30",
                       auto_adjust=True, progress=False)["Volume"]
vol = vol_raw

print(ret.shape, vol.shape, logret.shape)
ret.tail()
""")

md("## Daily ETF market-model abnormal returns\n\nDemo on Autos (CARZ), then all six ETFs.")

code("""
def expected_market_model(y, mkt):
    X = sm.add_constant(mkt.reindex(y.index))
    model = sm.OLS(y, X, missing="drop").fit()
    return model, model.predict(X)

def realized_volatility(lr, window=30):
    return (lr - lr.rolling(window).mean()).pow(2).rolling(window).sum().pow(0.5)

def volume_shock(v, window=60):
    return (v - v.rolling(window).median()) / v.rolling(window).median()

etf = "CARZ"
mod, exp = expected_market_model(ret[etf], ret[BENCH])
ar = (ret[etf] - exp).dropna()
print(f"{etf}: R2={mod.rsquared:.3f}  mean AR={ar.mean():.5f}  std AR={ar.std():.5f}")
""")

code("""
def _slice(df, start, end):
    return df.loc[(df.index >= start) & (df.index <= end)]

def etf_metrics_window(etf, start, end):
    y = _slice(ret[[etf]], start, end)[etf]
    m = _slice(ret[[BENCH]], start, end)[BENCH]
    v = _slice(vol[[etf]], start, end)[etf]
    lr = _slice(logret[[etf]], start, end)[etf]

    if len(y) < 60:
        return {k: np.nan for k in [
            "ETF_R2", "ETF_Mean_AR", "ETF_Std_AR", "ETF_RealVol_30d_avg",
            "ETF_VolShock_avg", "GARCH_AIC", "GARCH_BIC", "GARCH_LB_p",
        ]}

    X = sm.add_constant(m.reindex(y.index))
    model = sm.OLS(y, X, missing="drop").fit()
    ar = (y - model.predict(X)).dropna()
    rv = realized_volatility(lr, 30)
    vs = volume_shock(v, 60)

    g_stats = {"GARCH_AIC": np.nan, "GARCH_BIC": np.nan, "GARCH_LB_p": np.nan}
    try:
        series = (y.dropna() * 100)
        if len(series) >= 100:
            am = arch_model(series, vol="GARCH", p=1, q=1, dist="normal")
            res = am.fit(disp="off")
            lb = acorr_ljungbox(res.std_resid.dropna(), lags=[10], return_df=True)
            g_stats = {"GARCH_AIC": res.aic, "GARCH_BIC": res.bic,
                       "GARCH_LB_p": float(lb["lb_pvalue"].iloc[0])}
    except Exception:
        pass

    return {
        "ETF_R2": model.rsquared, "ETF_Mean_AR": float(ar.mean()), "ETF_Std_AR": float(ar.std()),
        "ETF_RealVol_30d_avg": float(rv.mean()), "ETF_VolShock_avg": float(vs.mean()), **g_stats,
    }

daily_metrics_rows = []
for ff_ind, etf in FF30_TO_ETF.items():
    m_recent = etf_metrics_window(etf, RECENT_START, RECENT_END)
    m_base = etf_metrics_window(etf, BASE_START, BASE_END)
    daily_metrics_rows.append({"Industry": INDUSTRY_LABELS[ff_ind], "ETF": etf,
                                **{f"BASE_{k}": v for k, v in m_base.items()},
                                **{f"RECENT_{k}": v for k, v in m_recent.items()}})

daily_metrics_df = pd.DataFrame(daily_metrics_rows).set_index(["Industry", "ETF"])
daily_metrics_df.round(4)
""")

md("""
## Event-study CARs (baseline-fit, applied to the Apr-Sep 2025 event window)

Both the ETF (daily) and FF30 (monthly) CARs use `ff30_car_event` / an equivalent market-model
fit-then-apply approach: the factor model is estimated on the 2023-01-01 to 2025-03-31 baseline,
then applied out-of-sample to the event window to get abnormal returns. This is the corrected,
single implementation — the original notebook had three near-duplicate versions of this logic
(cells 18, 23, 44) that inherited the corrupted FF30 frame and diverged in their minimum-N guards.
""")

code("""
def etf_car_event(etf, base_start, base_end, event_start, event_end):
    y_base = _slice(ret[[etf]], base_start, base_end)[etf]
    m_base = _slice(ret[[BENCH]], base_start, base_end)[BENCH]
    if len(y_base) < 60:
        return np.nan, pd.Series(dtype=float)
    Xb = sm.add_constant(m_base)
    mod = sm.OLS(y_base, Xb, missing="drop").fit()

    y_event = _slice(ret[[etf]], event_start, event_end)[etf]
    m_event = _slice(ret[[BENCH]], event_start, event_end)[BENCH]
    Xe = sm.add_constant(m_event.reindex(y_event.index))
    ar = (y_event - mod.predict(Xe)).dropna()
    return float(ar.sum()), ar

results = []
for ff_ind, etf in FF30_TO_ETF.items():
    etf_car, etf_ar = etf_car_event(etf, BASE_START, BASE_END, RECENT_START, RECENT_END)
    t_etf, p_etf, pb_etf = test_car_significance(etf_ar)

    ff3_car, ff3_ar = ff30_car_event(FF30, FF3M, FF5M, ff_ind, BASE_START, BASE_END,
                                      RECENT_START, RECENT_END, "FF3")
    t_ff3, p_ff3, pb_ff3 = test_car_significance(ff3_ar)

    ff5_car, ff5_ar = ff30_car_event(FF30, FF3M, FF5M, ff_ind, BASE_START, BASE_END,
                                      RECENT_START, RECENT_END, "FF5")
    t_ff5, p_ff5, pb_ff5 = test_car_significance(ff5_ar)

    results.append({
        "Industry": INDUSTRY_LABELS[ff_ind], "ETF": etf,
        "ETF_CAR": etf_car, "ETF_N": len(etf_ar), "ETF_t": t_etf, "ETF_p_boot": pb_etf,
        "FF3_CAR": ff3_car, "FF3_N": len(ff3_ar), "FF3_t": t_ff3, "FF3_p_boot": pb_ff3,
        "FF5_CAR": ff5_car, "FF5_N": len(ff5_ar), "FF5_t": t_ff5, "FF5_p_boot": pb_ff5,
    })

car_summary = pd.DataFrame(results).set_index(["Industry", "ETF"])
car_summary.round(4)
""")

code("""
for (industry, etf), row in car_summary.iterrows():
    _, ar = etf_car_event(etf, BASE_START, BASE_END, RECENT_START, RECENT_END)
    plot_car_curve(ar, f"CAR Curve (ETF, daily): {industry} [{etf}]")
    plt.savefig(f"../figures/car_etf_{industry.lower()}.png", dpi=110, bbox_inches="tight")
    plt.show()
""")

md("""
## Corrected industry ranking — the blocking check

The original notebook ranked Autos, Semis, and Steel as the most tariff-affected industries by
FF30 CAR magnitude, and Report 2's 15-stock portfolio was built on that ranking. Re-running the
ranking after fixing Bugs A and B does **not** reproduce that result cleanly: it depends on which
model you use, and none of the industry CARs are statistically distinguishable from noise in any
specification (bootstrap p-values all sit around 0.5-0.9). See `../CORRECTIONS.md` for the full
comparison across FF3, FF5, and the ETF market-model. Report 2 is published as analysis of a
**pre-selected** tariff-exposed basket, not as analysis of a validated "most-affected" ranking.
""")

code("""
ranking = car_summary.reset_index()[["Industry", "ETF", "FF3_CAR", "FF5_CAR", "ETF_CAR"]].copy()
ranking["abs_FF3_CAR"] = ranking["FF3_CAR"].abs()
ranking = ranking.sort_values("abs_FF3_CAR", ascending=False).drop(columns="abs_FF3_CAR")
print("Corrected ranking by |FF3 CAR| (most- to least-affected):")
ranking
""")

md("""
## Statistical power: could this design have detected a plausible industry effect?

A null result ("no industry CAR is significant") is a different claim from "this design could
have detected an effect if one existed." Below: the minimum detectable effect (MDE) at 80% power
for each industry/model, derived algebraically from the CAR and t-stat already computed above
(`sigma = CAR / (t*sqrt(N))`, so `MDE = z_combined * CAR / t` — see `src/regression.py::car_mde`).
No new fetch, no new specification — this describes the power of the test already run.
""")

code("""
Z_COMBINED = 2.801585  # z(0.975) + z(0.80), two-sided alpha=0.05, 80% power

power_rows = []
for (industry, etf), row in car_summary.iterrows():
    row_data = {"Industry": industry, "ETF": etf}
    for spec, car_col, t_col, n_col in [("ETF", "ETF_CAR", "ETF_t", "ETF_N"),
                                          ("FF3", "FF3_CAR", "FF3_t", "FF3_N"),
                                          ("FF5", "FF5_CAR", "FF5_t", "FF5_N")]:
        car, t, n = row[car_col], row[t_col], row[n_col]
        mde = Z_COMBINED * abs(car / t) if t != 0 else np.nan
        row_data[f"{spec}_N"] = n
        row_data[f"{spec}_CAR"] = car
        row_data[f"{spec}_MDE"] = mde
    power_rows.append(row_data)

industry_power_table = pd.DataFrame(power_rows).set_index(["Industry", "ETF"])
industry_power_table.round(4)
""")

md("""
**Was this design capable of detecting a plausible tariff effect? No — substantially
underpowered, at both the daily-ETF and monthly-FF30 layers.** ETF-based MDEs run roughly
17-45% over six months; FF30-based MDEs run roughly 9-31% for five of the six industries. (Autos
is a degenerate case here — its FF3 and FF5 t-stats are both close to zero, which makes the
algebraic `sigma = CAR/(t*sqrt(N))` recovery numerically unstable and produces an MDE near 100%.
That's an artifact of dividing by a near-zero t-stat, not a real detection threshold — read
Autos's FF3/FF5 MDE as "not meaningfully estimable from this table," not as "100% required.")
Compare the stable estimates to what a real tariff effect might plausibly look like: single-digit
to low-double-digit percentage moves over six months, not the 20%+ swings this design would need
to reliably detect. Only Steel's ETF-based CAR (37.9% observed vs. ~45% MDE) comes close to the
detection threshold; every other industry/model combination sits well below it. **The honest
conclusion is "this design could not have detected an industry effect of plausible size," not
"there is no industry effect."** That's a materially different — and more defensible — claim
than the null on its own, and it's the correct way to read every CAR result in this notebook.
""")

md("## Comparative view: CAR + volatility + volume shock, by industry")

code("""
matrix_rows = []
for (industry, etf), row in car_summary.iterrows():
    ff_ind = [k for k, v in INDUSTRY_LABELS.items() if v == industry][0]
    dm = daily_metrics_df.loc[(industry, etf)]
    matrix_rows.append({
        "Industry": industry, "ETF_CAR": row["ETF_CAR"], "FF3_CAR": row["FF3_CAR"],
        "FF5_CAR": row["FF5_CAR"], "RealizedVol_30d": dm["RECENT_ETF_RealVol_30d_avg"],
        "VolumeShock": dm["RECENT_ETF_VolShock_avg"],
    })
industry_matrix = pd.DataFrame(matrix_rows).set_index("Industry")
plot_industry_heatmap(industry_matrix, "Industry Performance Matrix (Apr-Sep 2025)")
plt.savefig("../figures/industry_performance_heatmap.png", dpi=110, bbox_inches="tight")
plt.show()
industry_matrix.round(4)
""")

md("""
## Industry CAR timeline with tariff events

Illustrative overlay of the ETF CAR curves against the major 2025 tariff policy dates. This is a
narrative visualization, not itself a significance test — see the CAR summary table above and
`../CORRECTIONS.md` for which of these moves are statistically distinguishable from noise (none,
at conventional thresholds).
""")

code("""
events = [
    {"date": "2025-04-02", "label": "Reciprocal tariffs announced"},
    {"date": "2025-05-12", "label": "US-China 90-day truce"},
    {"date": "2025-06-04", "label": "Tariff-doubling effective"},
    {"date": "2025-07-31", "label": "EO modifying reciprocal tariffs"},
]

fig, ax = plt.subplots(figsize=(12, 6))
for (industry, etf), row in car_summary.iterrows():
    _, ar = etf_car_event(etf, BASE_START, BASE_END, RECENT_START, RECENT_END)
    ax.plot(ar.cumsum().index, ar.cumsum().values, label=industry)

for ev in events:
    d = pd.to_datetime(ev["date"])
    ax.axvline(d, color="red", linestyle="--", alpha=0.6)
    ax.text(d, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] else 0.05, ev["label"],
            rotation=90, color="red", va="top", fontsize=8)

ax.axhline(0, color="black", linestyle="--")
ax.set_title("Industry CAR Curves (ETF, daily) with Tariff Events (2025)")
ax.set_ylabel("Cumulative Abnormal Return")
ax.legend()
plt.savefig("../figures/industry_car_timeline.png", dpi=110, bbox_inches="tight")
plt.show()
""")

md("""
## Q4 2025 outlook (hybrid CAR + volatility + momentum)

Forward-looking, GARCH-volatility-based outlook built by the team from the corrected CAR and
volatility inputs above. This is a model-driven extrapolation, not a forecast guarantee — the
CAR inputs feeding it are not individually significant (see above), so read the ranking here as
indicative, not causal.
""")

code("""
q4_rows = []
for (industry, etf), row in car_summary.iterrows():
    dm = daily_metrics_df.loc[(industry, etf)]
    y_recent = _slice(ret[[etf]], RECENT_START, RECENT_END)[etf]
    slope_60d = y_recent.tail(60).mean() * 60 if len(y_recent) >= 60 else np.nan
    q4_rows.append({
        "Industry": industry,
        "AnnVol_Q4_Forecast": dm["RECENT_ETF_RealVol_30d_avg"] * np.sqrt(252) if pd.notna(dm["RECENT_ETF_RealVol_30d_avg"]) else np.nan,
        "CAR_Slope_60d": slope_60d,
        "GARCH_LB_p": dm["RECENT_GARCH_LB_p"],
    })
q4_outlook = pd.DataFrame(q4_rows).set_index("Industry").sort_values("CAR_Slope_60d", ascending=False)
q4_outlook.round(4)
""")

md("""
## Cross-sectional dispersion

Dispersion of FF30 monthly industry returns, baseline vs. event window — a measure of how much
industries diverged from each other (not from the market), regardless of direction.
""")

code("""
csd_base = FF30.loc[(FF30.index >= BASE_START) & (FF30.index <= BASE_END)].std(axis=1)
csd_recent = FF30.loc[(FF30.index >= RECENT_START) & (FF30.index <= RECENT_END)].std(axis=1)
print(f"Cross-sectional dispersion (avg monthly std across industries):")
print(f"  Baseline (2023-01 to 2025-03): {csd_base.mean():.4f}")
print(f"  Event window (Apr-Sep 2025):   {csd_recent.mean():.4f}")
""")

md("""
## Risk-adjusted CAR efficiency

Sharpe-style AR efficiency and CAR-per-unit-of-volatility, using the same single consolidated
`ff30_car_event` / market-model functions as above — no separate re-implementation.
""")

code("""
def sharpe_like(ar, freq):
    if ar is None or len(ar) < 3:
        return np.nan
    mu, sd = np.mean(ar), np.std(ar, ddof=1)
    return np.nan if (sd == 0 or np.isnan(sd)) else float(mu / sd * np.sqrt(freq))

def car_per_vol_t(ar, car):
    if ar is None or len(ar) < 3:
        return np.nan
    sd, n = np.std(ar, ddof=1), len(ar)
    return np.nan if (sd == 0 or np.isnan(sd)) else float(car / (sd * np.sqrt(n)))

efficiency_rows = []
for ff_ind, etf in FF30_TO_ETF.items():
    etf_car, etf_ar = etf_car_event(etf, BASE_START, BASE_END, RECENT_START, RECENT_END)
    ff3_car, ff3_ar = ff30_car_event(FF30, FF3M, FF5M, ff_ind, BASE_START, BASE_END,
                                      RECENT_START, RECENT_END, "FF3")
    ff5_car, ff5_ar = ff30_car_event(FF30, FF3M, FF5M, ff_ind, BASE_START, BASE_END,
                                      RECENT_START, RECENT_END, "FF5")
    efficiency_rows.append({
        "Industry": INDUSTRY_LABELS[ff_ind], "ETF": etf,
        "ETF_CAR": etf_car, "ETF_Sharpe_ann": sharpe_like(etf_ar, 252),
        "ETF_CAR_perVol_t": car_per_vol_t(etf_ar, etf_car), "ETF_N": len(etf_ar),
        "FF3_CAR": ff3_car, "FF3_Sharpe_ann": sharpe_like(ff3_ar, 12),
        "FF3_CAR_perVol_t": car_per_vol_t(ff3_ar, ff3_car), "FF3_N": len(ff3_ar),
        "FF5_CAR": ff5_car, "FF5_Sharpe_ann": sharpe_like(ff5_ar, 12),
        "FF5_CAR_perVol_t": car_per_vol_t(ff5_ar, ff5_car), "FF5_N": len(ff5_ar),
    })
efficiency_table = pd.DataFrame(efficiency_rows).set_index(["Industry", "ETF"]).round(4)
efficiency_table
""")

md("""
## Final integrated view
""")

code("""
final_dashboard = car_summary.reset_index()[["Industry", "ETF", "ETF_CAR", "FF3_CAR", "FF5_CAR"]].merge(
    q4_outlook.reset_index(), on="Industry"
).set_index("Industry")
final_dashboard.round(4)
""")

md("""
## Limitations of the analysis

1. **Industry classification mismatch.** FF30 portfolios are academic constructs (value-weighted,
   annually reformed) that report one observation per month; ETFs are market-tradable and update
   every trading day. Comparing a monthly series against a daily one introduces a measurement-
   granularity mismatch, reflected in the FF3/FF5 vs. ETF CAR disagreement above.
2. **No individual industry CAR is statistically significant** at conventional thresholds once the
   significance-test formula is corrected (Bug A) — bootstrap p-values sit around 0.5-0.9 across
   every industry and every model specification. The "most-affected industries" framing in the
   original write-up should be read as descriptive ranking of point estimates, not as a tested
   finding.
3. **This design is underpowered, not just null.** The FF30 monthly event window has only 6
   observations; the ETF daily window has ~125. The power analysis above quantifies exactly what
   that costs: MDEs of roughly 17-45% (ETF) and 9-31% (FF30, for five of six industries), against
   plausible tariff effects of single- to low-double-digit percent. This isn't a caveat on the
   null result — it's the reason the null result alone doesn't tell you whether a real (smaller)
   effect exists.
4. **Rerun caveat.** yfinance returns split/dividend-adjusted prices, so a rerun on a later date
   will not exactly reproduce these figures even with identical code — see the README for the
   pinned rerun date and what changed between runs.
""")

nb["cells"] = cells
with open("notebooks/report1_industry_tariff_impact.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote {len(cells)} cells to notebooks/report1_industry_tariff_impact.ipynb")
