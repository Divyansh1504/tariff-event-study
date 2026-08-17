"""Post-hoc power / minimum-detectable-effect (MDE) analysis for the existing, already-reported
regressions and event-study CARs. This refits the SAME specifications already in the repo
(src/regression.py, unchanged) against the SAME saved data (data/report2_daily_returns.csv) purely
to extract standard errors with full precision -- it is not a new hypothesis test, and it does not
touch the return data with any new window, model, or event definition. For the Report 1 industry
CARs, MDE is derived algebraically from numbers already printed in the committed notebook
(sigma = CAR / (t * sqrt(N)), so MDE = z_combined * CAR / t) -- no data is re-fetched at all.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from scipy import stats

from data_fetch import load_ff6_daily
from regression import capm_alpha, ff6_alpha

Z_ALPHA = stats.norm.ppf(0.975)   # 1.959964 (two-sided, alpha=0.05)
Z_POWER = stats.norm.ppf(0.80)    # 0.841621 (power=0.80)
Z_COMBINED = Z_ALPHA + Z_POWER    # 2.801585

print(f"z(0.975) = {Z_ALPHA:.6f}, z(0.80) = {Z_POWER:.6f}, combined = {Z_COMBINED:.6f}")
print()

# ---------------------------------------------------------------------------
# 1. Alpha regressions (Report 2) -- refit identical CAPM/FF6 specs on saved data
#    to extract full-precision Newey-West standard errors.
# ---------------------------------------------------------------------------
daily = pd.read_csv("data/report2_daily_returns.csv", parse_dates=["Date"], index_col="Date")
ff6 = load_ff6_daily()
merged = daily.join(ff6, how="inner")
rf_daily = merged["RF"]

capm_eq = capm_alpha(merged["rp_eq"], merged["r_mkt"], rf_daily)
capm_val = capm_alpha(merged["rp_val"], merged["r_mkt"], rf_daily)
ff6_eq = ff6_alpha(merged["rp_eq"], merged, rf_daily)
ff6_val = ff6_alpha(merged["rp_val"], merged, rf_daily)

alpha_rows = []
for label, model in [("CAPM_EW", capm_eq), ("CAPM_VW", capm_val), ("FF6_EW", ff6_eq), ("FF6_VW", ff6_val)]:
    alpha_hat = model.params["const"]
    se = model.bse["const"]
    t = model.tvalues["const"]
    n = int(model.nobs)
    mde_daily = Z_COMBINED * se
    alpha_rows.append({
        "Spec": label, "N": n, "Alpha_bps_day": alpha_hat * 1e4, "t_stat": t,
        "SE_bps_day": se * 1e4, "MDE_bps_day": mde_daily * 1e4,
        "MDE_annualized_compounded": (1 + mde_daily) ** 252 - 1,
        "Observed/MDE_ratio": abs(alpha_hat) / mde_daily,
    })

alpha_power_df = pd.DataFrame(alpha_rows).set_index("Spec")
print("=== Alpha regression MDE (80% power, alpha=0.05, two-sided) ===")
print(alpha_power_df.round(4).to_string())
print()

# How many trading days would be needed to detect ~9 bps/day at 80% power?
# SE scales ~ 1/sqrt(N) for a fixed residual vol (Newey-West adjusts for autocorrelation but the
# N-scaling is still ~1/sqrt(N) to first order) -- solve N_required from the actual N=126 SE.
target_bps = 9.0
avg_se_bps = alpha_power_df["SE_bps_day"].mean()
avg_n = alpha_power_df["N"].mean()
# MDE(N) = Z_COMBINED * SE(126) * sqrt(126/N) = target  =>  N = 126 * (Z_COMBINED*SE(126)/target)^2
n_required = avg_n * (Z_COMBINED * avg_se_bps / target_bps) ** 2
print(f"Average SE across 4 specs: {avg_se_bps:.4f} bps/day (N={avg_n:.0f})")
print(f"Approx. trading days needed to detect a {target_bps} bps/day alpha at 80% power: {n_required:.0f}")
print(f"  ({n_required/252:.1f} years of trading days)")
print()

# ---------------------------------------------------------------------------
# 2. Report 1 industry CARs (6-month window) -- MDE derived algebraically from
#    CAR and t-stat already printed in the committed notebook. No new fetch.
#    sigma = CAR / (t * sqrt(N))  =>  MDE = Z_COMBINED * sigma * sqrt(N) = Z_COMBINED * CAR / t
# ---------------------------------------------------------------------------
report1_rows = [
    # Industry, ETF_CAR, ETF_N, ETF_t, FF3_CAR, FF3_N, FF3_t, FF5_CAR, FF5_N, FF5_t
    ("Autos",     0.1270, 125, 1.0709,  0.0012, 6, 0.0035,  0.0681, 6, 0.1996),
    ("Steel",     0.3794, 125, 2.3602,  0.0061, 6, 0.0684,  0.1324, 6, 1.2056),
    ("Semis",     0.0914, 125, 0.6856,  0.0733, 6, 1.1364,  0.0886, 6, 1.3073),
    ("Health",   -0.1242, 125, -1.1304, -0.0482, 6, -0.4849, -0.0584, 6, -0.5497),
    ("Finance",  -0.0741, 125, -1.2117, -0.0669, 6, -1.9824, -0.0645, 6, -1.2757),
    ("Utilities", 0.0322, 125, 0.3335, -0.0052, 6, -0.0835, -0.0137, 6, -0.1552),
]

print("=== Report 1 industry CAR MDE (6-month window, algebraic from published CAR/t) ===")
for ind, etf_car, etf_n, etf_t, ff3_car, ff3_n, ff3_t, ff5_car, ff5_n, ff5_t in report1_rows:
    etf_mde = Z_COMBINED * abs(etf_car / etf_t) if etf_t != 0 else np.nan
    print(f"{ind:10s} ETF(N={etf_n}) observed CAR={etf_car:+.4f} t={etf_t:+.4f}  -> MDE={etf_mde:.4f}")

# FF3/FF5 t-stats near zero (e.g. Autos FF3 t=0.0035) make CAR/t numerically unstable for those
# specific rows -- report a representative (median) sigma across industries instead for FF30.
etf_sigmas = [abs(c) / (abs(t) * np.sqrt(n)) for _, c, n, t, *_ in report1_rows]
ff3_sigmas = [abs(row[4]) / (abs(row[6]) * np.sqrt(row[5])) for row in report1_rows if abs(row[6]) > 0.05]
ff5_sigmas = [abs(row[7]) / (abs(row[9]) * np.sqrt(row[8])) for row in report1_rows if abs(row[9]) > 0.05]

med_etf_sigma, med_ff3_sigma, med_ff5_sigma = np.median(etf_sigmas), np.median(ff3_sigmas), np.median(ff5_sigmas)
print()
print(f"Median per-observation sigma -- ETF(daily,N=125): {med_etf_sigma:.5f}  "
      f"FF3(monthly,N=6): {med_ff3_sigma:.4f}  FF5(monthly,N=6): {med_ff5_sigma:.4f}")
print(f"Representative MDE -- ETF (6-month, N=125): {Z_COMBINED*med_etf_sigma*np.sqrt(125):.4f}")
print(f"Representative MDE -- FF30 FF3 (6-month, N=6):  {Z_COMBINED*med_ff3_sigma*np.sqrt(6):.4f}")
print(f"Representative MDE -- FF30 FF5 (6-month, N=6):  {Z_COMBINED*med_ff5_sigma*np.sqrt(6):.4f}")
print()

# ---------------------------------------------------------------------------
# 3. Report 2 event-day CARs (+/-3, +/-5, +/-10 day windows) -- daily AR vol from
#    the saved full-sample AR series (real data, no new fetch); N from the SAME
#    already-used event dates/window-mask logic (counting, not testing).
# ---------------------------------------------------------------------------
daily["AR_eq"] = daily["rp_eq"] - daily["r_mkt"]
daily["AR_val"] = daily["rp_val"] - daily["r_mkt"]
sigma_eq_daily = daily["AR_eq"].std(ddof=1)
sigma_val_daily = daily["AR_val"].std(ddof=1)
print(f"Full-sample daily AR std: EW={sigma_eq_daily:.5f}  VW={sigma_val_daily:.5f}")

event_dates = pd.to_datetime([
    "2025-04-02", "2025-04-09", "2025-05-08", "2025-05-12", "2025-05-14",
    "2025-05-30", "2025-06-04", "2025-07-07", "2025-07-31", "2025-09-04",
])

print()
print("=== Report 2 event-day CAR MDE (+/-3, +/-5, +/-10 day windows) ===")
for w in [3, 5, 10]:
    ns = []
    for event in event_dates:
        mask = (daily.index >= event - pd.Timedelta(days=w)) & (daily.index <= event + pd.Timedelta(days=w))
        ns.append(mask.sum())
    ns = [n for n in ns if n >= (w + 1)]  # drop boundary-truncated windows from the "typical N" count
    median_n = int(np.median(ns))
    mde_eq = Z_COMBINED * sigma_eq_daily * np.sqrt(median_n)
    mde_val = Z_COMBINED * sigma_val_daily * np.sqrt(median_n)
    print(f"+/-{w:>2}d  median N={median_n:2d} trading days  ->  MDE_EW={mde_eq:.4f}  MDE_VW={mde_val:.4f}")
