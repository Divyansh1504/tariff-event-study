# Corrections

This repo is a corrected republish of one layer of a four-person graduate capstone (see the
[README](README.md) for full attribution). Five defects were found in the original analysis
during the republish. Each is documented here — what was originally claimed, what's actually
correct, and what conclusion changes — because a pipeline that gets checked and corrected in the
open is more useful than one presented as flawless. Nothing here has been quietly patched over;
every fix is visible in `src/` and reproducible from the notebooks in `notebooks/`.

All "corrected" figures below come from the live rerun pinned in the README (**2026-08-16**).
Where a number moved from the original December 2025 run for reasons *other than* the bug fix
itself (yfinance re-serves split/dividend-adjusted prices, so historical prices are not static),
that's called out explicitly rather than left ambiguous.

---

## Bug A — significance test formula (`src/regression.py`)

**Original code** (`test_car_significance`, Report 1 notebook):
```python
t_stat = car / (sigma / np.sqrt(len(ar_series)))
```
The standard error of a **sum** of N i.i.d. draws is `sigma * sqrt(N)`, not `sigma / sqrt(N)`
(the SE of a *mean*). Dividing instead of multiplying inflates every t-stat by roughly a factor
of N.

**Effect:** the original notebook reported ETF-CAR t-stats of 133.86 (Autos) and 295.02 (Steel)
as if these were near-certain effects (`p ≈ 0.0`), while the notebook's own bootstrap — computed
correctly the whole time — gave p ≈ 0.5–0.9 for the same data. The two tests silently
contradicted each other and the write-up cited both as support.

**Fix:** `t_stat = car / (sigma * np.sqrt(len(ar_series)))`, verified against a synthetic series
with known mean/sigma/N before being wired back in (`src/regression.py`, run as `__main__`) — a
null series now returns t≈0.66 (matching the bootstrap's own p≈0.5), and a series with a real,
large mean shift returns t≈15.9, p<0.001, confirming the formula is directionally sane.

**Conclusion that changes:** none of the individual-industry CARs in Report 1 or Report 2 that
relied on this test are statistically significant. Every "the market reacted to X" claim in the
original write-up that leaned on a t-stat needs to be read as a point estimate, not a tested
effect.

---

## Bug B — FF30 monthly CAR date window (`src/data_fetch.py::load_ff30_monthly`)

**Original claim:** industry CARs of +304%/+314% (Autos, FF3/FF5), +491%/+442% (Semis),
+197%/+173% (Utilities) over the six-month event window — physically impossible magnitudes for
monthly industry returns.

**Root cause (traced live against the Ken French source file):** the raw
`30_Industry_Portfolios.csv` stacks four separate tables that all share the identical `YYYYMM`
date format:

| Table | Content |
|---|---|
| Average Value Weighted Returns — Monthly | actual industry returns |
| Average Equal Weighted Returns — Monthly | actual industry returns |
| Number of Firms in Portfolios | integer firm counts |
| Average Firm Size | $ millions |

The original loader (`pd.read_csv(file_path, skiprows=11)`) had no stop condition and read
straight to end-of-file; the `^\d{6}$` regex used to keep "real" rows can't distinguish the four
tables, so it silently concatenated firm counts and firm sizes onto the return series and treated
all of it as returns (divided by 100). Confirmed live: this produced 16–24 rows feeding an event
window that should contain exactly 6 monthly observations (Apr–Sep 2025) — the regression was
partly fit on firm-size figures, not returns. Three near-duplicate CAR implementations existed in
the original notebook (cells 18, 23, 44); all three inherited the corrupted frame.

**Fix:** `load_ff30_monthly()` reads only the "Average Value Weighted Returns -- Monthly" table's
own line range, stopping at the next table's header rather than end-of-file. Verified live: the
Apr–Sep 2025 window now returns exactly 6 rows with normal single-digit-percent monthly values.
The three duplicate CAR functions are consolidated into one, `ff30_car_event()`.

**Corrected figures (FF3 / FF5 monthly CAR, 2026-08-16 rerun):**

| Industry | Original (bugged) | Corrected FF3 | Corrected FF5 |
|---|---|---|---|
| Autos | +304% / +314% | +0.12% | +6.81% |
| Steel | -1% / -6%\* | +0.61% | +13.24% |
| Semis | +491% / +442% | +7.33% | +8.86% |
| Health | +31% / +25%\* | -4.82% | -5.84% |
| Finance | +89% / +81%\* | -6.69% | -6.45% |
| Utilities | +197% / +173% | -0.52% | -1.37% |

\* Autos, Semis, and Utilities are the figures reported in the original write-up. Steel, Health,
and Finance were equally corrupted in the underlying notebook cell but weren't separately called
out in the original report text; shown here for completeness, computed from the same buggy
notebook cell output using the same (raw-number-as-percent) convention as the three reported
figures.

**Conclusion that changes — this is the blocking-check result:** Report 1's original ranking
(Autos, Semis, Steel as most tariff-affected) does not survive the fix. The corrected ranking
depends on which model is used:

| Method | Top 3 by \|CAR\| |
|---|---|
| FF30 + FF3 factors | Semis, Finance, Health |
| FF30 + FF5 factors | Steel, Semis, Autos (matches original) |
| ETF daily market-model | Steel, Autos, Health |

And under every specification, **no industry's CAR is statistically distinguishable from noise**
(bootstrap p ≈ 0.49–0.99 across the board, once Bug A's formula is also fixed) — so "most
tariff-affected industries" isn't a finding that holds up at all, in any version. Report 2's
15-stock basket was built on the original (bugged) ranking. Per the decision made when this was
found: **the basket is kept as-is**, but Report 2 is published as analysis of a *pre-selected*
tariff-exposed basket, not as a validated "most-affected industries" study — see the note at the
top of `notebooks/report2_portfolio_event_study.ipynb`.

---

## Bug C — alpha significance claim (Report 2)

**Original claim:** "statistically significant positive alpha" (notebook markdown and the
original Word write-up), against t-stats of 1.45 (CAPM-EW), 1.36 (CAPM-VW), 1.46 (FF6-EW), 1.16
(FF6-VW) — none of which exceed the 1.96 threshold for 95% confidence, regardless of which run's
numbers you use.

**Fix:** no formula was wrong here (unlike Bug A) — this was a narrative-language defect: the
write-up called an insignificant result significant. Corrected language: alpha was **positive
but statistically indistinguishable from zero**. With only ~126 trading days and portfolio betas
of 1.36–1.58, the sample lacks the power to detect an alpha of roughly 9 bps/day even if a real
effect exists. Outperformance is largely explained by market and factor exposure (beta > 1.3
throughout), not unexplained excess return.

**Rerun note — read this as fragility, not partial vindication:** the live 2026-08-16 rerun gives
t-stats of 1.76 / 1.92 / 2.00 / 1.88, versus the December 2025 run's 1.45 / 1.36 / 1.46 / 1.16.
The alpha-estimation code did not change between these two runs — the only thing that changed is
that `yfinance` re-served revised split/dividend-adjusted prices eight months later. That is the
entire cause of the shift from "clearly below 1.96" to "one of four specifications marginally
above it."

**This fragility and the power problem below are the same fact, not two separate observations.**
The minimum detectable effect (MDE) analysis (see the Addendum below) shows every observed alpha
sits at only 60-71% of its own MDE — meaning a design roughly 1.4-1.6x more powerful (≈1.5x on
average across the four specs) would have turned these exact same point estimates into a
"significant" result. A test running that close under its own detection threshold is, by
construction, exactly the kind of test where a routine data revision can nudge one specification
over the 1.96 line by chance — not because the estimate got more real, but because the test was
already sitting in the region where noise decides the outcome. **If FF6-EW's alpha were a real
effect and not noise operating below a detectable threshold, its significance shouldn't hinge on
which day the data happened to be pulled.** A stable null doesn't wobble across the 1.96 line on
a data refresh; an underpowered test running at ~1.5x under its own MDE does exactly that, and
did. Treat all four t-stats as "not distinguishable from zero with any confidence," full stop —
not as three misses and one near-hit. See the full discussion in
`notebooks/report2_portfolio_event_study.ipynb`.

---

## Bug D — annualized figures presented as returns earned

**Original claim:** cells labeled a CAGR calculation — `(1+x).prod()**(252/N)-1` over ~126
trading days — as **"CAGR (Q2–Q3 2025)"**: 96.0% (EW), 130.5% (VW), 42.4% (SPY). This is an
annualized *rate*, presented as if it were the return actually earned over the six-month window.

**Fix:** the results table now reports the actual six-month **cumulative return** as the headline
figure, with CAGR shown alongside it and explicitly labeled "annualized from a 6-month window."

**Corrected figures (2026-08-16 rerun):**

| Portfolio | Cumulative return (actual, 6 months) | CAGR (annualized) |
|---|---|---|
| Equal-Weighted | +39.25% | 95.99% |
| Value-Weighted | +50.83% | 130.53% |
| SPY | +18.99% | 42.38% |

**Conclusion that changes:** the basket still outperformed SPY over the period, but by roughly
20–32 percentage points of actual six-month return — not the 54–88-point gap the annualized
figures implied. The annualized numbers are mathematically consistent with the actual returns
(they're the same data, just extrapolated); the problem was purely how they were labeled.

---

## Bug E — Report 2 event-study CARs had no significance testing

**Original claim:** individual-day CARs around each of 10 tariff-policy dates were narrated as
causal market reactions ("investors rewarded trade normalization," etc.) with no statistical
test attached.

**Fix:** each event's ±5-day CAR is now tested with the corrected `test_car_significance()`.
Result (2026-08-16 rerun): **0 of 10 events** produce a t-stat exceeding 1.96 for either the
equal- or value-weighted portfolio. Every event-day narrative in
`notebooks/report2_portfolio_event_study.ipynb` is qualified accordingly — the CAR numbers are
reported as point estimates, not as confirmed reactions to the named policy event.

---

## Addendum — statistical power: the null needed this to be complete

Fixing Bug A (the significance-test formula) made it possible to test these effects correctly.
It didn't answer a second question that the null result alone leaves open: **could this design
have detected a real tariff effect if one existed?** Without answering that, "no significant
effect" is ambiguous between "there's genuinely nothing here" and "this design was too weak to
find anything short of an implausibly large effect." A power analysis (minimum detectable effect,
MDE, at 80% power) resolves that ambiguity. It is descriptive of the design already used, not a
new hypothesis test — every number below comes from a standard error or CAR/t-stat already
produced by a regression already fit in the notebooks; no new window, model, or event definition
was tried to produce it.

**Alpha regressions (Report 2, N=124):** every observed alpha (9.29-13.50 bps/day across the four
CAPM/FF6, EW/VW specifications) sits at only 60-71% of its own MDE (14.82-19.73 bps/day) — this is
the same ~1.5x-under-threshold gap discussed in Bug C above, which is exactly why FF6-EW's
significance flipped on nothing more than a routine data revision. Detecting the observed ~9
bps/day alpha at 80% power would have required roughly **443 trading days** — about 3.5× the 126
actually available.

**Report 1 industry CARs (6-month window):** MDE runs roughly 17-45% (ETF, daily) and 9-31%
(FF30, monthly, for five of six industries — Autos's near-zero t-stat makes the algebraic MDE
recovery numerically unstable there) — against observed CARs mostly in the single digits to low
30s%. Only Steel's ETF CAR (37.9%) approaches its ~45% MDE.

**Report 2 event-day CARs:** MDE ranges from 6.2%/7.8% (EW/VW, ±3 days) to 10.3%/13.0% (±10 days)
— against observed CARs mostly in the 0.6%-7.8% range, below MDE at every window width tested.

**What this changes:** every "no significant effect" claim in this document should be read
together with "and this design lacked the power to detect a plausible effect even if one
existed." That's not a weaker finding than a clean null — it's a more precise and more honest one.
A confident null from an adequately-powered design would be *stronger* evidence of "nothing here"
than what this analysis actually produced; the correct, load-bearing conclusion throughout this
repo is **"this design could not have detected a tariff effect of plausible size,"** not **"there
is no tariff effect."** The distinction matters for how a reader should weigh every other
correction in this document — none of them should be read as ruling out a real, smaller effect
that this design was never built to see. Full computation:
`scripts/power_analysis.py` and the "Statistical power" sections in both notebooks.

---

## Summary of what changes in the overall story

- The core "did tariff exposure show up in returns" question is **not answered affirmatively** by
  this analysis once the statistical tests are fixed — none of the industry- or event-level
  effects clear conventional significance thresholds. **Nor is it answered negatively with much
  confidence** — the power analysis above shows this design couldn't have detected a plausible
  effect even if one existed, so the correct reading is "undetected," not "ruled out."
- The 15-stock basket's outperformance over Apr–Sep 2025 is real (in the sense that the returns
  did happen) but is explained by market/factor beta exposure, not by tariff-specific alpha.
- The "most tariff-affected industries" ranking that motivated the stock selection does not
  survive the fix and is sensitive to model choice — treat the industry framing as context for
  the basket, not as a validated result.
- Report 3 (individual work, qualitative transcript analysis — see the README) found that 11 of
  12 tariff-exposed firms studied discussed tariff cost impact extensively, often with specific
  dollar figures. That's compatible with, not contradicted by, the quantitative null above: a
  design shown here to lack the power to detect even a large return effect is not evidence
  against a real, disclosed operational one.
