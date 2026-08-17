# Tariff Event Study

A policy event-study pipeline — applied here to the 2025 US tariff announcements — for measuring
whether an event shows up in industry and portfolio returns once you actually test for it,
instead of just eyeballing a chart.

## Headline finding: no significant effect detected

Run correctly, this analysis **does not detect a statistically significant industry-level effect
from the 2025 tariff announcements**, in any factor specification tested (FF3, FF5, or an ETF
market-model). Every industry's cumulative abnormal return is consistent with noise — bootstrap
p-values sit around 0.49–0.99 across the board, nowhere close to conventional significance.

The ranking of "most tariff-affected industries" — the input that determined which industries got
selected for the portfolio study in this repo — **does not survive a change of factor model**.
FF3 ranks Semis/Finance/Health highest; FF5 ranks Steel/Semis/Autos; an ETF market-model ranks
Steel/Autos/Health. Three specifications, three different top-3 lists, none of them backed by a
significant result. A ranking that reorders itself depending on which textbook factor model you
reach for isn't a finding — it's noise with a sort order.

That null result is the point of this repo, not a caveat buried in it. The deliverable here is
the **pipeline** — a reusable, correctly-tested event-study method — and the honest output of
running it on this event is "no detectable effect," not a confirmation of the tariff-impact
story the original (buggy) version of this analysis told. Five defects in that original version,
including the bugs that produced the appearance of an effect, are documented in
[CORRECTIONS.md](CORRECTIONS.md).

## The question, in plain language

Did the 2025 US tariff announcements (starting with the April "Liberation Day" reciprocal-tariff
order and running through several rounds of escalation, relief, and adjustment) produce a
measurable, statistically real effect on returns for the industries most exposed to them —
Autos, Steel, Semiconductors, Health, Finance, and Utilities — relative to what the broader
market and standard risk factors already explain?

**No** — not at any conventional significance threshold, and not in a way that's stable across
reasonable modeling choices. See [Findings](#findings) below for the full breakdown.

## What the pipeline does

1. **Fetch** daily ETF prices (Yahoo Finance) and monthly FF30 industry portfolios / Fama-French
   factor data (Ken French's data library) — `src/data_fetch.py`.
2. **Compute returns** — daily/monthly returns, cumulative return, CAGR (always labeled as
   annualized), volatility, Sharpe, and equal-/value-weighted portfolio construction —
   `src/returns.py`.
3. **Run the event study** — fit a market-model or Fama-French factor model on a baseline window,
   apply it out-of-sample to an event window, sum the resulting abnormal returns into a CAR, and
   test it for significance with both a parametric t-test and a bootstrap —
   `src/regression.py`.
4. **Plot** — CAR curves, cumulative-return charts, and industry heatmaps with consistent styling
   — `src/plotting.py`.

Two notebooks drive this against the 2025 tariff event:

- [`notebooks/report1_industry_tariff_impact.ipynb`](notebooks/report1_industry_tariff_impact.ipynb)
  — industry-level event study (FF30 + sector ETFs).
- [`notebooks/report2_portfolio_event_study.ipynb`](notebooks/report2_portfolio_event_study.ipynb)
  — a 15-stock tariff-exposed portfolio: alpha estimation (CAPM, FF6) and event-day CARs around
  10 specific tariff policy dates.

### Pointing it at a different event

The pipeline isn't tariff-specific. To reuse it for another event:

1. Pick tickers/ETFs representative of the industries or names you care about, and a benchmark
   (`fetch_prices` in `src/data_fetch.py` takes any ticker list).
2. Set a baseline window (before the event, long enough to estimate a stable factor model — 24+
   months is a reasonable floor) and an event window (the period you think the event affected).
3. Call `ff30_car_event(...)` (for FF30 industries) or replicate its market-model pattern for
   individual stocks/ETFs, then `test_car_significance(...)` on the resulting AR series.
4. Everything else — plotting, portfolio construction, alpha estimation — is event-agnostic.

## Data sources and date ranges

- **Prices:** Yahoo Finance via `yfinance` (daily, split/dividend-adjusted).
- **Factors:** Ken French Data Library — FF3, FF5, FF6 (FF5 + Momentum), and FF30 industry
  portfolios (monthly).
- **Event window:** 2025-04-01 to 2025-09-30 (six months spanning the initial tariff announcement
  and subsequent escalation/relief/adjustment actions).
- **Baseline window:** 2023-01-01 to 2025-09-30 (used for fitting factor models before applying
  them out-of-sample to the event window; the exact baseline end date varies by notebook cell —
  see the notebook for specifics).

**Rerun date:** the notebooks in this repo were last executed **2026-08-16**. `yfinance` serves
split/dividend-adjusted prices that are revised over time, so re-running the notebooks on a later
date will not exactly reproduce the numbers here even with unchanged code — this is expected, not
a bug. Where a number in [CORRECTIONS.md](CORRECTIONS.md) differs from what the original capstone
reported, it's explicitly labeled as either the bug fix or a data revision since the original
December 2025 run.

If Yahoo Finance or the Ken French data server blocks or rate-limits a request during rerun, the
notebook will fail loudly at that cell rather than produce silently fabricated output.

## Methodology, for a non-quant reader

An **event study** asks: "did returns around this event look different from what you'd expect
anyway, given how this stock/industry normally moves with the market?" Concretely:

1. Fit a model (market-model beta, or a multi-factor model like Fama-French) on a *baseline*
   period, before the event — this captures how the industry normally behaves.
2. Apply that model to the *event* period to get an "expected" return each day/month.
3. **Abnormal return (AR)** = actual return − expected return. Summed across the event window,
   that's the **cumulative abnormal return (CAR)** — the part of the move the baseline
   relationship doesn't explain.
4. A CAR of, say, +5% doesn't mean much on its own — it could easily be noise. The **significance
   test** asks: given how much this series normally jumps around (its volatility), is a CAR this
   large a genuine outlier, or well within the range you'd see by chance? This repo reports both
   a standard t-test and a bootstrap (resampling the actual data many times to build an empirical
   distribution) — when the two disagree, that's a signal something's wrong with one of them (see
   Bug A in CORRECTIONS.md, where they used to disagree for exactly that reason).

## Findings

*(Full detail, including every original-vs-corrected number, is in
[CORRECTIONS.md](CORRECTIONS.md).)*

- **No industry's CAR is statistically distinguishable from noise**, in any factor specification
  (FF3, FF5, or ETF market-model) — bootstrap p-values sit around 0.49–0.99 across the board.
- **The "most tariff-affected industries" ranking is model-dependent.** FF3 ranks Semis/Finance/
  Health highest by CAR magnitude; FF5 ranks Steel/Semis/Autos (the original notebook's claimed
  ranking); the ETF market-model ranks Steel/Autos/Health. None of these rankings are backed by a
  significant result, so none should be read as a confirmed finding.
- **The 15-stock tariff-exposed portfolio (Autos/Semis/Steel, 5 tickers each) did outperform SPY**
  over the six-month event window — +39.3% (equal-weighted) and +50.8% (value-weighted) actual
  cumulative return vs. +19.0% for SPY. Alpha (CAPM and FF6, Newey-West HAC standard errors) is
  not reliably distinguishable from zero: t-stats sit in the 1.16–2.00 range across four
  specifications and two data pulls, straddling the 1.96 threshold rather than clearing it. A
  result that flips significance depending on eight months of price-adjustment revisions was
  never robust — see [CORRECTIONS.md](CORRECTIONS.md#bug-c--alpha-significance-claim-report-2)
  for why that instability counts as evidence *against* real alpha, not for it. Either way, the
  outperformance is substantially
  explained by the portfolio's higher market beta (1.36–1.58), not by tariff-specific skill.
- **None of the 10 individual tariff-event CARs** (±5-day windows around dates like the April 2
  "Liberation Day" announcement or the May 12 US-China truce) clear a 95% significance threshold.
  They're reported as point estimates in the notebook, not as confirmed market reactions to those
  specific events.

## Repo structure

```
src/
  data_fetch.py     # yfinance price/volume fetch; Ken French FF3/FF5/FF6/FF30 loaders
  returns.py         # returns, cumulative return, CAGR (explicitly labeled), vol, Sharpe, EW/VW
  regression.py       # Newey-West CAPM/FF6 alpha; market-model AR; significance test; FF30 CAR
  plotting.py         # shared CAR-curve / cumulative-return / heatmap plotting
notebooks/
  report1_industry_tariff_impact.ipynb   # industry-level event study
  report2_portfolio_event_study.ipynb    # 15-stock portfolio, alpha, event-day CARs
reports/
  report3_transcript_analysis.pdf                 # earnings-call transcript analysis (individual)
  report3_tariff_impact_assessment_matrix.pdf      # supporting assessment matrix (individual)
figures/              # charts exported from the notebooks
data/                 # gitignored; populated by running the notebooks
CORRECTIONS.md         # every defect found: original claim, corrected number, what changes
requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
python -m ipykernel install --user --name tariff-event-study
```

## Reproduce

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/report1_industry_tariff_impact.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/report2_portfolio_event_study.ipynb
```

Both notebooks fetch data live (Yahoo Finance + Ken French) on each run and write intermediate
CSVs to `data/` (gitignored). Expect a few minutes per notebook, most of it spent on network
fetches.

## Attribution

Built as part of a four-person graduate capstone: **Divyansh Sharma, Geethanjali, Jui, and
Siddharth**. Divyansh owned the data pipeline and quantitative analysis — the layer published in
this repo (`src/`, both notebooks, and Report 3). Framing, business recommendations, and the two
narrative capstone reports (Report 1 and Report 2 write-ups) were team work and are not
republished here; this repo is the underlying analysis engine those reports were built on, with
its defects fixed and documented. Report 3 (earnings-call transcript analysis) was individual
work and is included as a PDF in `reports/`.

See [CORRECTIONS.md](CORRECTIONS.md) for the full defect log and what it changes about the
findings.
