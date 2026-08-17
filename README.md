# Tariff Event Study

A policy event-study pipeline — applied here to the 2025 US tariff announcements — for measuring
whether an event shows up in industry and portfolio returns once you actually test for it,
instead of just eyeballing a chart.

## Headline finding: no significant effect — and this design couldn't have found one

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

**A power analysis (see [Statistical power](#statistical-power-was-this-design-capable-of-detecting-an-effect)
below) shows this isn't just a null result — the design lacked the ability to detect a plausible
effect in the first place.** Every observed alpha estimate sits at only 60-71% of the size it
would need to be reliably detectable at 80% power; detecting the observed ~9 bps/day alpha at 80%
power would have required roughly 443 trading days, against the 126 this six-month window
actually provided. Industry-level CARs would have needed to be roughly 20-45% over six months to
be reliably detectable — an order of magnitude larger than anything plausible for a tariff effect,
or than what was actually observed. **The precise, honest conclusion is: "this design could not
have detected a tariff effect of plausible size," not "there is no tariff effect."** Those are
different claims, and the gap between them is the real headline here alongside the null itself —
not a caveat, a load-bearing part of the result.

That combination — a corrected null, and a power analysis showing the design was never capable of
resolving anything smaller than an implausibly large effect — is the point of this repo, not a
caveat buried in it. The deliverable here is the **pipeline** — a reusable, correctly-tested,
power-aware event-study method — and the honest output of running it on this event is "no
detectable effect, and this design wasn't equipped to detect a plausible one anyway," not a
confirmation of the tariff-impact story the original (buggy) version of this analysis told. Five
defects in that original version, including the bugs that produced the appearance of an effect,
are documented in [CORRECTIONS.md](CORRECTIONS.md).

## The question, in plain language

Did the 2025 US tariff announcements (starting with the April "Liberation Day" reciprocal-tariff
order and running through several rounds of escalation, relief, and adjustment) produce a
measurable, statistically real effect on returns for the industries most exposed to them —
Autos, Steel, Semiconductors, Health, Finance, and Utilities — relative to what the broader
market and standard risk factors already explain?

**No detectable effect — and, separately, this design wasn't powerful enough to have detected a
plausible one even if it existed.** Not at any conventional significance threshold, not in a way
that's stable across reasonable modeling choices, and not with enough statistical power to rule
out a real but modest effect. See [Findings](#findings) and
[Statistical power](#statistical-power-was-this-design-capable-of-detecting-an-effect) below for
the full breakdown.

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

## Statistical power: was this design capable of detecting an effect?

A null result answers "was an effect detected?" It does not answer "could this design have
detected one if it existed?" Conflating the two overstates the null — a test that couldn't have
found a real effect either way tells you much less than a well-powered test that came back empty.
This section is **descriptive of the design already used**, not a new test: every number below is
either extracted directly from a regression already fit in the notebooks, or derived
algebraically from a CAR and t-stat already printed in them. No new window, model, industry
definition, or event date was tried in producing it — see `src/regression.py::minimum_detectable_effect`
and `car_mde`.

**Minimum detectable effect (MDE)** is the smallest true effect size a test could reliably (80%
power, two-sided, α=0.05) distinguish from zero, given how noisy the estimator already is.

**Alpha regressions (Report 2, N=124 trading days):**

| Spec | Observed α (bps/day) | t-stat | MDE (bps/day) | Observed/MDE |
|---|---|---|---|---|
| CAPM EW | 9.29 | 1.76 | 14.82 | 0.63 |
| CAPM VW | 13.50 | 1.92 | 19.73 | 0.68 |
| FF6 EW | 10.75 | 2.00 | 15.04 | 0.71 |
| FF6 VW | 12.42 | 1.88 | 18.47 | 0.67 |

Every observed alpha is only 60-71% of its own MDE — a design roughly 1.4-1.6x more powerful
(≈1.5x on average) would have turned these same point estimates into "significant" results. That
gap is also the mechanism behind the alpha fragility noted above: a test running that close under
its own detection threshold is exactly where a routine data revision can flip one specification
across the 1.96 line by chance (see [CORRECTIONS.md](CORRECTIONS.md#bug-c--alpha-significance-claim-report-2)
for the full argument) — the t-stat instability and this MDE shortfall are the same underlying
fact, not two separate findings. **Trading days needed to detect the observed ~9 bps/day alpha at
80% power: ≈443 (≈1.8 years)** — the actual sample was 126 days, roughly 3.5× too short.

**Report 1 industry CARs (6-month window):** MDE runs roughly 17-45% (ETF, daily, N=125) and
9-31% (FF30, monthly, N=6, for five of six industries — Autos is a numerically degenerate case
with a near-zero t-stat; see the notebook) — against observed CARs mostly in the single digits to
low 30s%. Only Steel's ETF CAR (37.9% observed) comes close to its ~45% MDE; nothing else is
close.

**Report 2 event-day CARs — the clearest illustration of the power problem in this repo:** MDE
ranges from 6.2%/7.8% (EW/VW, ±3 days) to 10.3%/13.0% (±10 days). The actual event-day CARs
reported use the ±5-day window (MDE 7.3% EW / 9.2% VW) — and *every single one* of the 10 events
falls below that line: the largest observed EW move is +4.6%, the largest VW move is +7.8%, both
smaller than the MDE for the window they were measured on. This isn't "most CARs happened to be
small" — it's that **none of the 10 reported CARs could have registered as significant no matter
their true underlying size**, which is exactly why 0 of 10 cleared 1.96. The design didn't fail to
find an effect at these specific events; it wasn't built to find one at this window width and
sample size regardless of what was actually happening.

**Verdict: underpowered, consistently, at every layer of this design.** Not marginally — observed
effects would need to be roughly 1.4-4x larger to be reliably detectable at 80% power. This is
why the null and the power finding are stated together as the headline, not as a result plus an
asterisk: **"no effect was detected" is compatible with, and largely explained by, "this design
could not have detected a real effect of plausible size."** If the design had been adequately
powered and still came back null, that would be a stronger, more informative result than what
this analysis actually produced — power-checking the design is what makes that distinction
visible instead of asserting a confident-sounding null on a study that was never built to resolve
one.

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
  outperformance is substantially explained by the portfolio's higher market beta (1.36–1.58),
  not by tariff-specific skill.
- **None of the 10 individual tariff-event CARs** (±5-day windows around dates like the April 2
  "Liberation Day" announcement or the May 12 US-China truce) clear a 95% significance threshold.
  They're reported as point estimates in the notebook, not as confirmed market reactions to those
  specific events.
- **This design lacked the statistical power to detect a plausible effect at every layer** —
  see [Statistical power](#statistical-power-was-this-design-capable-of-detecting-an-effect)
  above for the numbers behind that claim.

## Report 3 findings: what the transcript analysis found (descriptive)

Report 3 is separate, individual work: qualitative LDA topic modeling and LLM-assisted analysis
of earnings-call transcripts and disclosures from 12 tariff-exposed firms across four sectors
(automotive OEMs, auto suppliers, industrial machinery, and large retailers/apparel), built
around a structured question framework covering cost/margin pressure, sourcing, inventory,
pricing pass-through, and forward guidance, then cross-validated against news coverage, press
releases, and analyst commentary. Full write-up:
[`reports/report3_transcript_analysis.pdf`](reports/report3_transcript_analysis.pdf).

**This is descriptive, not inferential.** No significance testing applies to a qualitative reading
of what companies said — it isn't dressed up as one here. Treat everything below as "what the
disclosures showed," not as a tested hypothesis.

**Methodology note:** this analysis was done via manual reading and LLM-assisted extraction and
topic modeling (LDA) directly against transcript text, not through a script preserved in this
repo. There is no `src/` module or notebook backing it — saying so here rather than implying a
coded pipeline exists for it.

**What it found:**

- **Tariffs are a consistent, often precisely quantified cost driver across 11 of the 12 firms.**
  Ford disclosed an explicit $2B net tariff bill; GM told press it expects $4-5B for the year;
  Stellantis baked a €1.5B net tariff expense into guidance; Caterpillar and Deere both embed
  quantified tariff drag into margin bridges (Deere: $500M+). The one outlier is Tesla, where
  tariffs surface only inside broader geopolitical-risk language, never quantified — a real
  difference in disclosure posture, not just a data gap.
- **LDA topic modeling on analyst question language** (5 topics extracted from earnings-call
  Q&A) clusters into sales momentum, general business conditions, tariff/cost structure, growth
  outlook, and pricing strategy — with "margin," "tariff," "growth," and "cost" dominating the
  term frequency. The consistent pattern: tariff questions are embedded inside normal margin/
  demand/pricing discussion, not treated as an isolated topic — analysts ask about tariffs as
  part of how they model profitability, not as a separate line of inquiry.
- **Sector differences in framing, not just magnitude.** Automotive OEMs and suppliers
  (GM, Ford, Stellantis, BorgWarner, Magna, Caterpillar, Deere) describe tariffs as a **recurring,
  structural** cost addressed through footprint shifts, sourcing diversification, and OEM cost
  recoveries. Retailers (Walmart, Target) more often frame tariff costs as **timing-driven and
  transitory** — Target explicitly called most of its tariff costs "one-time" tied to order
  cancellations concentrated in Q2, expecting normalization going forward.
- **On whether disclosure emphasis shifted across quarters:** the underlying analysis is a
  cross-sectional snapshot of Q2-Q3 2025 disclosures, not a formal quarter-over-quarter trend
  study — that's a limitation worth stating plainly rather than overclaiming a longitudinal
  finding the report didn't set out to produce. A few firms individually gesture at forward
  normalization (Target's "not significant going forward"; Caterpillar's uncertainty over whether
  Q4 drag carries into 1H-2026), but these are firm-specific forward statements, not a systematic
  cross-firm time trend.

**An observation worth stating, not a causal claim:** management at 11 of 12 firms discussed
tariff impact extensively and often in specific dollar terms, and analysts consistently pressed
on tariff-driven margin and cost questions across every sector covered — while the quantitative
analysis in this same repo (Reports 1-2) detects no statistically significant industry- or
portfolio-level market reaction over the same window. These aren't necessarily in tension: the
power analysis above shows this repo's return-based design wasn't sensitive enough to detect even
a large effect, so extensive, real, dollar-quantified operational impact is entirely compatible
with "no detectable move in industry or portfolio returns" — the absence of a detected market
reaction doesn't imply the absence of a real operational one, and the presence of extensive
disclosure doesn't imply the market necessarily re-priced anything in response. Both readings are
consistent with the evidence; this repo doesn't have a design capable of adjudicating between
"the market already priced it in," "the effect was real but too small for this design to catch,"
and "disclosure volume doesn't translate to a return-moving surprise" — that would need a
different, more targeted study.

**Attribution note:** the PDF's cover page carries the shared capstone title template used across
all three reports (all four names), a holdover from the group deliverable format — the transcript
analysis, topic modeling, question framework, and write-up in Report 3 were done solely by
Divyansh. See [Attribution](#attribution) below for the full breakdown.

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
  report3_transcript_analysis.pdf                 # earnings-call transcript analysis (individual —
                                                    #   cover page carries the shared team template,
                                                    #   see Attribution)
  report3_tariff_impact_assessment_matrix.pdf      # supporting assessment matrix (individual, same note)
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
work by Divyansh and is included as a PDF in `reports/` — its cover page carries the shared
capstone title template used across all three deliverables, but the topic modeling, question
framework, and write-up were done solely by Divyansh, not the full team.

See [CORRECTIONS.md](CORRECTIONS.md) for the full defect log and what it changes about the
findings.
