# Tariff Event Study

An event-study pipeline for testing whether a policy event actually shows up in industry and portfolio returns. Applied here to the 2025 US tariff announcements.

## Headline finding

Run correctly, this analysis finds no statistically significant industry-level effect from the 2025 tariff announcements. Bootstrap p-values for industry CARs sit between 0.49 and 0.99 in every specification tested (FF3, FF5, and an ETF market model).

The ranking of "most affected industries" also depends on which factor model you pick. FF3 puts Semis, Finance, and Health on top. FF5 gives Steel, Semis, Autos. The ETF market model gives Steel, Autos, Health. Three specifications, three different top-three lists, none of them significant.

But a null result raises a second question: could this design have found an effect if one existed? A power analysis says no. Every observed alpha estimate sits at 60 to 71 percent of the size it would need to be reliably detectable. Detecting the observed 9 bps/day alpha at 80 percent power would take roughly 443 trading days. The six-month window provided 126.

So the accurate conclusion is that this design could not have detected a tariff effect of plausible size, which is a weaker claim than saying there was no effect. The deliverable is the pipeline, and the honest output of running it on this event is a null with a known reason behind it.

Five defects in the original version of this analysis, including the ones that produced the appearance of an effect, are documented in [CORRECTIONS.md](CORRECTIONS.md).

## The question

Did the 2025 US tariff announcements, starting with the April reciprocal-tariff order and running through several rounds of escalation and relief, move returns for the most exposed industries (Autos, Steel, Semiconductors, Health, Finance, Utilities) beyond what the market and standard risk factors already explain?

No detectable effect, at any conventional threshold, and not stably across reasonable modeling choices.

## What the pipeline does

1. **Fetch.** Daily ETF prices from Yahoo Finance, monthly FF30 industry portfolios and Fama-French factors from Ken French's data library. See `src/data_fetch.py`.
2. **Compute returns.** Daily and monthly returns, cumulative return, CAGR (always labeled as annualized), volatility, Sharpe, and equal- or value-weighted portfolio construction. See `src/returns.py`.
3. **Run the event study.** Fit a market model or factor model on a baseline window, apply it out of sample to the event window, sum the abnormal returns into a CAR, and test it with both a t-test and a bootstrap. See `src/regression.py`.
4. **Plot.** CAR curves, cumulative return charts, industry heatmaps. See `src/plotting.py`.

Two notebooks run this against the 2025 tariff event:

- [`report1_industry_tariff_impact.ipynb`](notebooks/report1_industry_tariff_impact.ipynb), industry-level event study using FF30 and sector ETFs.
- [`report2_portfolio_event_study.ipynb`](notebooks/report2_portfolio_event_study.ipynb), a 15-stock tariff-exposed portfolio with alpha estimation (CAPM, FF6) and event-day CARs around 10 policy dates.

### Using it for a different event

Nothing here is tariff-specific.

1. Pick tickers or ETFs for the industries you care about, plus a benchmark. `fetch_prices` takes any ticker list.
2. Set a baseline window before the event (24 months or more is a reasonable floor for a stable factor fit) and an event window.
3. Call `ff30_car_event(...)` for FF30 industries, or copy its market-model pattern for individual names, then run `test_car_significance(...)` on the resulting AR series.
4. Plotting, portfolio construction, and alpha estimation are all event-agnostic.

## Data and date ranges

**Prices:** Yahoo Finance via `yfinance`, daily, split and dividend adjusted.
**Factors:** Ken French Data Library. FF3, FF5, FF6 (FF5 plus momentum), and FF30 industry portfolios, monthly.
**Event window:** 2025-04-01 to 2025-09-30.
**Baseline window:** 2023-01-01 onward, used to fit factor models before applying them out of sample. The exact baseline end date varies by notebook cell.

**Rerun date: 2026-08-16.** `yfinance` serves adjusted prices that get revised over time, so re-running later will not reproduce these numbers exactly even with unchanged code. That is expected. Where a number in CORRECTIONS.md differs from the original capstone, it is labeled as either a bug fix or a data revision.

If Yahoo Finance or the Ken French server rate-limits a request during a rerun, the notebook fails at that cell rather than producing fabricated output.

## How an event study works

The question an event study asks is whether returns around an event looked different from what you would expect anyway, given how the stock or industry normally moves with the market.

1. Fit a model on a baseline period before the event. This captures normal behavior.
2. Apply that model to the event period to get an expected return for each day or month.
3. Abnormal return is actual minus expected. Summed across the event window, that is the cumulative abnormal return, or CAR.
4. A CAR of 5 percent means little by itself. The significance test asks whether a move that size is a real outlier given how much the series normally jumps around, or well within the range you would see by chance.

This repo reports both a t-test and a bootstrap. When the two disagree, something is wrong with one of them. That is how Bug A was found.

## Statistical power

A null answers whether an effect was detected. It does not answer whether the design could have detected one. This section is descriptive of the design already used, not a new test. Every number below comes from a regression already fit in the notebooks, or is derived algebraically from a CAR and t-stat already printed there. No new window, model, industry definition, or event date was tried. See `minimum_detectable_effect` and `car_mde` in `src/regression.py`.

Minimum detectable effect (MDE) is the smallest true effect a test could reliably distinguish from zero at 80 percent power, two-sided, alpha 0.05.

### Alpha regressions, Report 2

| Spec | Observed alpha (bps/day) | t-stat | MDE (bps/day) | Observed / MDE |
|---|---|---|---|---|
| CAPM EW | 9.29 | 1.76 | 14.82 | 0.63 |
| CAPM VW | 13.50 | 1.92 | 19.73 | 0.68 |
| FF6 EW | 10.75 | 2.00 | 15.04 | 0.71 |
| FF6 VW | 12.42 | 1.88 | 18.47 | 0.67 |

Every observed alpha lands at 60 to 71 percent of its own MDE. A design about 1.5x more powerful would have turned these same point estimates into significant results.

That gap also explains the alpha fragility noted in CORRECTIONS.md. A test running that close under its detection threshold is exactly where a routine data revision can push one specification across 1.96 by chance. The t-stat instability and the MDE shortfall are the same fact seen two ways.

Detecting the observed 9 bps/day alpha at 80 percent power would take roughly 443 trading days, about 1.8 years. The sample was 126 days.

### Industry CARs, Report 1

MDE runs roughly 17 to 45 percent for the daily ETF specification and 9 to 31 percent for FF30 monthly, across five of six industries. Autos is numerically degenerate here because its t-stat is near zero; see the notebook.

Observed CARs are mostly single digits to low 30s. Only Steel's ETF CAR (37.9 percent) comes close to its 45 percent MDE.

### Event-day CARs, Report 2

This is the clearest case. MDE ranges from 6.2 percent (EW) and 7.8 percent (VW) at a ±3 day window up to 10.3 and 13.0 percent at ±10 days. The reported CARs use the ±5 day window, where MDE is 7.3 percent EW and 9.2 percent VW.

All 10 events fall below that line. The largest observed EW move is 4.6 percent and the largest VW move is 7.8 percent, both under the MDE for the window they were measured on. None of the 10 reported CARs could have registered as significant regardless of the true underlying size, which is why 0 of 10 cleared 1.96.

### Verdict

Underpowered at every layer. Observed effects would need to be roughly 1.4x to 4x larger to be reliably detectable. If the design had been adequately powered and still come back null, that would be a stronger result than what this analysis produced.

## Findings

Full original-versus-corrected numbers are in [CORRECTIONS.md](CORRECTIONS.md).

**No industry CAR is distinguishable from noise** in any specification. Bootstrap p-values run 0.49 to 0.99.

**The affected-industry ranking is model-dependent.** FF3 gives Semis, Finance, Health. FF5 gives Steel, Semis, Autos, which is what the original notebook claimed. The ETF market model gives Steel, Autos, Health. None are backed by a significant result.

**The 15-stock portfolio did outperform SPY** over the event window: 39.3 percent equal-weighted and 50.8 percent value-weighted cumulative return against 19.0 percent for SPY. Alpha is not reliably distinguishable from zero, with t-stats between 1.16 and 2.00 across four specifications and two data pulls, straddling 1.96 rather than clearing it. Most of the outperformance is explained by the portfolio's higher market beta, 1.36 to 1.58, rather than anything tariff-specific.

**None of the 10 individual event CARs** clear a 95 percent threshold. They are reported as point estimates, not as confirmed reactions to those dates.

**The design lacked power to detect a plausible effect** at every layer. See the power section above.

## Report 3: transcript analysis

Report 3 is separate, individual work. It covers earnings-call transcripts and disclosures from 12 tariff-exposed firms across four sectors (automotive OEMs, auto suppliers, industrial machinery, and large retailers and apparel), using a structured question framework covering cost and margin pressure, sourcing, inventory, pricing pass-through, and forward guidance, cross-checked against news coverage and press releases. Full write-up in [`reports/report3_transcript_analysis.pdf`](reports/report3_transcript_analysis.pdf).

This part is descriptive. No significance testing applies to a qualitative reading of what companies said, and none is claimed. It was done through manual reading and LLM-assisted extraction and topic modeling directly against transcript text, not through a script preserved in this repo. There is no `src/` module behind it.

**Tariffs are a quantified cost driver for 11 of 12 firms.** Ford disclosed a $2B net tariff bill. GM told press it expects $4B to $5B for the year. Stellantis built a €1.5B net tariff expense into guidance. Caterpillar and Deere both embed quantified tariff drag in margin bridges, with Deere at $500M or more. Tesla is the outlier: tariffs appear only inside broader geopolitical risk language, never quantified.

**LDA topic modeling on analyst question language** produced five topics: sales momentum, general business conditions, tariff and cost structure, growth outlook, and pricing strategy. Term frequency is dominated by margin, tariff, growth, and cost. Tariff questions sit inside normal margin and demand discussion rather than forming a separate line of inquiry, which suggests analysts treat tariffs as an input to profitability models rather than as a standalone issue.

**Sectors differ in framing, not just magnitude.** Automotive OEMs and suppliers plus the industrial names describe tariffs as a recurring structural cost, addressed through footprint shifts, sourcing diversification, and OEM cost recoveries. Retailers frame them as timing-driven and transitory. Target explicitly called most of its tariff costs one-time, tied to order cancellations concentrated in Q2.

**On whether emphasis shifted across quarters:** the analysis is a cross-sectional snapshot of Q2 and Q3 2025, not a formal trend study, so it cannot answer that. A few firms gesture at forward normalization individually, but those are firm-specific statements rather than a cross-firm time trend.

**One observation, not a causal claim.** Management at 11 of 12 firms discussed tariff impact extensively and often in dollar terms, while the quantitative analysis in this same repo finds no significant market reaction over the same window. These are not necessarily in tension. The power analysis shows the return-based design was not sensitive enough to detect even a large effect, so real operational impact is compatible with no detectable move in returns. This repo cannot distinguish between "the market priced it in already," "the effect was real but too small for this design," and "disclosure volume does not translate into a return-moving surprise." That would need a different study.

**Attribution note:** the PDF cover page carries the shared capstone template listing all four names, a holdover from the group deliverable format. The transcript analysis, topic modeling, question framework, and write-up in Report 3 were done solely by Divyansh.

## Repo structure

```
src/
  data_fetch.py     yfinance price fetch, Ken French FF3/FF5/FF6/FF30 loaders
  returns.py        returns, cumulative return, CAGR, vol, Sharpe, EW/VW construction
  regression.py     Newey-West CAPM/FF6 alpha, market-model AR, significance test, MDE
  plotting.py       CAR curves, cumulative return charts, heatmaps
notebooks/
  report1_industry_tariff_impact.ipynb
  report2_portfolio_event_study.ipynb
reports/
  report3_transcript_analysis.pdf
  report3_tariff_impact_assessment_matrix.pdf
figures/            charts exported from the notebooks
data/               gitignored, populated by running the notebooks
CORRECTIONS.md      every defect found, with original and corrected numbers
requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name tariff-event-study
```

## Reproduce

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/report1_industry_tariff_impact.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/report2_portfolio_event_study.ipynb
```

Both notebooks fetch data live on each run and write intermediate CSVs to `data/`, which is gitignored. Expect a few minutes each, mostly network time.

## Attribution

Built as part of a four-person graduate capstone: Divyansh Sharma, Geethanjali, Jui, and Siddharth.

Divyansh owned the data pipeline and quantitative analysis, which is the layer published here: `src/`, both notebooks, and Report 3. Framing, business recommendations, and the two narrative capstone reports were team work and are not republished. This repo is the analysis engine those reports were built on, with its defects fixed and documented.

Report 3 was individual work and is included as a PDF in `reports/`. Its cover page carries the shared capstone template, but the analysis and write-up were done solely by Divyansh.

See [CORRECTIONS.md](CORRECTIONS.md) for the full defect log.
