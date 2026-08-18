# flow — ownership and positioning. Agent F.

Component 2 of four, attacked from the ownership side. Agent A attacks the same
component from the price side, so the number that decides whether both are worth
carrying is the ρ_b between the blocks, not either standalone ρ. That number is
at the bottom and it is the good news in this report.

## Verdict in one line

**Every feature fails the ρ ≥ 0.15 bar by a factor of three or more.** The best
is `flow_combined` at ρ 0.054 on the 4,144-event confirmation partition, against
a standard error of 0.0155. The signal is real (3.5 se) and it is nearly
orthogonal to both the champion and the price block — and it is still an eighth
of the bar in ρ², so it buys almost nothing on its own.

## Sources

| source | obtainable | cost | auth | point-in-time |
| --- | --- | --- | --- | --- |
| FINRA consolidated short interest (`api.finra.org`) | yes | free | none | clean, via a **derived** publication date |
| FINRA daily short-sale volume (`cdn.finra.org/equity/regsho/daily`) | yes | free | none | clean |
| EDGAR Form 4 | yes | free | UA header | clean (`acceptanceDateTime`) |
| 13F institutional holdings | yes, but | free | UA header | **useless by construction — see below** |
| point-in-time index membership | no | — | — | not obtainable free; base rate kills it anyway |

### FINRA short interest — the publication-date trap, and how it was handled

`api.finra.org/data/group/otcMarket/name/consolidatedShortInterest` returns
`settlementDate` — the date the short position is measured — and **carries no
dissemination date at all**. Using `settlementDate` as the availability date is
the single largest lookahead trap in this source: on 2026-06-30 the 2026-06-30
vintage does not exist.

`publication_date()` therefore derives it as **settlement + 9 business days**,
which over the 27 vintages pulled lands 11–13 calendar days after settlement.

Verified against FINRA's own 2026 schedule (finra.org short-interest reporting
page): firms report by 6 p.m. ET on the second business day after settlement,
and FINRA disseminates about a week after that.

| FINRA's published 2026 schedule | derived `publication_date()` |
| --- | --- |
| settle Thu 15 Jan → publish Tue 27 Jan | Wed 28 Jan (1 day late) |
| settle Fri 30 Jan → publish Tue 10 Feb | Thu 12 Feb (2 days late) |

The derivation is **conservative on both checked vintages and never early**,
which is the direction that matters: erring long costs a stale vintage on a
handful of events, erring short is a Rules §04 violation.

The as-of join is `merge_asof(..., allow_exact_matches=False)` on
`publication_date` against the event's **cutoff date**, so a vintage published on
the cutoff date is refused rather than being assumed to have hit the tape before
16:00 ET.

Cost of the honesty: the freshest vintage available at an event is 11–28 days
old in *position* terms even before the ~15-day sampling interval is counted.

### FINRA daily short-sale volume — a stricter boundary than the price bar

`CNMSshvol<YYYYMMDD>.txt` is consolidated daily short volume per symbol, one
file per trading day, ~11k symbols, free, no auth. Retrieved 2025-08-01 →
2026-08-07: 256 trading days, 10 weekday files absent (holidays), 635,765 rows
after filtering to the archive's tickers.

The file for trade date T is published **after T's close**, so unlike the daily
price bar — which `runner.blocks.price_window` correctly allows on the cutoff
date, since the organisers price their own surprise metric off it — the
short-volume file dated the cutoff date is *not* in bounds. The last file used is
the last one dated strictly before the cutoff date. Measured lag from that file
to the event: median 1 day, mean 1.4, max 43 (a symbol that stopped reporting).

All features are **differences against the symbol's own 63-day mean**, never the
level. A symbol's short-volume ratio is dominated by its share of off-exchange
market-maker flow, which is persistent and has nothing to do with this quarter's
earnings; differencing mixes that out.

## Features and scores

ρ is the partial correlation with `y` controlling for `surprise_pct`, per quarter,
Fisher-z pooled. No fitting anywhere.

| feature | ρ full dev | ρ selection (n≈2k) | **ρ confirmation (n=4,144)** | ρ_b champion | coverage |
| --- | --- | --- | --- | --- | --- |
| `flow_combined` | 0.041 | 0.014 | **0.054** | 0.051 | 99.85% |
| `sv_21d_vs_63d` | −0.028 | 0.013 | **−0.047** | −0.033 | 99.85% |
| `si_chg_6` | −0.036 | −0.014 | **−0.045** | −0.006 | 99.17% |
| `sv_excess_5d_over_adv` | −0.032 | −0.019 | **−0.038** | −0.045 | 99.82% |
| `sv_10d_vs_63d` | −0.026 | −0.000 | **−0.038** | −0.050 | 99.85% |
| `si_chg_3` | −0.035 | −0.031 | **−0.037** | −0.010 | 99.76% |
| `sv_5d_z` | −0.020 | 0.008 | **−0.034** | −0.044 | 99.85% |
| `si_chg_3_over_adv` | −0.020 | −0.011 | **−0.033** | 0.007 | 99.76% |
| `sv_5d_vs_63d` | −0.019 | 0.005 | **−0.030** | −0.046 | 99.85% |
| `si_chg_1_over_adv` | −0.017 | 0.011 | **−0.028** | 0.007 | 99.87% |
| `si_chg_1` | −0.023 | −0.025 | **−0.021** | 0.011 | 99.87% |
| `si_days_to_cover` | 0.021 | 0.054 | **0.008** | −0.086 | 99.87% |

Resolution first: se on the confirmation partition is 1/√4144 = **0.0155**, on
the selection partition 0.022, on full dev 0.0128. So `flow_combined` at 0.054 is
3.5 se and `si_chg_6` at −0.045 is 2.9 se — distinguishable from zero. Nothing
here is distinguishable from **0.075**, half the bar.

The signs are all the ones the short-interest literature predicts: short interest
rising into the print, or the short-sale share running above its own baseline,
both go with a *lower* abnormal-return rank. Getting the sign right for free on
eleven features is weak evidence that the 0.03–0.05 is a real effect rather than
noise, which is worth saying because 0.03 in isolation would not be.

`si_days_to_cover` is the one **level**, and it is the one that behaves like a
selection artifact: 0.054 on the selection partition, 0.008 on confirmation. The
level is a firm characteristic; the change is the positioning signal. Confirmed.

`flow_combined` is registered `point_in_time=False`. It is unfitted — no weight
is estimated from `y` — but it averages *within-quarter percentile ranks*, and a
within-quarter cross-section is not reproducible at prediction time. For a single
feature the rank is monotone and changes nothing; for an average of five it is an
approximation and is flagged as one. Its five components were also chosen after
seeing dev ρ, which is why its dev number (0.041) is worth less than its
confirmation number (0.054).

### Interactions with surprise — tested and dead

Positioning theory says a crowded short should *amplify* a positive surprise (a
squeeze). Every `feature × (surprise_pct − 0.5)` interaction was scored: the
largest is |ρ| 0.015, all eleven under 1.2 se. **There is no measured squeeze
interaction in this data.** Ruled out.

### The horizon is a hump at 21 days, and 21 days is the ceiling

Scoring `sv_<w>d_vs_63d` for w ∈ {2, 5, 10, 21, 42}, confirmation partition:

```
w =  2      5      10     21     42
   -0.031 -0.030 -0.038 -0.047 -0.032
```

Not monotone — a hump peaking at **21 days**. Freshest is not best: the two-day
short-flow read right before the print is worse than the month-long drift. That
is the shape you would expect if the signal is slow accumulation of positioning
rather than informed pre-announcement trading. It also means the horizon has been
searched and **0.047 is this source's ceiling in this shape**; there is no window
left to tune.

### Rank transform — tested, no help

Rank-transforming the features within quarter before scoring (fat tails are a
real concern for a Pearson partial correlation) moved nothing: best |ρ| went from
0.036 to 0.032. The features are not tail-limited.

## ρ_b — the number this block was really built for

Against Agent A's price block and Agent C's regime block, surprise projected out,
Fisher-z pooled over dev quarters:

| | A.overnight_20d | A.pos_52w_range | A.runup_1d | A.volume_trend_20_60 | A.xsret_since_prior_earnings | A.xsrunup_20d | A.xsrunup_5d | A.xsrunup_60d | C.regime_slope | champion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flow_combined` | 0.007 | 0.076 | −0.010 | 0.005 | 0.109 | −0.023 | −0.046 | 0.067 | 0.046 | 0.051 |
| `si_chg_3` | −0.005 | −0.038 | 0.003 | 0.108 | −0.071 | −0.031 | 0.003 | −0.050 | −0.033 | −0.010 |
| `si_chg_6` | 0.017 | −0.058 | 0.001 | 0.029 | −0.053 | −0.015 | 0.018 | −0.026 | −0.037 | −0.006 |
| `sv_21d_vs_63d` | −0.009 | −0.021 | 0.011 | −0.069 | −0.064 | 0.042 | 0.033 | −0.026 | −0.022 | −0.033 |
| `sv_excess_5d_over_adv` | 0.012 | −0.014 | −0.011 | −0.063 | −0.037 | 0.049 | 0.125 | −0.005 | −0.034 | −0.045 |
| `si_days_to_cover` | −0.064 | −0.111 | 0.023 | −0.133 | −0.057 | −0.006 | 0.003 | −0.071 | −0.013 | −0.086 |

Maximum |ρ_b| between any flow feature and any price feature is **0.133**, and
that is `si_days_to_cover` against `volume_trend_20_60` — two volume aggregates,
and the flow feature involved is the one that failed confirmation. Among the
features that survive, |ρ_b| against price is ≤ 0.125 and mostly under 0.07.

**Ownership and price are genuinely separate reads of positioning.** The
hypothesis that they duplicate each other is falsified. The problem is not
duplication; the problem is that neither has signal.

## 13F — the branch is closed

**Verdict: 13F cannot be made both point-in-time correct and recent enough to
matter. Permanently ruled out for this competition.**

The arithmetic, derived, no download required. A 13F-HR is due 45 days after
quarter end. For an event at 16:00 ET on date D, the newest 13F whose acceptance
precedes D reports positions as of the last quarter end at least 45 days before
D. Computed over all 8,020 events, assuming *every* fund files exactly on its
deadline (the most favourable possible assumption — in reality filings cluster
in the final days, which only makes this worse):

```
content staleness at the event cutoff, in days
min 48   p25 112   median 121   p75 127   max 137
```

**The median event would be reading a snapshot of institutional ownership 121
days — four months — old.** Not 45. The 45-day deadline is the lag from the
*quarter end*, and a randomly-placed event falls on average half a quarter past
the deadline, so the two lags add.

Set against the measured result that a **11-to-28-day-stale** short interest
change — a direct, aggregate, exchange-reported positioning measure — produces
ρ 0.037, a 121-day-stale quarterly ownership delta clearing 0.15 is not
plausible. The cost of finding out would be the download and parse of roughly
5,000–8,000 13F-HR information tables per quarter across four quarters, tens of
GB, for a feature that is structurally staler than the one already measured at a
quarter of the bar.

No 13F feature was built. The rigorous answer is that the vintage exists and the
`acceptanceDateTime` filter is mechanically easy — the source is *obtainable* and
*point-in-time-able*. It is simply too old to be about this earnings event.

## Index membership — not built

Base rate kills it before point-in-time does. S&P 500 additions and deletions run
roughly 5–10 per quarter. Across four quarters against 8,020 events that is on
the order of 30–60 events with a nonzero value, ~0.5% coverage, and a feature
present on 0.5% of events cannot move a pooled ρ regardless of how strong it is
on its own sub-sample. Free point-in-time constituent history is also genuinely
hard (S&P's own press releases are the primary source and are not offered as a
dataset). **Not worth a fetch.** Ruled out on coverage, not on point-in-time.

## Form 4 — insider transactions

Agent B owns the EDGAR Form 4 parse and had not published a `data/edgar/form4*`
table when this block was assembled, so this is the *minimal version, and it says
so*. It does reuse B's `data/edgar/submissions/` cache read-only, which is where
`acceptanceDateTime` per accession comes from — **the only timestamp usable for
cutoff filtering**, because `filingDate` rolls to the next business day for
anything accepted after 17:30 ET.

Only transaction codes **P** (open-market purchase) and **S** (open-market sale)
are retained. Code A grants are booked at price 0, are compensation rather than a
view, and dominate by count.

Measured scope: with a 120-day window and acceptance strictly before the cutoff,
the 7,730 events whose CIK is in B's cache need **101,692 unique Form 4
accessions**, a mean of 15.1 Form 4s per event. At the SEC's 10 req/s ceiling
that is **~2.8 hours of wall clock**, which is the single reason this did not
land in this cycle. A stratified 2,500-event dev sample (~28k accessions, ~50
min) was launched and had not returned when this was written.

**Status: pipeline written and scoped, no ρ measured.** `f4_net_share_ratio`,
`f4_net_dollar_ratio` and `f4_net_count` are **not in the block** and no Form 4
result is claimed here. The cache lands in `data/flow/form4_txn.parquet` and
`form4_features()` scores off it without another fetch.

The expensive half is already paid for, and it is not the XML: it is the
CIK → accession → `acceptanceDateTime` index, which Agent B's
`data/edgar/submissions/` cache already covers for 2,446 of 2,610 tickers
(98.6% of events). Whoever finishes this should reuse it rather than re-crawl.

Prior, stated now so it can be checked later: short interest change is a
*direct, aggregate, exchange-reported* measure of the same "who is positioned
which way" question, and it measures 0.037. Insider net buying is a narrower,
noisier read of a smaller group. Expect 0.03–0.06, not 0.15.

## Cutoff-safety proof

- 8,020/8,020 events have `knowledge_cutoff` at 16:00 ET (verified upstream).
- Short interest: value used is a vintage whose *derived publication date*
  (settlement + 9 business days, one day longer than FINRA's stated
  dissemination) is **strictly** before the cutoff **date**;
  `allow_exact_matches=False` refuses a same-date vintage rather than assuming
  its publication hour.
- Short volume: value used is a rolling window ending on a trade date **strictly
  before** the cutoff date — stricter than the price-bar rule, because the file
  is published after that day's close.
- Form 4: `acceptanceDateTime` strictly less than the cutoff instant, in
  America/New_York.
- Every retrieval went through `sources.fetch` and is in
  `data/audit/fetches.jsonl`: 266 `finra_short_volume` file fetches (256 files,
  10 weekday holidays that 404), 116 `finra_short_interest` API pages, and the
  `edgar_form4` batch. **`not_live_safe = 0` for both FINRA sources** — the
  block ships as measured, which is not true of every source in this cycle.

Known coverage limitation: the FINRA caches were filtered to the ticker set drawn
from `harness.load` (2,518 tickers, the scorable events). The 180 archive records
that `harness.load` drops for missing `y`/`surprise_pct` contribute ~92 tickers
that were not in that filter, so block coverage is 97.6–97.9% of 8,020 but
**99.2–99.9% of the 7,840 scorable events**. Re-pulling for the unscorable
remainder is cheap and pointless for scoring; it would matter for a live deploy.

## Disclosure

`2026Q3` is sealed. While checking per-quarter sign stability I printed
per-quarter ρ for all four quarters, including 2026Q3, before catching it. **No
decision in this report used it** — every feature had already failed the bar on
dev alone, and no feature, threshold, or weight was changed after seeing it. For
completeness: the dev-quarter signs are inconsistent (`si_chg_3` runs −0.048 /
+0.009 / −0.067 across 2025Q4 / 2026Q1 / 2026Q2), which is what a pooled ρ of
0.035 at a per-quarter se of 0.022 should look like.

## What I would build next

1. **Nothing more in this block.** The horizon has been searched (hump at 21
   days), the interaction has been tested (dead), the transform has been tested
   (no help), and the two live sources are already at their natural resolution.
   Squeezing this source further is spending on a channel whose ceiling is
   measured at 0.047.
2. **Finish Form 4 only if someone is already paying the EDGAR crawl.** 2.8
   hours of wall clock for an expected 0.03–0.06. It is the right thing to hand
   to Agent B — who is already crawling EDGAR — rather than a standalone job.
3. **The one genuinely untried ownership source: ETF and index-fund flow.**
   Daily creation/redemption data is free from issuers, and unlike 13F it is
   T+1. Not attempted here, and honestly assessed: it is a *sector*-level signal
   that would need mapping to single names, so it is closer to Agent C's regime
   channel than to positioning.

## Ruled out permanently

- **13F, in any form.** Median content staleness at the event cutoff is 121
  days under the most favourable possible filing assumption. The source is
  point-in-time-*able* and still useless. Closed.
- **Index membership and changes.** ~5–10 S&P 500 changes per quarter against
  8,020 events is ~0.5% coverage. Closed on base rate, before point-in-time even
  comes up.
- **Short-interest *level* as a signal.** `si_days_to_cover` scored 0.054 on the
  selection partition and 0.008 on confirmation — the textbook shape of a
  selection artifact. It is a firm characteristic; only the change is
  positioning. Confirmed by measurement, not assumed.
- **Any squeeze interaction between short positioning and surprise.** All eleven
  `feature × surprise` terms under 1.2 se.
- **`settlementDate` as an availability date for FINRA short interest.** Not a
  feature, a trap; naming it so nobody in a later cycle rediscovers it the
  expensive way.
