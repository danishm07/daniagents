# regime — the response function over time (Agent C, C2)

**Verdict: dead, and closed permanently.** Not "too noisy to estimate" — the
hypothesis itself does not price, and that is shown by an oracle rather than by
a failure to fit.

## The hypothesis

`y` is a percentile rank *within the quarter*. If the market's mapping from
surprise to rank drifts across the quarter, two identical disclosures land at
different percentiles depending on when they occur. The feature the hypothesis
predicts is therefore not the slope, it is **own surprise × the change in
slope** — where *this* surprise lands *today* rather than on average.

## The drift is real. It just does not pay.

Measured, `harness.load_all()`, weekly buckets of ≥40 events within a quarter:
the slope of `y` on `surprise_pct` has mean 0.21 and **sd 0.156**, ranging
−0.026 to 0.452 inside 2025Q4 alone. The response function moves a lot.

Then the ceiling. An **oracle** was built that fits the weekly slope on the
quarter's own events — information no live system can have — and compares it to
the quarter-wide slope:

| oracle variant | ρ (partial, vs `surprise_pct`) | n |
|---|---|---|
| in-sample weekly interaction | +0.071 | 5,965 |
| in-sample weekly level | +0.076 | 5,965 |
| **leave-one-out weekly interaction** | **+0.020** | 5,965 |
| **leave-one-out weekly level** | **+0.033** | 5,965 |

At n = 5,965 the standard error on ρ is 0.013. The leave-one-out interaction at
0.020 is 1.5 SE — not distinguishable from zero. The gap between the in-sample
0.071 and the leave-one-out 0.020 is the measure of how much of the observed
slope "drift" is just estimation noise in a 120-event weekly regression: most
of it.

**0.020 is the ceiling of this channel with future information.** The bar is
0.15. No estimator can close a 7× gap to a quantity that does not exist.

## Why, structurally

`car1` is already market-adjusted (`r_i − r_m`, `n_obs == 2` on 100% of archive
rows). The common component that a "what is the regime today" feature would
capture has been differenced out of the target by construction before we ever
see it.

## Built (point-in-time, all of it)

Trailing pool = every event whose return window closed at or before this
event's cutoff. Ranks are computed **within the trailing pool**, never from the
quarter-wide `y`/`surprise_pct` pair, which does not exist until the quarter
closes.

| feature | ρ dev | ρ confirm | ρ_b champion |
|---|---|---|---|
| `regime_season_day` | −0.0366 | −0.0438 | −0.071 |
| `regime_n_fit` (normaliser) | −0.0337 | −0.0433 | −0.037 |
| `regime_level` | −0.0253 | −0.0320 | +0.013 |
| `regime_slope_delta` | −0.0234 | −0.0263 | −0.002 |
| `regime_slope` | −0.0208 | −0.0236 | +0.001 |
| `regime_x_slope` | −0.0179 | −0.0107 | +0.039 |
| `regime_resid_sd` (normaliser) | +0.0115 | +0.0117 | +0.005 |
| `regime_x_slope_delta` | **+0.0006** | **+0.0092** | +0.024 |

The measured PIT number for the interaction (0.0006 dev / 0.0092 confirm) sits
inside the oracle's own 0.020 ceiling. The estimator is not the bottleneck.

## Two bugs caught here, worth carrying forward

1. **`Series.astype("int64")` on a tz-aware datetime returns microseconds in
   pandas 2, not nanoseconds.** The first build's "21-day" trailing window was
   therefore 21,000 days — the entire archive — and every event got the same
   slope, so `regime_slope_delta` was identically 0.0. Use
   `.dt.as_unit("ns").astype("int64")`. `sources_archive._ns` does this in one
   place.
2. **`peer_view` in `runner/arms_builtin.py` guesses the settlement instant** as
   `knowledge_cutoff + 1 day`. Measured against the archive's own
   `event_returns[*].window_end_date`: cutoff+1 holds on 7,382 of 8,020 events;
   616 settle 2–7 days later and 12 settle same-day. On those 616 the existing
   arm admits peers whose window had **not** closed. A small live leak, and it
   inflates the arm's reported number.
