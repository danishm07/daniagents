# peers — sector-relative reaction (Agent C, C1)

Built on Agent B's `data/reference/sic.parquet` (97.6% event-level SIC
coverage; 346 four-digit, 230 three-digit, 66 two-digit groups). 70 columns:
own-ticker history, market-wide reference, and SIC widths 4/3/2 × windows
10/30/90 days × {mean, n, excess-over-market, most-recent, beat-only,
miss-only, beat−miss spread}.

**Verdict: nothing clears the bar. Best lagged feature ρ 0.066 dev / 0.052
confirm. But the sector effect is real and the ceiling is now measured, so
this closes with a number rather than a shrug.**

## The ceiling — a contemporaneous oracle on real SIC

The strongest possible version of the hypothesis: know the residual reaction of
every *same-industry* announcer, contemporaneously, leave-one-out. No live
system can have this — a same-day peer's return window closes after our cutoff.

| cohort | same DAY | same WEEK | same QUARTER |
|---|---|---|---|
| SIC-4 | **+0.113** (n 2,082) | +0.064 (n 3,726) | +0.041 (n 5,517) |
| SIC-3 | +0.091 (n 2,755) | +0.045 (n 4,339) | +0.028 (n 5,785) |
| SIC-2 | +0.075 (n 4,255) | +0.030 (n 5,455) | +0.006 (n 6,087) |
| market-wide | +0.043 (n 6,124) | +0.028 (n 6,142) | — |
| text-cluster proxy | +0.053 (n 4,940) | +0.036 (n 5,875) | — |

Three things this settles.

1. **Sector-relative genuinely beats market-wide**, 0.113 vs 0.043 at SIC-4,
   and it beats a text-clustered proxy (0.053) — Agent B's file was worth
   fetching. Narrow is better than broad: 4-digit > 3-digit > 2-digit,
   monotonically, at every horizon.
2. **The effect is same-day and decays fast**: 0.113 → 0.064 → 0.041 going
   day → week → quarter. What a live system can build is the lagged version,
   which is the weakest column of that table.
3. **The ceiling is still 0.113, at 43% coverage, using information nobody
   has.** The bar is 0.15. Even the impossible version does not clear it, and
   the buildable version is a fraction of the impossible one. My best lagged
   SIC feature measures ~0.03, consistent with the same-week 0.064 attenuated
   by lag and cohort thinness.

Structural reason the number is small: `car1` is `r_i − r_m`, market-adjusted
at source (`n_obs == 2` on 100% of archive rows). The common factor a peer
aggregate would capture is already differenced out of the target. Only the
industry-specific residual survives, and that is the 0.113.

## Leading features

| feature | ρ dev | ρ confirm | ρ_b champion | coverage |
|---|---|---|---|---|
| `peer_own_hist_mean` | −0.0661 | −0.0510 | **+0.161** | 0.234 |
| `peer_sic2_w30_n` | −0.0493 | −0.0523 | +0.036 | 0.988 |
| `peer_sic2_w90_n` | −0.0478 | −0.0518 | +0.047 | 0.988 |
| `peer_sic3_w90_n` | −0.0469 | −0.0456 | +0.057 | 0.988 |
| `peer_sic3_w30_n` | −0.0467 | −0.0437 | +0.041 | 0.988 |
| `peer_own_hist_last` | −0.0384 | −0.0377 | +0.126 | 0.234 |
| `peer_mkt_w90` | −0.0365 | −0.0331 | −0.013 | 0.994 |
| `peer_sic2_w30_beat` | −0.0298 | −0.0306 | −0.030 | 0.854 |
| `peer_mkt_w30` | −0.0271 | −0.0340 | +0.013 | 0.993 |
| `peer_sic2_w30_recent` | +0.0229 | +0.0249 | −0.012 | 0.891 |

Max |ρ| across all 70 columns: **0.066 dev, 0.052 confirmation**. At n ≈ 6,000
the SE is 0.013, so the top of this list is real and roughly a third of the bar.

Two things worth naming.

- **The `_n` columns beat every actual peer aggregate.** Cohort size at ρ ≈
  −0.049 stable across dev and confirmation, at 98.8% coverage, is the most
  reliable column in the block — and it is a *count*, not a reaction. It is
  registered `directional=False`. What it is measuring is industry reporting
  crowding, not peer information; it also tracks `regime_season_day` (ρ −0.037),
  which is the same calendar fact in another coordinate. Somebody could chase
  this as a crowding/attention channel, but it is a magnitude quantity and the
  project has three independent failures of that shape already.
- **`peer_own_hist_mean` is the largest and least useful.** ρ_b 0.161 with the
  champion (the most correlated column in either block), coverage 0.234, and at
  n=1,440 its SE is 0.026, so −0.066 is 2.5 SE. Small, thin, and not
  decorrelated. Company-level earnings-reaction momentum is mildly negative
  (reversal) and not worth a channel.

Conditioning on peer surprise sign helps in the right direction — `_beat`
(−0.030) beats plain `_mean` at the same width and window — which is the idea
working, at a magnitude that does not matter. Most-recent-peer (`_recent`) flips
sign against the window mean at ~+0.023, also consistent, also too small.
`_excess` (sector minus market) is ρ_b **0.995** with `_mean` at SIC-4 — the
market term is noise at that width and subtracting it does nothing.

## Within-block ρ_b (dev, surprise projected out)

```
                      champ  mkt10  mkt90  own_h  s2beat  s2_n  s2rec  s4exc  s4mean
champion              1.000  0.032 -0.013  0.161  -0.030  0.036 -0.012  0.023   0.022
peer_mkt_w10          0.032  1.000  0.588 -0.030   0.091 -0.002  0.072 -0.076   0.007
peer_mkt_w90         -0.013  0.588  1.000 -0.005   0.120  0.057  0.082 -0.039   0.032
peer_own_hist_mean    0.161 -0.030 -0.005  1.000   0.031  0.015 -0.010  0.016   0.017
peer_sic2_w30_beat   -0.030  0.091  0.120  0.031   1.000 -0.014  0.189  0.373   0.384
peer_sic2_w30_n       0.036 -0.002  0.057  0.015  -0.014  1.000  0.025  0.045   0.049
peer_sic2_w30_recent -0.012  0.072  0.082 -0.010   0.189  0.025  1.000  0.051   0.052
peer_sic4_w30_excess  0.023 -0.076 -0.039  0.016   0.373  0.045  0.051  1.000   0.995
peer_sic4_w30_mean    0.022  0.007  0.032  0.017   0.384  0.049  0.052  0.995   1.000
```

Cross-block: `peer_mkt_w10` and `regime_level` are ρ_b **0.868** — the same
statistic under two names. The SIC columns are properly decorrelated from the
market ones (`peer_sic4_w30_excess` vs `peer_mkt_w10` = −0.076), which is the
one design goal this block did hit. It bought nothing, because decorrelation
without signal buys nothing.

## Cutoff safety

- Peer admissibility is `peer.settled <= our.knowledge_cutoff`, where `settled`
  = 16:00 America/New_York on the peer's own `event_returns[*].window_end_date`
  read from the archive, converted through the tz database so the Nov-2025 and
  Mar-2026 DST boundaries are right.
- Audited directly: **0 of 8,020 events** use an input that settled after their
  cutoff. 7,975 events have at least one input settling *at exactly* the cutoff
  instant (mean 108 such inputs) — that is the same 16:00 ET close the
  organisers price their own surprise metric off, so it is in-bounds, and it is
  the boundary the whole block leans on.
- 12 events have `settled <= cutoff` for their *own* row. All 12 are excluded by
  event_id from their own cohort (and dropped to NaN in `regime`).
- The looser "the peer reported before we did" filter was **not** used.
- **Nothing aggregates `y`.** All aggregates are over `car1` divided by a single
  trailing 90-day cross-sectional dispersion — deliberately not the cohort's own
  sd, since a four-name cohort's sd carries ~40% error and dividing by it
  amplifies noise exactly when the cohort is least trustworthy.

## Point-in-time status

Clean throughout. SIC is a slow-moving company attribute taken from EDGAR
submissions; it is a *current* value rather than an as-of-date value, so the
cohort *identity* is approximate in the same mild way a sector label always is.
Every other input — window membership, peer reactions, the normaliser — is
strictly point-in-time.
