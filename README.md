[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/giber76/gta6-twitch-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/giber76/gta6-twitch-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch forecast

Entry for [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) ($250): forecast the Twitch "Grand Theft Auto VI" category's concurrent viewers hour by hour (hours 2–24 UTC of release day, 19 Nov 2026), as five quantiles (5%, 25%, 50%, 75%, 95%) per hour. Score is pinball loss on the log scale; a reward needs a lower loss than the house baseline and the house AI model.

## How it works

`main.py` (Python 3.13, standard library only, offline, deterministic) reads one case from `/input` and writes `/output/<id>.out`:

1. **Clean comparables only.** Each comparable is a past big launch with a full release-day hourly curve. The kit tags every comparable's unlock text with a `Tier:` — early-access-contaminated launches, weak analogues (free-to-play / slow-burn), and the GTA VI category's own pre-release history are excluded; only the `comparable` tier is used. The 4 practice cases (Wolverine, RDR2, GoW Ragnarök, Spider-Man 2) are public past launches not in the kit's comparables list, so they are added as extra training points; the target itself is always excluded by name (leave-one-out).
2. **Log-ratio curves.** For each comparable, `r(h) = ln(viewers_h / viewers_1)`. The ratio's level depends strongly on how big hour 1 already was (corr(ln p1, r_h) ≈ −0.9 by the evening): games already live everywhere at 00:00Z stay flat, games still unfolding in later time zones grow several-fold.
3. **Local-linear kernel regression.** Per hour h = 2..24, a Gaussian-kernel (bandwidth 1.5 in ln p1) local-linear fit of r(h) on ln p1, evaluated at the target's ln p1. The slope shrinks toward 0 when the effective sample is small, so a target bigger than every comparable (GTA VI's hour 1 may be) falls back to the neighbourhood mean instead of extrapolating a noisy slope; the median ratio is clamped to the observed range.
4. **US-unlock alignment.** Rolling-midnight launches jump when the US East Coast unlocks (04:00Z under daylight time, 05:00Z otherwise). Each non-simultaneous comparable's curve is shifted so its biggest early step (hours 3–8, if ≥ 0.25 log) lands on the target's US-midnight hour, computed from the release date and the US daylight-time calendar. Simultaneous global unlocks are not shifted.
5. **Quantiles.** Normal around the fitted median, with a per-hour spread taken from a leave-one-out (LOO) out-of-sample estimate, scaled by 0.25: each comparable is predicted from all the others, and the standard deviation of those out-of-sample residuals sets the width. The in-sample kernel-weighted residual spread is too narrow — the fit absorbs noise — while the LOO spread is an honest out-of-sample width, wider mid-day where the unlock transition and midday trough make the shape unpredictable, relatively narrower late.

Fallbacks keep every output valid: if too few clean comparables have a complete release day, all complete ones are used; if still too few, the persistence forecast (flat at hour 1, random-walk spread) is used.

## Practice grade (local scorer, 4 practice launches)

| case | loss |
|---|---|
| Marvel's Wolverine | 0.094 |
| Red Dead Redemption 2 | 0.118 |
| God of War Ragnarök | 0.088 |
| Marvel's Spider-Man 2 | 0.177 |
| **mean** | **0.119** |

House baseline: **0.387** (typical miss 82.8%). This entry: **0.119** (typical miss 21.1%, 90% coverage 0.891) — a 69% lower pinball loss. For reference, on the live practice board (~3,600 entries) 0.119 would rank around #29–30 (top-20 cutoff ~0.115). The final is a single case (GTA VI, whose hour-1 print will likely dwarf every practice launch), so the practice grade is encouraging but no guarantee.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see [LICENSE](LICENSE).
