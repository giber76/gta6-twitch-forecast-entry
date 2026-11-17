"""gta6-twitch-ccu-forecast-01: launch-day Twitch viewership forecaster.

Model (built from the kit's practice cases, no network, stdlib only):

  * Each comparable is a past big launch with a full release-day hourly curve.
    We keep only the clean ones: the kit annotates every comparable's unlock
    text with a "Tier:" tag; early-access-contaminated launches, weak
    analogues (free-to-play / slow-burn) and the GTA VI category's own
    pre-release history are excluded. What remains is the "comparable" tier.
    The 4 practice cases (Wolverine, RDR2, GoW Ragnarok, Spider-Man 2) are
    public past launches that are NOT in the kit's comparables list, so
    they are added here as extra training points (PRACTICE_COMPARABLES);
    the target itself is always excluded by name, so a prediction never
    trains on the case it scores.
  * For each comparable we take the release-day log-ratio curve
    r(h) = ln(viewers_h / viewers_1).  The ratio's level depends strongly on
    how big hour 1 already was (corr(ln p1, r_h) ~ -0.9 by the evening):
    games already live everywhere at 00:00Z stay flat, games still unfolding
    in later time zones grow several-fold.
  * Per hour h = 2..24 we fit a local-linear (Gaussian kernel, bandwidth 1.5
    in ln p1) regression of r(h) on ln p1, evaluated at the target's ln p1.
    The kernel slope is shrunk toward 0 when the effective sample is small,
    so a target bigger than every comparable (GTA VI's hour 1 may be) falls
    back gracefully to the neighbourhood mean instead of extrapolating a
    noisy slope; the median ratio is also clamped to the observed range.
  * Rolling-midnight launches jump when the US East Coast unlocks (04:00Z
    under daylight time, 05:00Z otherwise).  Each non-simultaneous
    comparable's curve is shifted so its biggest early step (hours 3..8,
    if the step is at least 0.25 log) lands on the target's US-midnight
    hour, computed from the target's release date and the US daylight-time
    calendar.  Simultaneous global unlocks are not shifted.
  * Quantiles are normal around the fitted median, with a per-hour spread
    taken from a leave-one-out (LOO) estimate, scaled by 0.25: for each
    comparable we predict its curve from all the others and take the
    standard deviation of those out-of-sample residuals.  The in-sample
    kernel-weighted residual spread is too narrow — the fit absorbs noise —
    while the LOO spread is an honest out-of-sample width, wider in the
    middle hours where the unlock transition and the midday trough make the
    shape genuinely unpredictable, relatively narrower late.  On the 4
    practice cases the LOO spread cuts the mean loss from 0.130 to 0.122,
    and adding the 4 practice cases themselves as comparables (leave-one-out)
    cuts it further to 0.119 (every case holds or improves except RDR2 and
    Spider-Man 2, which worsen ~0.5% — the twin pair at nearly identical
    ln p1 pulling each other; the 0.25 multiplier is flat-bottomed
    0.225–0.275 and stable when each case is held out from the tuning).
    The LOO SD runs ~2x the in-sample kernel SD in the middle hours, so
    the gain comes from the per-hour shape, not the overall width: the
    kernel spread at its own optimal 0.5 multiplier is the same width
    mid-day but too wide late.  The residual mixture is tight around the
    median with a few far outliers, so the full normal spread is far too
    wide on pinball loss (validated on the 4 practice cases: 0.130 vs
    0.387 for the house baseline, 0.122 with the LOO spread, and 0.119
    with the practice cases added as comparables).
  * Fallbacks keep every output valid: if too few clean comparables have a
    complete release day, use all complete ones; if still too few, fall back
    to the persistence forecast (flat at hour 1, spread widening like a
    random walk).  Deterministic: fixed file order, fixed comparable order,
    no randomness, no wall-clock or network use.

I/O contract (same as the kit starter): read every <id>.in in the input
directory (MAKEFUN_INPUT_DIR, or /input), write <id>.out to the output
directory (MAKEFUN_OUTPUT_DIR, or /output):
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ..., {"hour": 24, ...}]}
"""

import datetime as _dt
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
Z_BY_LEVEL = dict(zip(LEVELS, Z))

BANDWIDTH = 1.5        # Gaussian kernel bandwidth in ln(p1)
SPREAD_MULT = 0.5      # residual-spread multiplier (pinball-optimal on practice)
LOO_SPREAD_MULT = 0.25  # multiplier for the LOO out-of-sample spread
# (the LOO SD runs ~2x the in-sample kernel SD in the middle hours, so its
# pinball-optimal multiplier is half the kernel one; the gain over the
# kernel spread comes from the per-hour shape, not the overall width)
MIN_JUMP_STEP = 0.25   # log step that counts as the US-unlock jump
HOURS = list(range(2, 25))

# Extra training points: the 4 practice cases are public past launches
# (hourly Twitch category averages from SullyGnome, the same source the
# kit's comparables come from), but they are NOT in the kit's comparables
# list -- so they are added here as extra comparables. Each entry is
# (release day, hour-1 viewers, viewers for hours 1..24). All four are
# rolling-midnight launches. When the program scores one of them, that
# case is excluded by name in forecast_case (leave-one-out): a prediction
# never trains on the case it scores. At final time (GTA VI) all four
# are used. Adding them (LOO) cuts the practice mean loss from 0.1218
# to 0.1190: Wolverine 0.0987 -> 0.0937, RDR2 0.1171 -> 0.1177,
# GoW Ragnarok 0.0957 -> 0.0879, Spider-Man 2 0.1759 -> 0.1767.
# The two ~0.5% worsenings (RDR2, Spider-Man 2) are the twin pair at
# nearly identical ln p1 (11.67 vs 11.69) pulling each other toward the
# other's shape; the gains (Wolverine -5%, GoW -8%) are an order of
# magnitude larger, and at final scale (ln p1 14-15) the two models agree
# within 0.04 log, so the twin effect is irrelevant to the final.
PRACTICE_COMPARABLES = {
    "Marvel's Wolverine": ("2026-09-15", [23357,
        19397, 20292, 18329, 35760, 40523, 38196, 30468, 37322, 32884, 36770,
        41223, 45078, 58758, 78329, 91382, 133910, 189396, 175604, 206982,
        212243, 196416, 153926, 147358]),
    "Red Dead Redemption 2": ("2018-10-26", [116550,
        102778, 99986, 84459, 152428, 176319, 163532, 148485, 134888, 120686,
        119921, 138073, 165093, 195646, 231529, 264043, 330969, 369358,
        322991, 290598, 281596, 258394, 223948, 192584]),
    "God of War Ragnarök": ("2022-11-09", [269517,
        220787, 146900, 141669, 156301, 281917, 273454, 258613, 196345, 186781,
        189204, 152840, 179704, 217655, 271761, 310384, 376075, 435038,
        444580, 460387, 445097, 434600, 411192, 383725]),
    "Marvel's Spider-Man 2": ("2023-10-20", [119087,
        79573, 51615, 50272, 166214, 146818, 125021, 109152, 93203, 83997,
        71214, 43730, 68551, 100971, 131596, 164722, 176018, 225745, 250538,
        240849, 258378, 263140, 253088, 216387]),
}


def practice_comparable_records():
    """Kit-shaped comparable records for the 4 practice cases."""
    recs = []
    for name, (day, hours) in PRACTICE_COMPARABLES.items():
        hourly = [{"from": "%sT%02d:00:00Z" % (day, h - 1),
                   "viewers": v, "reads": 1}
                  for h, v in enumerate(hours, start=1)]
        recs.append({
            "name": name,
            "platforms": "",
            "day_start": day + "T00:00:00Z",
            "hourly": hourly,
            "unlock": "Rolling midnight local launch. Tier: comparable",
            "source": "SullyGnome (practice case, public past launch)",
        })
    return recs


def round2(x):
    """Round to 2 decimals, half up (as JavaScript Math.round(x*100)/100)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def us_east_midnight_hour(date):
    """Hour index (1 = 00:00-01:00Z) of US-East midnight on a UTC date.

    Eastern daylight time (UTC-4) puts midnight at 04:00Z (hour 5); Eastern
    standard time (UTC-5) puts it at 05:00Z (hour 6).  US daylight time runs
    from the second Sunday of March to the first Sunday of November.
    """
    def nth_sunday(year, month, n):
        d = _dt.date(year, month, 1)
        return d + _dt.timedelta(days=(6 - d.weekday()) % 7 + 7 * (n - 1))

    start = nth_sunday(date.year, 3, 2)
    end = nth_sunday(date.year, 11, 1)
    return 5 if start <= date < end else 6


def tier_of(comp):
    """Tier tag from the unlock text, or None if the text carries none."""
    text = comp.get("unlock") or ""
    for tag in ("early-access-contaminated", "weak", "target-his", "comparable"):
        if "Tier: " + tag in text:
            return tag
    return None


def is_simultaneous(comp):
    """True if the launch was one single global unlock (no rolling midnight)."""
    text = (comp.get("unlock") or "").lower()
    return any(p in text for p in (
        "single global", "single simultaneous", "simultaneous across",
        "pc single time", "simultaneous midnight gmt",
    ))


def release_day_hours(comp):
    """{hour: viewers} for the comparable's release day, or None if incomplete."""
    day = (comp.get("day_start") or "")[:10]
    if not day:
        return None
    hours = {}
    for pt in comp.get("hourly") or []:
        if (pt.get("from") or "")[:10] != day:
            continue
        v = pt.get("viewers")
        if v is None or v <= 0:
            continue
        hours[int(pt["from"][11:13]) + 1] = v
    if 1 not in hours or any(h not in hours for h in HOURS):
        return None
    return hours


def parse_comparable(comp):
    """Turn one comparable into model inputs, or None if unusable."""
    hours = release_day_hours(comp)
    if hours is None:
        return None
    p1 = hours[1]
    ratios = {h: math.log(hours[h] / p1) for h in HOURS}
    # The US-unlock jump: biggest log step in the early hours.
    jump_h, jump_step = 5, 0.0
    for h in range(3, 9):
        step = ratios[h] - ratios[h - 1]
        if step > jump_step:
            jump_h, jump_step = h, step
    return {
        "name": comp.get("name") or "",
        "lnp1": math.log(p1),
        "ratios": ratios,
        "jump_h": jump_h,
        "jump_step": jump_step,
        "simul": is_simultaneous(comp),
        "tier": tier_of(comp),
        "day": (comp.get("day_start") or "")[:10],
    }


def loo_spreads(comps, target_jump_h):
    """Per-hour out-of-sample spread: leave each comparable out, predict its
    log-ratio curve from all the others (aligned to its own US-midnight
    hour), and take the standard deviation of the residuals.  Returns
    {hour: sd}, or None if there are too few comparables for a stable
    estimate.  Deterministic: fixed comparable order, no randomness."""
    if len(comps) < 5:
        return None
    out = {}
    for h in HOURS:
        resids = []
        for c in comps:
            others = [o for o in comps if o is not c]
            c_day = _dt.datetime.fromisoformat(c["day"]).date()
            c_jump_h = us_east_midnight_hour(c_day)
            med_r, _, _, _ = fit_hour(others, h, c["lnp1"], c_jump_h)
            resids.append(c["ratios"][h] - med_r)
        mean = sum(resids) / len(resids)
        out[h] = math.sqrt(sum((r - mean) ** 2 for r in resids) / len(resids))
    return out


def aligned_ratio(comp, h, target_jump_h):
    """The comparable's log-ratio at hour h, shifted so its US-unlock jump
    lands on the target's US-midnight hour.  Simultaneous launches are not
    shifted; neither are comparables whose early jump is too small to trust."""
    if comp["simul"] or comp["jump_step"] < MIN_JUMP_STEP:
        return comp["ratios"].get(h)
    src = h - (target_jump_h - comp["jump_h"])
    if src in comp["ratios"]:
        return comp["ratios"][src]
    nearest = min(comp["ratios"], key=lambda x: abs(x - src))
    return comp["ratios"][nearest]


def weighted_linear(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x.  Returns (a, b)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    return my - b * mx, b


def fit_hour(comps, h, lp, target_jump_h):
    """Local-linear fit for one hour. Returns (med_r, sd, res_pairs, ess)."""
    xs = [c["lnp1"] for c in comps]
    ys = [aligned_ratio(c, h, target_jump_h) for c in comps]
    ws = [math.exp(-0.5 * ((x - lp) / BANDWIDTH) ** 2) for x in xs]
    a, b = weighted_linear(xs, ys, ws)
    # Shrink the slope toward 0 when the effective sample is small
    # (protects extrapolation when the target is bigger than every
    # comparable), then re-center the intercept at the shrunk slope.
    ess = sum(ws) ** 2 / sum(w * w for w in ws)
    shrink = ess / (ess + 3.0)
    b *= shrink
    a = sum(w * (y - b * x) for w, x, y in zip(ws, xs, ys)) / sum(ws)
    med_r = a + b * lp
    # Clamp the median ratio to the observed range, widened by 10%.
    lo, hi = min(ys), max(ys)
    med_r = max(lo - 0.1 * (hi - lo), min(hi + 0.1 * (hi - lo), med_r))
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    sw = sum(ws)
    sd = math.sqrt(sum(w * r * r for w, r in zip(ws, res)) / sw)
    return med_r, sd, list(zip(res, ws)), ess


def forecast_hours(p1, comps, target_jump_h, zs=Z):
    """Per-hour quantiles from the fitted model, at the given z-values."""
    lp = math.log(p1)
    loo = loo_spreads(comps, target_jump_h)
    out = []
    for h in HOURS:
        med_r, sd, _, _ = fit_hour(comps, h, lp, target_jump_h)
        # Spread: the leave-one-out out-of-sample SD when available
        # (honest width; the in-sample fit residual is too narrow),
        # otherwise the in-sample kernel-weighted residual SD.
        mult = SPREAD_MULT
        if loo is not None:
            sd = loo[h]
            mult = LOO_SPREAD_MULT
        qs = []
        for z in zs:
            q = math.exp(lp + med_r + z * sd * mult)
            qs.append(min(1e9, max(1.0, q)))
        for i in range(1, 5):
            qs[i] = max(qs[i], qs[i - 1])
        out.append({"hour": h, "q": [round2(q) for q in qs]})
    return out


def persistence_hours(p1, hours):
    """House-baseline fallback: flat at hour 1, random-walk spread."""
    out = []
    for h in hours:
        qs = [round2(min(1e9, max(1.0, p1 * math.exp(z * 0.15 * math.sqrt(h - 1)))))
              for z in Z]
        for i in range(1, 5):
            qs[i] = max(qs[i], qs[i - 1])
        out.append({"hour": h, "q": qs})
    return out


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    zs = [Z_BY_LEVEL.get(lv) for lv in levels]
    if any(z is None for z in zs):
        levels, zs = LEVELS, Z
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    hours = [row["hour"] for row in case.get("forecast") or [{"hour": h} for h in HOURS]]
    if not hours:
        hours = HOURS

    parsed = []
    for comp in case.get("comparables") or []:
        pc = parse_comparable(comp)
        if pc is not None:
            parsed.append(pc)
    # Extra training points: the 4 practice cases (public past launches,
    # not in the kit's comparables list). The target itself is excluded
    # by name -- a prediction never trains on the case it scores.
    target_name = (case.get("game") or {}).get("name") or ""
    for rec in practice_comparable_records():
        if rec["name"] == target_name:
            continue
        pc = parse_comparable(rec)
        if pc is not None:
            parsed.append(pc)
    parsed.sort(key=lambda c: c["name"])
    clean = [c for c in parsed if c["tier"] in (None, "comparable")]
    use = clean if len(clean) >= 5 else parsed

    if len(use) >= 3 and set(hours) <= set(HOURS):
        day = _dt.datetime.fromisoformat(
            (case.get("day_start") or "")[:10]).date()
        target_jump_h = us_east_midnight_hour(day)
        return {"hours": forecast_hours(p1, use, target_jump_h, zs)}
    return {"hours": persistence_hours(p1, hours)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
