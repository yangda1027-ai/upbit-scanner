#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast five-slot V7.6 replay using saved rank-history JSONL snapshots only.

No historical full-market Upbit scan is performed. Entry and exit crossings use
observed snapshot prices, not intrabar high/low, so results are approximate and
must not be presented as exact executable fills.
"""

import bisect
import json
import os
import random
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

HISTORY_DIR = Path(os.getenv("BT_HISTORY_DIR", "docs/data/v7_6_rank_history"))
OUT = Path(os.getenv("BT_OUT", "docs/data/v76_five_slots_snapshot_backtest_latest.json"))
STEP_HOURS = int(os.getenv("BT_STEP_HOURS", "6"))
HOLD_HOURS = int(os.getenv("BT_HOLD_HOURS", "168"))
TARGET_PCT = float(os.getenv("BT_TARGET_PCT", "20"))
STOP_PCTS = tuple(float(x) for x in os.getenv("BT_STOP_PCTS", "3,5,7,10").split(","))
REPEAT_HOURS = int(os.getenv("BT_REPEAT_HOURS", "72"))
NEW_HOURS = int(os.getenv("BT_NEW_HOURS", "24"))
EPISODE_GAP_MIN = int(os.getenv("BT_EPISODE_GAP_MIN", "60"))
NOT_PUMPED_RET3_MAX = float(os.getenv("BT_NOT_PUMPED_RET3_MAX", "5"))
NOT_PUMPED_RET60_MAX = float(os.getenv("BT_NOT_PUMPED_RET60_MAX", "2"))
ASTR_ACCEL15_MIN = float(os.getenv("BT_ASTR_ACCEL15_MIN", "5"))
ASTR_VALUE5_MIN = float(os.getenv("BT_ASTR_VALUE5_MIN", "5"))
OUTLIER_SCORE_MIN = float(os.getenv("BT_OUTLIER_SCORE_MIN", "80"))
MIN_EXIT_COVERAGE = float(os.getenv("BT_MIN_EXIT_COVERAGE", "0.90"))
RANDOM_SEED = int(os.getenv("BT_RANDOM_SEED", "7605"))
UTC = timezone.utc


def parse_ts(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def pct(a, b):
    return (a / b - 1) * 100 if b else 0.0


def load_snapshots():
    snapshots, bad_lines = [], 0
    for path in sorted(HISTORY_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    ts = parse_ts(obj["ts"])
                    rows = obj.get("rows") or []
                    snapshots.append({"ts": ts, "rows": rows, "source_file": path.name, "source_line": line_no})
                except Exception:
                    bad_lines += 1
    snapshots.sort(key=lambda x: x["ts"])
    # Guard against a workflow retry appending the same timestamp twice.
    deduped = []
    for snap in snapshots:
        if deduped and snap["ts"] == deduped[-1]["ts"]:
            if len(snap["rows"]) >= len(deduped[-1]["rows"]):
                deduped[-1] = snap
        else:
            deduped.append(snap)
    return deduped, bad_lines


def select_checkpoints(snapshots):
    if not snapshots:
        return []
    chosen = [snapshots[0]]
    next_at = snapshots[0]["ts"] + timedelta(hours=STEP_HOURS)
    for snap in snapshots[1:]:
        if snap["ts"] >= next_at:
            chosen.append(snap)
            next_at = snap["ts"] + timedelta(hours=STEP_HOURS)
    return chosen


def normalize_rows(rows):
    priority = {"EARLY": 0, "WATCH": 1, "CHASE": 2}
    out = []
    for z0 in rows:
        z = dict(z0)
        for key in ("score", "price", "ret_5m", "ret_15m", "ret_30m", "ret_60m", "value_ratio_5m", "value_accel_15m", "value_accel_30m"):
            try:
                z[key] = float(z.get(key, 0))
            except Exception:
                z[key] = 0.0
        z["A"] = bool(z.get("A")); z["B"] = bool(z.get("B")); z["C"] = bool(z.get("C"))
        z["ABC_ALL"] = bool(z.get("ABC_ALL", z["A"] and z["B"] and z["C"]))
        if z.get("market") and z["price"] > 0:
            out.append(z)
    out.sort(key=lambda z: (priority.get(z.get("label"), 9), -z["score"], -z["value_accel_15m"]))
    for rank, z in enumerate(out, 1):
        z["rank"] = rank
    return out


def historical_return(price_history, market, now, current, hours=72):
    target = now - timedelta(hours=hours)
    items = price_history[market]
    if not items:
        return None
    best = min(items, key=lambda x: abs((x[0] - target).total_seconds()))
    if abs((best[0] - target).total_seconds()) > STEP_HOURS * 3600 * 0.75:
        return None
    return round(pct(current, best[1]), 3)


def not_pumped(z):
    r3 = z.get("ret_3d")
    return z["ret_60m"] <= NOT_PUMPED_RET60_MAX and (r3 is None or r3 < NOT_PUMPED_RET3_MAX)


def choose_slots(ranked, episodes, at):
    used, slots = set(), []

    def add(name, pool, key, reason):
        choices = [z for z in pool if z["market"] not in used]
        z = sorted(choices, key=key)[0] if choices else None
        if z:
            used.add(z["market"])
        slots.append({"slot": name, "status": "selected" if z else "empty", "reason": reason if z else "No market met the preregistered conditions", "candidate": dict(z) if z else None})

    early = [z for z in ranked if z.get("label") == "EARLY"]
    add("1_TOP_EARLY", early, lambda z: (-z["score"], -z["value_accel_15m"]), "Highest V7.6 EARLY")
    add("2_FF_ABC_ALL", [z for z in ranked if z["ABC_ALL"] and not_pumped(z)], lambda z: (-z["value_accel_15m"], -z["score"]), "ABC_ALL while price remains unextended")
    astr = [z for z in ranked if 4 <= z["rank"] <= 30 and z.get("label") == "EARLY" and not_pumped(z) and (z["value_accel_15m"] >= ASTR_ACCEL15_MIN or z["value_ratio_5m"] >= ASTR_VALUE5_MIN)]
    add("3_ASTR_RANK4_30", astr, lambda z: (-max(z["value_accel_15m"] / ASTR_ACCEL15_MIN, z["value_ratio_5m"] / ASTR_VALUE5_MIN), -z["score"]), "Rank 4-30 with extreme trading-value acceleration")
    repeat = []
    cutoff = at - timedelta(hours=REPEAT_HOURS)
    for z in early:
        count = sum(ts >= cutoff for ts in episodes[z["market"]])
        z["early_episodes_72h"] = count
        if count >= 2 and not_pumped(z):
            repeat.append(z)
    add("4_REPEAT_EARLY", repeat, lambda z: (-z["early_episodes_72h"], -z["score"]), "At least two independent EARLY episodes in 72h without extension")
    new_cutoff = at - timedelta(hours=NEW_HOURS)
    outliers = []
    for z in ranked[30:]:
        prior = sum(ts >= new_cutoff for ts in episodes[z["market"]])
        strong = z.get("label") == "EARLY" and ((z["A"] or z["B"] or z["C"]) or z["score"] >= OUTLIER_SCORE_MIN)
        if strong and prior == 0 and not_pumped(z):
            z["prior_episodes_24h"] = 0
            outliers.append(z)
    add("5_NEW_OUTLIER_OUTSIDE30", outliers, lambda z: (-int(z["ABC_ALL"]), -sum((z["A"], z["B"], z["C"])), -z["score"]), "First strong EARLY/ABC episode outside TOP30")
    return slots


def build_market_series(snapshots):
    series = defaultdict(list)
    for snap in snapshots:
        ts = snap["ts"]
        for z in snap["rows"]:
            try:
                market, price = z.get("market"), float(z.get("price"))
                if market and price > 0:
                    series[market].append((ts, price))
            except Exception:
                pass
    return series


def replay_snapshot(series, market, entry_ts, entry, target_pct, stop_pct=None):
    items = series.get(market, [])
    times = [x[0] for x in items]
    start = bisect.bisect_right(times, entry_ts)
    end_ts = entry_ts + timedelta(hours=HOLD_HOURS)
    end = bisect.bisect_right(times, end_ts)
    path = items[start:end]
    if not path:
        return {"outcome": "insufficient_future", "return_pct": None, "coverage_pct": 0.0}
    tp = entry * (1 + target_pct / 100)
    sl = entry * (1 - stop_pct / 100) if stop_pct is not None else None
    max_ret, min_ret = -1e9, 1e9
    for ts, price in path:
        move = pct(price, entry)
        max_ret, min_ret = max(max_ret, move), min(min_ret, move)
        if sl is not None and price <= sl:
            return {"outcome": "stop_observed", "return_pct": -float(stop_pct), "exit_ts": ts.isoformat(), "mfe_pct": round(max_ret, 3), "mae_pct": round(min_ret, 3)}
        if price >= tp:
            return {"outcome": "target_observed", "return_pct": float(target_pct), "exit_ts": ts.isoformat(), "mfe_pct": round(max_ret, 3), "mae_pct": round(min_ret, 3)}
    coverage = min(1.0, max(0.0, (path[-1][0] - entry_ts).total_seconds() / (HOLD_HOURS * 3600)))
    if coverage < MIN_EXIT_COVERAGE:
        return {"outcome": "insufficient_future", "return_pct": None, "coverage_pct": round(coverage * 100, 2), "mfe_pct": round(max_ret, 3), "mae_pct": round(min_ret, 3)}
    return {"outcome": "timeout", "return_pct": round(pct(path[-1][1], entry), 3), "exit_ts": path[-1][0].isoformat(), "coverage_pct": round(coverage * 100, 2), "mfe_pct": round(max_ret, 3), "mae_pct": round(min_ret, 3)}


def summarize(records, key):
    vals = [r[key] for r in records if r.get(key) and r[key].get("return_pct") is not None]
    rets = [x["return_pct"] for x in vals]
    n = len(rets)
    return {"n_evaluable": n, "n_insufficient_future": sum(r.get(key, {}).get("outcome") == "insufficient_future" for r in records),
            "avg_return_pct": round(sum(rets) / n, 3) if n else None,
            "median_return_pct": round(sorted(rets)[n // 2], 3) if n else None,
            "win_rate_pct": round(100 * sum(x > 0 for x in rets) / n, 2) if n else None,
            "target_rate_pct": round(100 * sum(x["outcome"] == "target_observed" for x in vals) / n, 2) if n else None}


def main():
    snapshots, bad_lines = load_snapshots()
    if not snapshots:
        raise SystemExit(f"No JSONL snapshots found in {HISTORY_DIR}")
    checkpoints = select_checkpoints(snapshots)
    series = build_market_series(snapshots)
    episodes, price_history = defaultdict(deque), defaultdict(deque)
    records, diagnostics = [], []
    rng = random.Random(RANDOM_SEED)

    for index, snap in enumerate(checkpoints, 1):
        at = snap["ts"]
        ranked = normalize_rows(snap["rows"])
        for z in ranked:
            z["ret_3d"] = historical_return(price_history, z["market"], at, z["price"], 72)
        slots = choose_slots(ranked, episodes, at)
        picks = [s["candidate"] for s in slots if s["candidate"]]
        excluded = {z["market"] for z in picks}
        random_pool = [z for z in ranked if z.get("label") != "CHASE" and z["market"] not in excluded]
        random_picks = rng.sample(random_pool, min(len(picks), len(random_pool))) if picks else []

        for z, slot in [(s["candidate"], s["slot"]) for s in slots if s["candidate"]] + [(z, "RANDOM_CONTROL") for z in random_picks]:
            rec = {"decision_ts": at.isoformat(), "slot": slot, **z}
            rec["no_stop"] = replay_snapshot(series, z["market"], at, z["price"], TARGET_PCT, None)
            for stop in STOP_PCTS:
                rec[f"sl_{int(stop) if stop.is_integer() else stop}"] = replay_snapshot(series, z["market"], at, z["price"], TARGET_PCT, stop)
            records.append(rec)

        diagnostics.append({"decision_ts": at.isoformat(), "source_file": snap["source_file"], "source_line": snap["source_line"],
                            "scanned": len(ranked), "early": sum(z.get("label") == "EARLY" for z in ranked),
                            "abc_all": sum(z["ABC_ALL"] for z in ranked), "slots": slots})

        # Add independent episodes only after selection to avoid look-ahead.
        for z in ranked:
            market = z["market"]
            price_history[market].append((at, z["price"]))
            while price_history[market] and price_history[market][0][0] < at - timedelta(hours=78):
                price_history[market].popleft()
            strong = z.get("label") == "EARLY" or z["A"] or z["B"] or z["C"]
            if strong and (not episodes[market] or at - episodes[market][-1] >= timedelta(minutes=EPISODE_GAP_MIN)):
                episodes[market].append(at)
            while episodes[market] and episodes[market][0] < at - timedelta(hours=max(REPEAT_HOURS, NEW_HOURS)):
                episodes[market].popleft()
        print(f"replayed {index}/{len(checkpoints)} {at.isoformat()}")

    keys = ["no_stop"] + [f"sl_{int(x) if x.is_integer() else x}" for x in STOP_PCTS]
    names = ["ALL_FIVE_SLOTS", "1_TOP_EARLY", "2_FF_ABC_ALL", "3_ASTR_RANK4_30", "4_REPEAT_EARLY", "5_NEW_OUTLIER_OUTSIDE30", "RANDOM_CONTROL"]
    summary = {}
    for name in names:
        subset = [r for r in records if (r["slot"] != "RANDOM_CONTROL" if name == "ALL_FIVE_SLOTS" else r["slot"] == name)]
        summary[name] = {key: summarize(subset, key) for key in keys}

    result = {"version": "V76_FIVE_SLOTS_SNAPSHOT_FAST_V1", "status": "complete",
              "config": {"history_dir": str(HISTORY_DIR), "source_snapshots": len(snapshots), "checkpoints": len(checkpoints),
                         "first_snapshot": snapshots[0]["ts"].isoformat(), "last_snapshot": snapshots[-1]["ts"].isoformat(),
                         "step_hours": STEP_HOURS, "hold_hours": HOLD_HOURS, "target_pct": TARGET_PCT, "stop_pcts": list(STOP_PCTS),
                         "episode_gap_min": EPISODE_GAP_MIN, "bad_jsonl_lines": bad_lines,
                         "v76_integrity": "Consumes stored V7.6 rank/features without recalculating or tuning them.",
                         "measurement_warning": "Targets/stops use observed snapshot prices, not 5m candle high/low. Crossings between snapshots can be missed; results are approximate.",
                         "oos_warning": "Only snapshots accumulated prospectively after logging began are available. Do not infer six-month performance until six months have actually accumulated."},
              "summary": summary, "diagnostics": diagnostics, "records": records}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
