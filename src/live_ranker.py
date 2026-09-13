import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

SIGNALS = Path("docs/data/v7_6_signals.json")
OUT = Path("docs/data/live_ranker.json")

LOOKBACK_HOURS = 72
TOP_N = 5
RANK_N = 20
STALE_HOURS = 18
VERY_STALE_HOURS = 36

UPBIT = "https://api.upbit.com"


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def dt(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def get(path, params=None):
    r = requests.get(UPBIT + path, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def current_tickers(markets):
    out = {}
    for i in range(0, len(markets), 100):
        chunk = markets[i:i + 100]
        rows = get("/v1/ticker", {"markets": ",".join(chunk)})
        for x in rows:
            out[x["market"]] = {
                "price": f(x.get("trade_price")),
                "change_rate_24h": f(x.get("signed_change_rate")) * 100,
                "trade_value_24h": f(x.get("acc_trade_price_24h")),
            }
        time.sleep(0.12)
    return out


def abc_count(r):
    return int(bool(r.get("A"))) + int(bool(r.get("B"))) + int(bool(r.get("C")))


def strength(r):
    # ë¹êµì© ì í¸ê°ë. íë¥ ì´ ìë.
    s = f(r.get("score"))
    a15 = f(r.get("value_accel_15m"), 1)
    a30 = f(r.get("value_accel_30m"), 1)
    vr = f(r.get("value_ratio_5m"), 1)
    overlap = abc_count(r)

    x = 0.0
    x += clamp((s - 70) / 25, 0, 1) * 35
    x += clamp((a15 - 1) / 7, 0, 1) * 15
    x += clamp((a30 - 1) / 5, 0, 1) * 20
    x += clamp((vr - 1) / 9, 0, 1) * 10
    x += overlap * 4
    if r.get("ABC_ALL"):
        x += 8
    return clamp(x, 0, 100)


def main():
    now = datetime.now(timezone.utc)

    if not SIGNALS.exists():
        raise SystemExit(f"missing {SIGNALS}")

    rows = json.loads(SIGNALS.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("v7_6_signals.json must be a list")

    cutoff = now - timedelta(hours=LOOKBACK_HOURS)
    recent = []

    for r in rows:
        try:
            t = dt(r.get("ts"))
        except Exception:
            continue

        if t >= cutoff:
            q = dict(r)
            q["_dt"] = t
            recent.append(q)

    by_market = defaultdict(list)
    for r in recent:
        market = r.get("market")
        if market:
            by_market[market].append(r)

    markets = sorted(by_market)
    tick = current_tickers(markets)
    ranked = []

    for market, rs in by_market.items():
        rs.sort(key=lambda x: x["_dt"])
        first = rs[0]
        latest = rs[-1]

        t = tick.get(market)
        if not t or t["price"] <= 0:
            continue

        cur = t["price"]
        first_price = f(first.get("price"))
        latest_price = f(latest.get("price"))
        if first_price <= 0 or latest_price <= 0:
            continue

        age_h = (now - latest["_dt"]).total_seconds() / 3600
        span_h = (latest["_dt"] - first["_dt"]).total_seconds() / 3600

        episode_n = len(rs)
        strong85_n = sum(f(r.get("score")) >= 85 for r in rs)
        strong90_n = sum(f(r.get("score")) >= 90 for r in rs)
        abc_all_n = sum(bool(r.get("ABC_ALL")) for r in rs)
        overlap2_n = sum(abc_count(r) >= 2 for r in rs)

        max_score = max(f(r.get("score")) for r in rs)
        max_a30 = max(f(r.get("value_accel_30m"), 1) for r in rs)
        max_strength = max(strength(r) for r in rs)

        last12 = [r for r in rs if r["_dt"] >= now - timedelta(hours=12)]
        prev12_36 = [
            r for r in rs
            if now - timedelta(hours=36) <= r["_dt"] < now - timedelta(hours=12)
        ]

        last12_best = max((strength(r) for r in last12), default=0)
        prev_best = max((strength(r) for r in prev12_36), default=0)
        reignition_delta = last12_best - prev_best

        last12_90 = sum(f(r.get("score")) >= 90 for r in last12)
        last12_abc = sum(bool(r.get("ABC_ALL")) for r in last12)
        last12_overlap2 = sum(abc_count(r) >= 2 for r in last12)

        latest_a30 = f(latest.get("value_accel_30m"), 1)
        older_a30s = [f(r.get("value_accel_30m"), 1) for r in rs[:-1]]
        older_med_a30 = statistics.median(older_a30s) if older_a30s else 1.0
        a30_reaccel = latest_a30 / max(older_med_a30, 0.25)

        from_first_pct = (cur / first_price - 1) * 100
        from_latest_pct = (cur / latest_price - 1) * 100

        base = 0.0
        base += clamp(max_strength / 100, 0, 1) * 24
        base += clamp(math.log1p(episode_n) / math.log(11), 0, 1) * 12
        base += clamp(strong90_n / 4, 0, 1) * 7
        base += clamp(overlap2_n / 5, 0, 1) * 7
        base += clamp(abc_all_n / 3, 0, 1) * 7

        # ìµê·¼ ì¬ì í ê°ì¤ì¹ë¥¼ ë í¬ê² ë 
        base += clamp(last12_best / 100, 0, 1) * 13
        base += clamp((reignition_delta + 10) / 35, 0, 1) * 10
        base += clamp(last12_90 / 2, 0, 1) * 5
        base += clamp(last12_abc / 2, 0, 1) * 6
        base += clamp((a30_reaccel - 0.8) / 2.7, 0, 1) * 5

        penalty = 0.0

        # ì´ë¯¸ ê¸ë±í ì¢ëª© ì¶ê²© ë°©ì§
        if from_latest_pct >= 20:
            penalty += 35
        elif from_latest_pct >= 12:
            penalty += 22
        elif from_latest_pct >= 7:
            penalty += 12
        elif from_latest_pct >= 4:
            penalty += 5

        if from_first_pct >= 25:
            penalty += 18
        elif from_first_pct >= 15:
            penalty += 10
        elif from_first_pct >= 8:
            penalty += 4

        # ì¤ë ì í¸ë§ ëì¤ê³  ê°ê²©ë°ì ìì¼ë©´ ë¸íí ê°ì 
        if age_h >= VERY_STALE_HOURS and from_first_pct < 5:
            penalty += 18
        elif age_h >= STALE_HOURS and from_first_pct < 5:
            penalty += 9

        # ìµê·¼ 12ìê° ì¬ì íê° ì½íë©´ ê°ì 
        if last12_best < 65:
            penalty += 12
        elif last12_best < 75:
            penalty += 6

        if latest.get("label") == "CHASE":
            penalty += 25

        live_score = round(clamp(base - penalty, 0, 100), 2)

        if live_score >= 72 and last12_abc >= 1 and from_latest_pct < 7:
            state = "REIGNITION"
        elif live_score >= 65 and last12_best >= 75 and from_latest_pct < 7:
            state = "HOT"
        elif episode_n >= 3 and from_first_pct < 8:
            state = "SLEEPER"
        elif from_latest_pct >= 7:
            state = "TOO_LATE"
        else:
            state = "WATCH"

        ranked.append({
            "market": market,
            "live_score": live_score,
            "state": state,
            "current_price": cur,
            "change_rate_24h": round(t["change_rate_24h"], 3),
            "trade_value_24h": round(t["trade_value_24h"], 2),

            "episode_n_72h": episode_n,
            "strong85_n_72h": strong85_n,
            "strong90_n_72h": strong90_n,
            "abc_all_n_72h": abc_all_n,
            "overlap2_n_72h": overlap2_n,

            "last12h_signal_n": len(last12),
            "last12h_score90_n": last12_90,
            "last12h_abc_all_n": last12_abc,
            "last12h_overlap2_n": last12_overlap2,

            "max_score_72h": round(max_score, 2),
            "max_a30_72h": round(max_a30, 3),
            "latest_a30": round(latest_a30, 3),
            "older_median_a30": round(older_med_a30, 3),
            "a30_reaccel_ratio": round(a30_reaccel, 3),

            "last12h_best_strength": round(last12_best, 2),
            "prev12_36h_best_strength": round(prev_best, 2),
            "reignition_delta": round(reignition_delta, 2),

            "first_signal_ts": first.get("ts"),
            "latest_signal_ts": latest.get("ts"),
            "latest_signal_age_h": round(age_h, 2),
            "signal_span_h": round(span_h, 2),

            "first_signal_price": first_price,
            "latest_signal_price": latest_price,
            "price_from_first_pct": round(from_first_pct, 3),
            "price_from_latest_pct": round(from_latest_pct, 3),

            "latest_score": f(latest.get("score")),
            "latest_label": latest.get("label"),
            "latest_A": bool(latest.get("A")),
            "latest_B": bool(latest.get("B")),
            "latest_C": bool(latest.get("C")),
            "latest_ABC_ALL": bool(latest.get("ABC_ALL")),
            "latest_value_ratio_5m": f(latest.get("value_ratio_5m")),
            "latest_a15": f(latest.get("value_accel_15m")),
            "latest_a30": f(latest.get("value_accel_30m")),
        })

    state_priority = {
        "REIGNITION": 0,
        "HOT": 1,
        "SLEEPER": 2,
        "WATCH": 3,
        "TOO_LATE": 4,
    }

    ranked.sort(
        key=lambda x: (
            state_priority.get(x["state"], 9),
            -x["live_score"]
        )
    )

    eligible = [
        x for x in ranked
        if x["state"] in {"REIGNITION", "HOT", "SLEEPER"}
        and x["price_from_latest_pct"] < 7
        and x["change_rate_24h"] < 15
    ]
    eligible.sort(key=lambda x: -x["live_score"])

    top5 = eligible[:TOP_N]

    OUT.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "Live Ranker v0.1",
        "generated_at_utc": now.isoformat(),
        "lookback_hours": LOOKBACK_HOURS,
        "top_n": TOP_N,
        "method_note": (
            "ì¤íì© ë¹êµì ìì´ë©° íë¥ ì´ ìë. ìµê·¼72h ëë¦½ episode, ì í¸ê°ë, "
            "ìµê·¼12h ì¬ì í, ABC ì¤ì²©, a30 ì¬ìì¹, ê°ê²© ë¯¸ë°ìì ì°ì íê³  "
            "ì´ë¯¸ ê¸ë±/ì¤ëë ë¬´ë°ì ì í¸ë¥¼ ê°ì ."
        ),
        "top5": top5,
        "ranked_top20": ranked[:RANK_N],
    }

    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("generated", OUT)
    for i, x in enumerate(top5, 1):
        print(
            i,
            x["market"],
            x["live_score"],
            x["state"],
            "ep72=", x["episode_n_72h"],
            "abc12=", x["last12h_abc_all_n"],
            "reignite=", x["reignition_delta"],
            "px_latest=", x["price_from_latest_pct"],
        )


if __name__ == "__main__":
    main()
