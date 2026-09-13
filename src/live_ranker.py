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
RANK_N = 30

UPBIT = "https://api.upbit.com"

STABLE_TICKERS = {
    "USDT", "USDC", "DAI", "USD1", "USDE", "FDUSD", "TUSD",
    "RLUSD", "EURC"
}

def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def dt(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))

def ticker_symbol(market):
    return str(market).split("-")[-1].upper()

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

def signal_strength(r):
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

def best_strength(rows):
    return max((signal_strength(r) for r in rows), default=0.0)

def calc_potential(rs, latest, now, cur):
    first = rs[0]
    first_price = f(first.get("price"))
    latest_price = f(latest.get("price"))

    last1 = [r for r in rs if r["_dt"] >= now - timedelta(hours=1)]
    last3 = [r for r in rs if r["_dt"] >= now - timedelta(hours=3)]
    last6 = [r for r in rs if r["_dt"] >= now - timedelta(hours=6)]
    last12 = [r for r in rs if r["_dt"] >= now - timedelta(hours=12)]
    prev6_24 = [
        r for r in rs
        if now - timedelta(hours=24) <= r["_dt"] < now - timedelta(hours=6)
    ]

    b1 = best_strength(last1)
    b3 = best_strength(last3)
    b6 = best_strength(last6)
    b12 = best_strength(last12)
    bprev = best_strength(prev6_24)

    episode_n = len(rs)
    strong85_n = sum(f(r.get("score")) >= 85 for r in rs)
    strong90_n = sum(f(r.get("score")) >= 90 for r in rs)
    abc_all_n = sum(bool(r.get("ABC_ALL")) for r in rs)
    overlap2_n = sum(abc_count(r) >= 2 for r in rs)

    last3_90 = sum(f(r.get("score")) >= 90 for r in last3)
    last3_overlap2 = sum(abc_count(r) >= 2 for r in last3)
    last3_abc = sum(bool(r.get("ABC_ALL")) for r in last3)

    latest_score = f(latest.get("score"))
    latest_overlap = abc_count(latest)
    latest_abc = bool(latest.get("ABC_ALL"))
    latest_a30 = f(latest.get("value_accel_30m"), 1)

    older_a30s = [f(r.get("value_accel_30m"), 1) for r in rs[:-1]]
    older_med_a30 = statistics.median(older_a30s) if older_a30s else 1.0
    a30_reaccel = latest_a30 / max(older_med_a30, 0.25)

    latest_age_h = (now - latest["_dt"]).total_seconds() / 3600
    span_h = (latest["_dt"] - first["_dt"]).total_seconds() / 3600

    from_first_pct = (cur / first_price - 1) * 100 if first_price > 0 else 0
    from_latest_pct = (cur / latest_price - 1) * 100 if latest_price > 0 else 0

    max_strength = max(signal_strength(r) for r in rs)
    max_score = max(f(r.get("score")) for r in rs)
    max_a30 = max(f(r.get("value_accel_30m"), 1) for r in rs)

    base = 0.0
    base += clamp(max_strength / 100, 0, 1) * 10
    base += clamp(math.log1p(episode_n) / math.log(11), 0, 1) * 7
    base += clamp(strong90_n / 5, 0, 1) * 4
    base += clamp(abc_all_n / 4, 0, 1) * 4
    base += clamp(overlap2_n / 6, 0, 1) * 4

    reignition_delta = b3 - bprev
    base += clamp(b1 / 100, 0, 1) * 13
    base += clamp(b3 / 100, 0, 1) * 13
    base += clamp(b6 / 100, 0, 1) * 6
    base += clamp((reignition_delta + 10) / 35, 0, 1) * 8
    base += clamp(last3_90 / 2, 0, 1) * 4
    base += clamp(last3_overlap2 / 2, 0, 1) * 4
    base += clamp(last3_abc / 2, 0, 1) * 5

    base += clamp((latest_score - 70) / 25, 0, 1) * 7
    base += clamp((latest_a30 - 1) / 4, 0, 1) * 5
    base += clamp((a30_reaccel - 0.8) / 2.7, 0, 1) * 4
    base += latest_overlap * 1.5
    if latest_abc:
        base += 3

    penalty = 0.0

    if from_latest_pct >= 20:
        penalty += 40
    elif from_latest_pct >= 12:
        penalty += 25
    elif from_latest_pct >= 7:
        penalty += 14
    elif from_latest_pct >= 4:
        penalty += 6

    if from_first_pct >= 25:
        penalty += 20
    elif from_first_pct >= 15:
        penalty += 12
    elif from_first_pct >= 8:
        penalty += 6

    if latest_score < 60:
        penalty += 16
    elif latest_score < 70:
        penalty += 9
    elif latest_score < 80:
        penalty += 3

    if latest_overlap == 0:
        penalty += 8
    elif latest_overlap == 1:
        penalty += 2

    if latest_a30 < 1:
        penalty += 8
    elif latest_a30 < 1.5:
        penalty += 5
    elif latest_a30 < 2:
        penalty += 2

    if not last3:
        penalty += 14
    elif b3 < 65:
        penalty += 10
    elif b3 < 75:
        penalty += 5

    if latest_age_h >= 12:
        penalty += 16
    elif latest_age_h >= 6:
        penalty += 9
    elif latest_age_h >= 3:
        penalty += 4

    if latest.get("label") == "CHASE":
        penalty += 25

    potential = round(clamp(base - penalty, 0, 100), 2)

    return {
        "potential_score": potential,
        "episode_n_72h": episode_n,
        "strong85_n_72h": strong85_n,
        "strong90_n_72h": strong90_n,
        "abc_all_n_72h": abc_all_n,
        "overlap2_n_72h": overlap2_n,
        "best_strength_1h": round(b1, 2),
        "best_strength_3h": round(b3, 2),
        "best_strength_6h": round(b6, 2),
        "best_strength_12h": round(b12, 2),
        "prev6_24h_best_strength": round(bprev, 2),
        "reignition_delta_3h_vs_prev": round(reignition_delta, 2),
        "max_score_72h": round(max_score, 2),
        "max_a30_72h": round(max_a30, 3),
        "latest_signal_age_h": round(latest_age_h, 2),
        "signal_span_h": round(span_h, 2),
        "first_signal_price": first_price,
        "latest_signal_price": latest_price,
        "price_from_first_pct": round(from_first_pct, 3),
        "price_from_latest_pct": round(from_latest_pct, 3),
        "older_median_a30": round(older_med_a30, 3),
        "a30_reaccel_ratio": round(a30_reaccel, 3),
    }

def calc_launch(latest):
    """
    실제 '발사 시작' 확인 점수.
    거래량만 강한 코인보다 5/15/30/60분 가격 모멘텀이 동시에 퍼지는 코인을 우선한다.
    """
    r5 = f(latest.get("ret_5m"))
    r15 = f(latest.get("ret_15m"))
    r30 = f(latest.get("ret_30m"))
    r60 = f(latest.get("ret_60m"))

    latest_score = f(latest.get("score"))
    a15 = f(latest.get("value_accel_15m"), 1)
    a30 = f(latest.get("value_accel_30m"), 1)
    vr = f(latest.get("value_ratio_5m"), 1)
    overlap = abc_count(latest)

    rets = (r5, r15, r30, r60)
    positives = sum(x > 0 for x in rets)

    launch = 0.0

    launch += positives * 9
    launch += clamp(r5 / 0.8, 0, 1) * 8
    launch += clamp(r15 / 1.2, 0, 1) * 11
    launch += clamp(r30 / 2.0, 0, 1) * 14
    launch += clamp(r60 / 3.5, 0, 1) * 16

    if r60 > r30 > 0:
        launch += 5
    if r30 > r15 > 0:
        launch += 4
    if r15 > r5 > 0:
        launch += 2

    if positives >= 3:
        launch += clamp((a30 - 1) / 4, 0, 1) * 4
        launch += clamp((a15 - 1) / 6, 0, 1) * 3
        launch += clamp((vr - 1) / 8, 0, 1) * 3
        launch += clamp((latest_score - 80) / 15, 0, 1) * 3
        launch += min(overlap, 2) * 1.5

    # 거래량만 강하고 가격이 아직 안 움직이면 Launch 진입 제한
    if positives <= 1:
        launch = min(launch, 20)
    elif positives == 2:
        launch = min(launch, 38)

    if r15 <= 0 or r30 <= 0:
        launch = min(launch, 35)

    if r60 <= 0:
        launch = min(launch, 45)

    return round(clamp(launch, 0, 100), 2)

def final_stage(potential, launch, latest, cur, latest_signal_price):
    r60 = f(latest.get("ret_60m"))
    latest_label = latest.get("label")

    multiplier = 0.55 + 0.45 * (launch / 100.0)
    final_score = potential * multiplier

    from_latest_pct = 0.0
    if latest_signal_price > 0:
        from_latest_pct = (cur / latest_signal_price - 1) * 100

    chase_penalty = 0.0
    if latest_label == "CHASE":
        chase_penalty += 18
    if from_latest_pct >= 10:
        chase_penalty += 18
    elif from_latest_pct >= 6:
        chase_penalty += 10
    elif from_latest_pct >= 3:
        chase_penalty += 4
    if r60 >= 8:
        chase_penalty += 12
    elif r60 >= 5:
        chase_penalty += 6

    final_score = round(clamp(final_score - chase_penalty, 0, 100), 2)

    if potential >= 70 and launch >= 55 and latest_label != "CHASE":
        stage = "LAUNCH"
    elif potential >= 72 and launch >= 30:
        stage = "IGNITION"
    elif potential >= 65:
        stage = "POTENTIAL"
    elif potential >= 50:
        stage = "SLEEPER"
    else:
        stage = "WATCH"

    return final_score, stage

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
        market = r.get("market")
        if not market:
            continue
        if ticker_symbol(market) in STABLE_TICKERS:
            continue
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
        by_market[r["market"]].append(r)

    markets = sorted(by_market)
    tick = current_tickers(markets)

    ranked = []

    for market, rs in by_market.items():
        rs.sort(key=lambda x: x["_dt"])
        latest = rs[-1]

        t = tick.get(market)
        if not t or t["price"] <= 0:
            continue

        cur = t["price"]
        latest_signal_price = f(latest.get("price"))
        if latest_signal_price <= 0:
            continue

        pot = calc_potential(rs, latest, now, cur)
        launch = calc_launch(latest)
        final_score, stage = final_stage(
            pot["potential_score"], launch, latest, cur, latest_signal_price
        )

        ranked.append({
            "market": market,
            "final_score": final_score,
            "stage": stage,
            "potential_score": pot["potential_score"],
            "launch_score": launch,
            "current_price": cur,
            "change_rate_24h": round(t["change_rate_24h"], 3),
            "trade_value_24h": round(t["trade_value_24h"], 2),
            **pot,
            "latest_signal_ts": latest.get("ts"),
            "latest_score": round(f(latest.get("score")), 2),
            "latest_label": latest.get("label"),
            "latest_ret5": round(f(latest.get("ret_5m")), 3),
            "latest_ret15": round(f(latest.get("ret_15m")), 3),
            "latest_ret30": round(f(latest.get("ret_30m")), 3),
            "latest_ret60": round(f(latest.get("ret_60m")), 3),
            "latest_A": bool(latest.get("A")),
            "latest_B": bool(latest.get("B")),
            "latest_C": bool(latest.get("C")),
            "latest_ABC_ALL": bool(latest.get("ABC_ALL")),
            "latest_value_ratio_5m": round(f(latest.get("value_ratio_5m")), 3),
            "latest_a15": round(f(latest.get("value_accel_15m")), 3),
            "latest_a30": round(f(latest.get("value_accel_30m")), 3),
        })

    stage_priority = {"LAUNCH": 0, "IGNITION": 1, "POTENTIAL": 2, "SLEEPER": 3, "WATCH": 4}

    ranked.sort(key=lambda x: (stage_priority.get(x["stage"], 9), -x["final_score"]))

    # v0.4 핵심: Potential과 Launch를 완전히 분리

    # 아직 크게 안 갔지만 수급/축적이 강한 후보
    potential_candidates = [
        x for x in ranked
        if x["potential_score"] >= 60
        and x["latest_label"] != "CHASE"
        and x["change_rate_24h"] < 12
        and x["price_from_latest_pct"] < 6
        and x["price_from_first_pct"] < 15
    ]
    potential_candidates.sort(
        key=lambda x: (
            -x["potential_score"],
            -x["latest_score"],
            -x["latest_a30"],
            x["latest_signal_age_h"],
        )
    )
    potential_top5 = potential_candidates[:TOP_N]

    # 잠재력 + 실제 가격확인이 동시에 있는 후보
    # 5개를 억지로 채우지 않는다.
    launch_candidates = [
        x for x in ranked
        if x["potential_score"] >= 55
        and x["launch_score"] >= 35
        and x["latest_label"] != "CHASE"
        and x["change_rate_24h"] < 15
        and x["price_from_latest_pct"] < 7
    ]
    launch_candidates.sort(
        key=lambda x: (
            -x["launch_score"],
            -x["potential_score"],
            -x["latest_score"],
        )
    )
    launch_top5 = launch_candidates[:TOP_N]

    ignition_watch = sorted(
        [
            x for x in ranked
            if x["potential_score"] >= 65
            and 20 <= x["launch_score"] < 35
            and x["latest_label"] != "CHASE"
            and x["change_rate_24h"] < 15
        ],
        key=lambda x: (-x["potential_score"], -x["launch_score"])
    )[:10]

    sleeper_watch = sorted(
        [
            x for x in ranked
            if x["potential_score"] >= 55
            and x["launch_score"] < 20
            and x["latest_label"] != "CHASE"
        ],
        key=lambda x: (-x["potential_score"], x["latest_signal_age_h"])
    )[:10]

    payload = {
        "version": "Live Ranker v0.4",
        "generated_at_utc": now.isoformat(),
        "lookback_hours": LOOKBACK_HOURS,
        "top_n": TOP_N,
        "method_note": (
            "실험용 비교점수이며 확률이 아님. v0.4는 Potential TOP5와 Launch TOP5를 완전히 분리한다. "
            "Potential은 72h 축적/재점화 후보, Launch는 5m/15m/30m/60m 실제 가격확인 후보다. "
            "거래량만 강하고 가격확인이 없는 종목은 Launch 점수를 강하게 제한한다."
        ),
        # 기존 대시보드 호환용: top5는 Launch TOP5와 동일
        "top5": launch_top5,
        "launch_top5": launch_top5,
        "potential_top5": potential_top5,
        "ignition_watch": ignition_watch,
        "sleeper_watch": sleeper_watch,
        "ranked_top30": ranked[:RANK_N],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("generated", OUT)

    print("=== LAUNCH TOP5 ===")
    if not launch_top5:
        print("No confirmed launch candidates")
    for i, x in enumerate(launch_top5, 1):
        print(
            i, x["market"],
            "stage=", x["stage"],
            "final=", x["final_score"],
            "potential=", x["potential_score"],
            "launch=", x["launch_score"],
            "ret=", (x["latest_ret5"], x["latest_ret15"], x["latest_ret30"], x["latest_ret60"])
        )

    print("=== POTENTIAL TOP5 ===")
    for i, x in enumerate(potential_top5, 1):
        print(
            i, x["market"],
            "potential=", x["potential_score"],
            "launch=", x["launch_score"],
            "ret=", (x["latest_ret5"], x["latest_ret15"], x["latest_ret30"], x["latest_ret60"])
        )

if __name__ == "__main__":
    main()
