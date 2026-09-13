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

UPBIT = "https://api.upbit.com"

# 안정자산/스테이블 계열 제외
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


def strength(r):
    # 비교용 강도점수. 확률이 아님.
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
    return max((strength(r) for r in rows), default=0.0)


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

        latest_age_h = (now - latest["_dt"]).total_seconds() / 3600
        span_h = (latest["_dt"] - first["_dt"]).total_seconds() / 3600

        episode_n = len(rs)
        strong85_n = sum(f(r.get("score")) >= 85 for r in rs)
        strong90_n = sum(f(r.get("score")) >= 90 for r in rs)
        abc_all_n = sum(bool(r.get("ABC_ALL")) for r in rs)
        overlap2_n = sum(abc_count(r) >= 2 for r in rs)

        max_score = max(f(r.get("score")) for r in rs)
        max_a30 = max(f(r.get("value_accel_30m"), 1) for r in rs)
        max_strength = max(strength(r) for r in rs)

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

        last3_90 = sum(f(r.get("score")) >= 90 for r in last3)
        last3_overlap2 = sum(abc_count(r) >= 2 for r in last3)
        last3_abc = sum(bool(r.get("ABC_ALL")) for r in last3)

        last6_90 = sum(f(r.get("score")) >= 90 for r in last6)
        last6_overlap2 = sum(abc_count(r) >= 2 for r in last6)
        last6_abc = sum(bool(r.get("ABC_ALL")) for r in last6)

        latest_score = f(latest.get("score"))
        latest_overlap = abc_count(latest)
        latest_abc = bool(latest.get("ABC_ALL"))
        latest_a15 = f(latest.get("value_accel_15m"), 1)
        latest_a30 = f(latest.get("value_accel_30m"), 1)
        latest_vr = f(latest.get("value_ratio_5m"), 1)

        older_a30s = [f(r.get("value_accel_30m"), 1) for r in rs[:-1]]
        older_med_a30 = statistics.median(older_a30s) if older_a30s else 1.0
        a30_reaccel = latest_a30 / max(older_med_a30, 0.25)

        from_first_pct = (cur / first_price - 1) * 100
        from_latest_pct = (cur / latest_price - 1) * 100

        # v0.2 핵심:
        # 과거 이력보다 "지금 1~3시간 안에 다시 강해졌는가"에 더 큰 가중치.
        base = 0.0

        # 72h 누적 기반
        base += clamp(max_strength / 100, 0, 1) * 12
        base += clamp(math.log1p(episode_n) / math.log(11), 0, 1) * 8
        base += clamp(strong90_n / 5, 0, 1) * 5
        base += clamp(abc_all_n / 4, 0, 1) * 5
        base += clamp(overlap2_n / 6, 0, 1) * 5

        # 현재 1~3h 재점화
        base += clamp(b1 / 100, 0, 1) * 16
        base += clamp(b3 / 100, 0, 1) * 16
        base += clamp(b6 / 100, 0, 1) * 7

        # 최근 강도 상승폭
        reignition_delta = b3 - bprev
        base += clamp((reignition_delta + 10) / 35, 0, 1) * 8

        # 최근 겹침
        base += clamp(last3_90 / 2, 0, 1) * 5
        base += clamp(last3_overlap2 / 2, 0, 1) * 5
        base += clamp(last3_abc / 2, 0, 1) * 6

        # 최신 순간값
        base += clamp((latest_score - 70) / 25, 0, 1) * 6
        base += clamp((latest_a30 - 1) / 4, 0, 1) * 5
        base += clamp((a30_reaccel - 0.8) / 2.7, 0, 1) * 4
        base += latest_overlap * 1.5
        if latest_abc:
            base += 3

        penalty = 0.0

        # 이미 오른 종목 추격 방지
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

        # 최신 신호가 약하면 강하게 감점
        if latest_score < 60:
            penalty += 18
        elif latest_score < 70:
            penalty += 10
        elif latest_score < 80:
            penalty += 4

        if latest_overlap == 0:
            penalty += 10
        elif latest_overlap == 1:
            penalty += 3

        if latest_a30 < 1:
            penalty += 10
        elif latest_a30 < 1.5:
            penalty += 6
        elif latest_a30 < 2:
            penalty += 3

        # 최근 3시간에 유효 재점화가 없으면 감점
        if not last3:
            penalty += 16
        elif b3 < 65:
            penalty += 12
        elif b3 < 75:
            penalty += 6

        # 마지막 신호가 오래됐으면 감점
        if latest_age_h >= 12:
            penalty += 18
        elif latest_age_h >= 6:
            penalty += 10
        elif latest_age_h >= 3:
            penalty += 5

        if latest.get("label") == "CHASE":
            penalty += 25

        live_score = round(clamp(base - penalty, 0, 100), 2)

        # 상태 판정도 최신성 중심
        if (
            live_score >= 72
            and latest_score >= 80
            and latest_overlap >= 2
            and latest_a30 >= 2
            and latest_age_h <= 3
            and from_latest_pct < 7
        ):
            state = "REIGNITION"
        elif (
            live_score >= 65
            and latest_score >= 75
            and latest_age_h <= 3
            and from_latest_pct < 7
        ):
            state = "HOT"
        elif (
            episode_n >= 3
            and from_first_pct < 8
            and latest_age_h <= 12
        ):
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

            "last1h_signal_n": len(last1),
            "last3h_signal_n": len(last3),
            "last6h_signal_n": len(last6),

            "last3h_score90_n": last3_90,
            "last3h_overlap2_n": last3_overlap2,
            "last3h_abc_all_n": last3_abc,

            "last6h_score90_n": last6_90,
            "last6h_overlap2_n": last6_overlap2,
            "last6h_abc_all_n": last6_abc,

            "best_strength_1h": round(b1, 2),
            "best_strength_3h": round(b3, 2),
            "best_strength_6h": round(b6, 2),
            "best_strength_12h": round(b12, 2),
            "prev6_24h_best_strength": round(bprev, 2),
            "reignition_delta_3h_vs_prev": round(reignition_delta, 2),

            "max_score_72h": round(max_score, 2),
            "max_a30_72h": round(max_a30, 3),

            "latest_signal_ts": latest.get("ts"),
            "latest_signal_age_h": round(latest_age_h, 2),
            "signal_span_h": round(span_h, 2),

            "first_signal_price": first_price,
            "latest_signal_price": latest_price,
            "price_from_first_pct": round(from_first_pct, 3),
            "price_from_latest_pct": round(from_latest_pct, 3),

            "latest_score": round(latest_score, 2),
            "latest_label": latest.get("label"),
            "latest_A": bool(latest.get("A")),
            "latest_B": bool(latest.get("B")),
            "latest_C": bool(latest.get("C")),
            "latest_ABC_ALL": latest_abc,
            "latest_value_ratio_5m": round(latest_vr, 3),
            "latest_a15": round(latest_a15, 3),
            "latest_a30": round(latest_a30, 3),
            "older_median_a30": round(older_med_a30, 3),
            "a30_reaccel_ratio": round(a30_reaccel, 3),
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
        "version": "Live Ranker v0.2",
        "generated_at_utc": now.isoformat(),
        "lookback_hours": LOOKBACK_HOURS,
        "top_n": TOP_N,
        "method_note": (
            "실험용 비교점수이며 확률이 아님. v0.2는 72h 이력보다 최근 1~3h 재점화를 더 중시하고, "
            "최신 신호 score/ABC중첩/a30가 약하면 강하게 감점. 스테이블 계열 제외."
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
            "latest_score=", x["latest_score"],
            "overlap=", int(x["latest_A"]) + int(x["latest_B"]) + int(x["latest_C"]),
            "a30=", x["latest_a30"],
            "age_h=", x["latest_signal_age_h"],
            "px_latest=", x["price_from_latest_pct"],
        )


if __name__ == "__main__":
    main()
