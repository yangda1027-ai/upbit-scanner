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
DEBUG_ALL_KRW = True

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



def calc_response_quality(rs, latest, now):
    """
    v0.6: 반복 강신호가 실제 가격반응으로 이어졌는지 평가.
    같은 코인의 '강신호 횟수'를 무조건 가점하지 않고,
    후속 신호에서 가격확인이 없었던 누적을 failed ignition으로 감점한다.
    미래 캔들을 새로 조회하지 않고, 저장된 후속 V7.6 스냅샷만 사용한다.
    """
    prior = [r for r in rs[:-1] if r["_dt"] >= now - timedelta(hours=72)]
    strong = []
    for i, r in enumerate(prior):
        if f(r.get("score")) < 82:
            continue
        if abc_count(r) < 2 and not bool(r.get("ABC_ALL")):
            continue
        if max(
            f(r.get("value_ratio_5m"), 1),
            f(r.get("value_accel_15m"), 1),
            f(r.get("value_accel_30m"), 1),
        ) < 2:
            continue
        strong.append((i, r))

    successes = 0
    failures = 0
    evaluated = 0

    for i, r in strong:
        t0 = r["_dt"]
        # 저장된 후속 신호 중 30분~6시간 뒤의 가격확인을 본다.
        future = [
            q for q in rs
            if timedelta(minutes=30) <= (q["_dt"] - t0) <= timedelta(hours=6)
        ]
        if not future:
            continue
        evaluated += 1

        confirmed = False
        for q in future:
            rets = [
                f(q.get("ret_5m")),
                f(q.get("ret_15m")),
                f(q.get("ret_30m")),
                f(q.get("ret_60m")),
            ]
            positives = sum(x > 0 for x in rets)
            if positives >= 3 and max(rets[1:]) >= 1.0:
                confirmed = True
                break

        if confirmed:
            successes += 1
        else:
            failures += 1

    if evaluated:
        response_rate = successes / evaluated
    else:
        response_rate = 0.5  # 데이터 부족은 중립

    # 실패가 반복될수록 비선형 감점. 성공 이력은 소폭 회복.
    failure_penalty = min(30.0, failures * 4.0)
    if failures >= 4 and response_rate < 0.25:
        failure_penalty += 8.0
    failure_penalty = min(38.0, failure_penalty)

    response_bonus = min(10.0, successes * 2.0)

    return {
        "evaluated_ignitions_72h": evaluated,
        "successful_ignitions_72h": successes,
        "failed_ignitions_72h": failures,
        "ignition_response_rate": round(response_rate * 100, 2),
        "failed_ignition_penalty": round(failure_penalty, 2),
        "response_bonus": round(response_bonus, 2),
    }

def calc_transition(rs, latest, now, change_rate_24h):
    """
    v0.5 핵심: 잠복 -> 재점화 -> 가격확인 상태전이를 점수화.
    현재 한 장면이 아니라, 직전 72시간에 실제로 '가격 미반응 강신호'가 있었는지 본다.
    """
    prior = [r for r in rs[:-1] if r["_dt"] >= now - timedelta(hours=72)]
    prior6_72 = [r for r in prior if r["_dt"] < now - timedelta(hours=1)]

    def muted_precursor(r):
        score = f(r.get("score"))
        r15 = f(r.get("ret_15m"))
        r30 = f(r.get("ret_30m"))
        r60 = f(r.get("ret_60m"))
        a15 = f(r.get("value_accel_15m"), 1)
        a30 = f(r.get("value_accel_30m"), 1)
        vr = f(r.get("value_ratio_5m"), 1)
        overlap = abc_count(r)

        strong_flow = (
            score >= 82
            and (overlap >= 2 or bool(r.get("ABC_ALL")))
            and (a15 >= 2 or a30 >= 2 or vr >= 3)
        )
        price_not_gone = (
            -2.5 <= r15 <= 1.5
            and -2.5 <= r30 <= 1.5
            and r60 <= 2.0
        )
        return strong_flow and price_not_gone

    precursors = [r for r in prior6_72 if muted_precursor(r)]

    # 최근 24시간 재점화: 가격보다 거래량이 먼저 튄 기록
    reignitions = []
    for r in prior:
        age_h = (now - r["_dt"]).total_seconds() / 3600
        if age_h > 24:
            continue
        vr = f(r.get("value_ratio_5m"), 1)
        a15 = f(r.get("value_accel_15m"), 1)
        a30 = f(r.get("value_accel_30m"), 1)
        r15 = f(r.get("ret_15m"))
        r30 = f(r.get("ret_30m"))
        r60 = f(r.get("ret_60m"))
        if (
            f(r.get("score")) >= 82
            and (vr >= 8 or a15 >= 5 or a30 >= 3)
            and r15 <= 2.0 and r30 <= 2.0 and r60 <= 3.0
        ):
            reignitions.append(r)

    precursor_count = len(precursors)
    reignition_count = len(reignitions)

    precursor_best = max((signal_strength(r) for r in precursors), default=0.0)
    precursor_abc_n = sum(bool(r.get("ABC_ALL")) for r in precursors)

    max_prior_vr = max((f(r.get("value_ratio_5m"), 1) for r in reignitions), default=1.0)
    max_prior_a15 = max((f(r.get("value_accel_15m"), 1) for r in reignitions), default=1.0)
    max_prior_a30 = max((f(r.get("value_accel_30m"), 1) for r in reignitions), default=1.0)

    latest_launch = calc_launch(latest)

    # v0.6: 포화를 줄인 0~90 기본점수 + 최대 10 early bonus.
    # 극단적인 VR/a15 하나만으로 100에 붙지 않도록 로그형/완만한 상한을 사용.
    score = 0.0
    score += clamp(precursor_best / 110, 0, 1) * 12
    score += clamp(precursor_count / 6, 0, 1) * 8
    score += clamp(precursor_abc_n / 4, 0, 1) * 5

    score += clamp((max_prior_vr - 2) / 60, 0, 1) * 7
    score += clamp((max_prior_a15 - 1) / 20, 0, 1) * 7
    score += clamp((max_prior_a30 - 1) / 15, 0, 1) * 7
    score += clamp(reignition_count / 6, 0, 1) * 4

    score += clamp(latest_launch / 100, 0, 1) * 40

    # 너무 늦게 잡은 발사는 감점, 0~3%대 초기발사는 가점
    if -2 <= change_rate_24h <= 3:
        early_bonus = 10.0
    elif change_rate_24h <= 5:
        early_bonus = 6.0
    elif change_rate_24h <= 8:
        early_bonus = 2.0
    elif change_rate_24h <= 12:
        early_bonus = -6.0
    else:
        early_bonus = -15.0

    score += early_bonus

    # 전조 없이 현재만 갑자기 오른 경우는 낮춘다.
    if precursor_count == 0:
        score -= 18
    if reignition_count == 0:
        score -= 8

    return {
        "transition_score": round(clamp(score, 0, 100), 2),
        "precursor_n_72h": precursor_count,
        "precursor_abc_n_72h": precursor_abc_n,
        "precursor_best_strength": round(precursor_best, 2),
        "reignition_n_24h": reignition_count,
        "max_prior_value_ratio_5m": round(max_prior_vr, 3),
        "max_prior_a15": round(max_prior_a15, 3),
        "max_prior_a30": round(max_prior_a30, 3),
        "early_launch_bonus": round(early_bonus, 2),
    }

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

    # FULL PIPELINE DEBUG:
    # 전체 저장 데이터에서 market별 최신 기록/72h 기록을 따로 보존한다.
    all_signal_rows_by_market = defaultdict(list)
    malformed_ts_count = defaultdict(int)

    for r in rows:
        market = r.get("market")
        if not market:
            continue
        if ticker_symbol(market) in STABLE_TICKERS:
            continue
        try:
            t = dt(r.get("ts"))
        except Exception:
            malformed_ts_count[market] += 1
            continue

        q = dict(r)
        q["_dt"] = t
        all_signal_rows_by_market[market].append(q)

        if t >= cutoff:
            recent.append(q)

    by_market = defaultdict(list)
    for r in recent:
        by_market[r["market"]].append(r)

    # Upbit의 현재 전체 KRW 마켓을 받아 Debug의 시작점을 "전체 KRW"로 만든다.
    market_rows = get("/v1/market/all")
    all_krw_markets = sorted(
        x["market"] for x in market_rows
        if str(x.get("market", "")).startswith("KRW-")
        and ticker_symbol(x["market"]) not in STABLE_TICKERS
    )

    # 실제 랭킹 계산은 기존과 동일하게 최근 72h V7.6 신호가 있는 시장만 수행.
    scoring_markets = sorted(by_market)
    tick = current_tickers(all_krw_markets)

    ranked = []
    pipeline_debug = {}

    for market in all_krw_markets:
        hist = all_signal_rows_by_market.get(market, [])
        hist.sort(key=lambda x: x["_dt"])
        recent_rs = by_market.get(market, [])

        entry = {
            "market": market,
            "ticker": ticker_symbol(market),
            "in_upbit_krw_market": True,
            "stablecoin_excluded": False,
            "total_saved_signal_rows": len(hist),
            "recent_signal_rows_72h": len(recent_rs),
            "malformed_ts_rows": malformed_ts_count.get(market, 0),
            "pipeline_status": "PENDING",
            "pipeline_fail_stage": None,
            "pipeline_fail_reason": None,
        }

        if hist:
            entry["last_saved_signal_ts"] = hist[-1].get("ts")
            entry["last_saved_signal_age_h"] = round(
                (now - hist[-1]["_dt"]).total_seconds() / 3600, 2
            )
        else:
            entry["last_saved_signal_ts"] = None
            entry["last_saved_signal_age_h"] = None

        if not recent_rs:
            entry["pipeline_status"] = "EXCLUDED_BEFORE_SCORING"
            entry["pipeline_fail_stage"] = "RECENT_72H_SIGNAL_GATE"
            if hist:
                entry["pipeline_fail_reason"] = (
                    f"최근 {LOOKBACK_HOURS}시간 V7.6 저장 신호 없음; "
                    f"마지막 저장 신호 {entry['last_saved_signal_age_h']}시간 전"
                )
            else:
                entry["pipeline_fail_reason"] = (
                    "v7_6_signals.json에 이 KRW 종목의 저장 신호가 한 건도 없음"
                )
            pipeline_debug[market] = entry
            continue

        tcur = tick.get(market)
        if not tcur or tcur["price"] <= 0:
            entry["pipeline_status"] = "EXCLUDED_BEFORE_SCORING"
            entry["pipeline_fail_stage"] = "CURRENT_TICKER_GATE"
            entry["pipeline_fail_reason"] = "Upbit 현재 ticker/가격 조회 실패"
            pipeline_debug[market] = entry
            continue

        rs2 = sorted(recent_rs, key=lambda x: x["_dt"])
        latest2 = rs2[-1]
        latest_signal_price2 = f(latest2.get("price"))
        if latest_signal_price2 <= 0:
            entry["pipeline_status"] = "EXCLUDED_BEFORE_SCORING"
            entry["pipeline_fail_stage"] = "LATEST_SIGNAL_PRICE_GATE"
            entry["pipeline_fail_reason"] = "최근 V7.6 신호의 price가 없거나 0 이하"
            entry["latest_signal_ts"] = latest2.get("ts")
            pipeline_debug[market] = entry
            continue

        entry["pipeline_status"] = "ELIGIBLE_FOR_SCORING"
        entry["pipeline_fail_stage"] = None
        entry["pipeline_fail_reason"] = None
        entry["latest_signal_ts"] = latest2.get("ts")
        entry["latest_signal_price"] = latest_signal_price2
        entry["current_price"] = tcur["price"]
        pipeline_debug[market] = entry

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
        raw_potential_score = round(f(pot.get("potential_score")), 2)
        launch = calc_launch(latest)
        transition = calc_transition(rs, latest, now, t["change_rate_24h"])
        raw_transition_score = round(f(transition.get("transition_score")), 2)
        response = calc_response_quality(rs, latest, now)

        # 실패한 ignition 이력은 Potential/Transition에서 직접 차감.
        pot["potential_score"] = round(clamp(
            pot["potential_score"]
            - response["failed_ignition_penalty"] * 0.55
            + response["response_bonus"] * 0.35,
            0, 100
        ), 2)
        transition["transition_score"] = round(clamp(
            transition["transition_score"]
            - response["failed_ignition_penalty"] * 0.65
            + response["response_bonus"] * 0.50,
            0, 100
        ), 2)

        final_score, stage = final_stage(
            pot["potential_score"], launch, latest, cur, latest_signal_price
        )

        # v0.5 stage 재정의: Launch는 실제 가격확인 + 과거 전조가 동시에 있어야 함
        if (
            launch >= 55
            and transition["transition_score"] >= 50
            and pot["potential_score"] >= 55
            and response["failed_ignitions_72h"] <= 5
            and latest.get("label") != "CHASE"
        ):
            stage = "LAUNCH"
        elif (
            launch >= 30
            and transition["transition_score"] >= 45
            and pot["potential_score"] >= 60
        ):
            stage = "IGNITION"

        ranked.append({
            "market": market,
            "final_score": final_score,
            "stage": stage,
            "raw_potential_score": raw_potential_score,
            "potential_score": pot["potential_score"],
            "potential_adjustment": round(pot["potential_score"] - raw_potential_score, 2),
            "launch_score": launch,
            "raw_transition_score": raw_transition_score,
            "transition_score": transition["transition_score"],
            "transition_adjustment": round(transition["transition_score"] - raw_transition_score, 2),
            "current_price": cur,
            "change_rate_24h": round(t["change_rate_24h"], 3),
            "trade_value_24h": round(t["trade_value_24h"], 2),
            **pot,
            **transition,
            **response,
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

    # FULL PIPELINE DEBUG: 실제 점수 계산까지 도달한 종목의 결과를 연결한다.
    ranked_by_market = {x["market"]: x for x in ranked}
    for market, entry in pipeline_debug.items():
        x = ranked_by_market.get(market)
        if x is None:
            # recent signal gate는 통과했는데 ranked에 없다면 기존 scoring loop 내부 gate에서 빠진 것.
            if entry["pipeline_status"] == "ELIGIBLE_FOR_SCORING":
                entry["pipeline_status"] = "EXCLUDED_DURING_SCORING"
                entry["pipeline_fail_stage"] = "SCORING_LOOP_GATE"
                entry["pipeline_fail_reason"] = (
                    "최근 72h 신호는 있으나 기존 scoring loop에서 ranked 생성 전 탈락 "
                    "(현재 ticker 또는 latest signal price 조건 확인 필요)"
                )
            continue

        entry.update({
            "pipeline_status": "SCORED",
            "pipeline_fail_stage": None,
            "pipeline_fail_reason": None,
            "stage": x["stage"],
            "raw_potential_score": x["raw_potential_score"],
            "potential_score": x["potential_score"],
            "potential_adjustment": x["potential_adjustment"],
            "launch_score": x["launch_score"],
            "raw_transition_score": x["raw_transition_score"],
            "transition_score": x["transition_score"],
            "transition_adjustment": x["transition_adjustment"],
            "failed_ignitions_72h": x["failed_ignitions_72h"],
            "successful_ignitions_72h": x["successful_ignitions_72h"],
            "ignition_response_rate": x["ignition_response_rate"],
            "failed_ignition_penalty": x["failed_ignition_penalty"],
            "response_bonus": x["response_bonus"],
            "change_rate_24h": x["change_rate_24h"],
            "price_from_latest_pct": x["price_from_latest_pct"],
            "price_from_first_pct": x["price_from_first_pct"],
            "latest_label": x["latest_label"],
            "latest_ret5": x["latest_ret5"],
            "latest_ret15": x["latest_ret15"],
            "latest_ret30": x["latest_ret30"],
            "latest_ret60": x["latest_ret60"],
            "latest_score": x["latest_score"],
            "latest_signal_age_h": x["latest_signal_age_h"],
            "episode_n_72h": x["episode_n_72h"],
            "precursor_n_72h": x["precursor_n_72h"],
            "reignition_n_24h": x["reignition_n_24h"],
        })

    stage_priority = {"LAUNCH": 0, "IGNITION": 1, "POTENTIAL": 2, "SLEEPER": 3, "WATCH": 4}

    ranked.sort(key=lambda x: (stage_priority.get(x["stage"], 9), -x["final_score"]))

    # v0.5 핵심:
    # ① Potential = 아직 안 간 축적 후보
    # ② Launch = stage가 실제 LAUNCH이며 상태전이까지 확인된 후보
    # ③ Transition = CVC형 '잠복 -> 재점화 -> 가격확인' 우선

    potential_candidates = [
        x for x in ranked
        if x["potential_score"] >= 60
        and x["latest_label"] != "CHASE"
        and x["change_rate_24h"] < 12
        and x["price_from_latest_pct"] < 6
        and x["price_from_first_pct"] < 15
        and x["failed_ignition_penalty"] < 30
    ]
    potential_candidates.sort(
        key=lambda x: (
            -x["potential_score"],
            -x["transition_score"],
            -x["latest_score"],
            -x["latest_a30"],
            x["latest_signal_age_h"],
        )
    )
    potential_top5 = potential_candidates[:TOP_N]

    launch_candidates = [
        x for x in ranked
        if x["stage"] == "LAUNCH"
        and x["launch_score"] >= 55
        and x["transition_score"] >= 55
        and x["latest_label"] != "CHASE"
        and x["change_rate_24h"] < 12
        and x["price_from_latest_pct"] < 7
        and x["failed_ignition_penalty"] < 26
    ]

    # 상태전이 점수를 1순위, 그 다음 실제 Launch, 그 다음 Potential.
    # 이미 +8% 이상 간 종목은 early_launch_bonus로 자연스럽게 밀린다.
    launch_candidates.sort(
        key=lambda x: (
            -x["transition_score"],
            x["failed_ignitions_72h"],
            -x["ignition_response_rate"],
            -x["launch_score"],
            -x["potential_score"],
            x["change_rate_24h"],
        )
    )
    launch_top5 = launch_candidates[:TOP_N]

    ignition_watch = sorted(
        [
            x for x in ranked
            if x["stage"] == "IGNITION"
            and x["latest_label"] != "CHASE"
            and x["change_rate_24h"] < 12
        ],
        key=lambda x: (-x["transition_score"], -x["launch_score"], -x["potential_score"])
    )[:10]

    sleeper_watch = sorted(
        [
            x for x in ranked
            if x["potential_score"] >= 55
            and x["launch_score"] < 30
            and x["latest_label"] != "CHASE"
        ],
        key=lambda x: (-x["potential_score"], -x["transition_score"], x["latest_signal_age_h"])
    )[:10]

    # v0.6.1 DEBUG ONLY:
    # 점수/임계값/순위는 v0.6 그대로 유지하고, 왜 TOP5에서 빠졌는지만 기록한다.
    pot_rank = {x["market"]: i + 1 for i, x in enumerate(potential_candidates)}
    launch_rank = {x["market"]: i + 1 for i, x in enumerate(launch_candidates)}

    def potential_fail_reasons(x):
        reasons = []
        if x["potential_score"] < 60:
            reasons.append(f"potential<{60} ({x['potential_score']})")
        if x["latest_label"] == "CHASE":
            reasons.append("latest_label=CHASE")
        if x["change_rate_24h"] >= 12:
            reasons.append(f"24h>=12% ({x['change_rate_24h']}%)")
        if x["price_from_latest_pct"] >= 6:
            reasons.append(f"latest_signal_delta>=6% ({x['price_from_latest_pct']}%)")
        if x["price_from_first_pct"] >= 15:
            reasons.append(f"first_signal_delta>=15% ({x['price_from_first_pct']}%)")
        if x["failed_ignition_penalty"] >= 30:
            reasons.append(f"failed_penalty>=30 ({x['failed_ignition_penalty']})")
        return reasons

    def launch_fail_reasons(x):
        reasons = []
        if x["stage"] != "LAUNCH":
            reasons.append(f"stage={x['stage']}")
        if x["launch_score"] < 55:
            reasons.append(f"launch<55 ({x['launch_score']})")
        if x["transition_score"] < 55:
            reasons.append(f"transition<55 ({x['transition_score']})")
        if x["latest_label"] == "CHASE":
            reasons.append("latest_label=CHASE")
        if x["change_rate_24h"] >= 12:
            reasons.append(f"24h>=12% ({x['change_rate_24h']}%)")
        if x["price_from_latest_pct"] >= 7:
            reasons.append(f"latest_signal_delta>=7% ({x['price_from_latest_pct']}%)")
        if x["failed_ignition_penalty"] >= 26:
            reasons.append(f"failed_penalty>=26 ({x['failed_ignition_penalty']})")
        return reasons

    debug_rows = []
    for x in ranked:
        pr = pot_rank.get(x["market"])
        lr = launch_rank.get(x["market"])
        p_reasons = potential_fail_reasons(x)
        l_reasons = launch_fail_reasons(x)

        x["potential_candidate_rank"] = pr
        x["launch_candidate_rank"] = lr
        x["potential_fail_reasons"] = p_reasons
        x["launch_fail_reasons"] = l_reasons
        x["potential_status"] = (
            "TOP5" if pr is not None and pr <= TOP_N
            else ("PASSED_BUT_OUTSIDE_TOP5" if pr is not None else "FILTERED")
        )
        x["launch_status"] = (
            "TOP5" if lr is not None and lr <= TOP_N
            else ("PASSED_BUT_OUTSIDE_TOP5" if lr is not None else "FILTERED")
        )

        debug_rows.append({
            "market": x["market"],
            "stage": x["stage"],
            "raw_potential_score": x["raw_potential_score"],
            "potential_score": x["potential_score"],
            "potential_adjustment": x["potential_adjustment"],
            "launch_score": x["launch_score"],
            "raw_transition_score": x["raw_transition_score"],
            "transition_score": x["transition_score"],
            "transition_adjustment": x["transition_adjustment"],
            "failed_ignitions_72h": x["failed_ignitions_72h"],
            "successful_ignitions_72h": x["successful_ignitions_72h"],
            "ignition_response_rate": x["ignition_response_rate"],
            "failed_ignition_penalty": x["failed_ignition_penalty"],
            "response_bonus": x["response_bonus"],
            "potential_candidate_rank": pr,
            "launch_candidate_rank": lr,
            "potential_status": x["potential_status"],
            "launch_status": x["launch_status"],
            "potential_fail_reasons": p_reasons,
            "launch_fail_reasons": l_reasons,
            "change_rate_24h": x["change_rate_24h"],
            "price_from_latest_pct": x["price_from_latest_pct"],
            "price_from_first_pct": x["price_from_first_pct"],
            "latest_label": x["latest_label"],
            "latest_ret5": x["latest_ret5"],
            "latest_ret15": x["latest_ret15"],
            "latest_ret30": x["latest_ret30"],
            "latest_ret60": x["latest_ret60"],
            "latest_score": x["latest_score"],
            "latest_signal_age_h": x["latest_signal_age_h"],
        })

    # 후보 필터/순위 결과도 FULL PIPELINE DEBUG에 합친다.
    for x in ranked:
        entry = pipeline_debug.get(x["market"])
        if entry is None:
            continue
        pr = pot_rank.get(x["market"])
        lr = launch_rank.get(x["market"])
        p_reasons = potential_fail_reasons(x)
        l_reasons = launch_fail_reasons(x)

        entry.update({
            "potential_candidate_rank": pr,
            "launch_candidate_rank": lr,
            "potential_status": (
                "TOP5" if pr is not None and pr <= TOP_N
                else ("PASSED_BUT_OUTSIDE_TOP5" if pr is not None else "FILTERED")
            ),
            "launch_status": (
                "TOP5" if lr is not None and lr <= TOP_N
                else ("PASSED_BUT_OUTSIDE_TOP5" if lr is not None else "FILTERED")
            ),
            "potential_fail_reasons": p_reasons,
            "launch_fail_reasons": l_reasons,
        })

        if pr is not None or lr is not None:
            entry["pipeline_status"] = "PASSED_CANDIDATE_FILTER"
        else:
            entry["pipeline_status"] = "SCORED_BUT_FILTERED"

    # 검색용 전체 KRW Debug. TOP60 제한 없음.
    debug_all_krw = [pipeline_debug[m] for m in all_krw_markets]

    payload = {
        "version": "Live Ranker v0.6.3 FULL PIPELINE DEBUG",
        "generated_at_utc": now.isoformat(),
        "lookback_hours": LOOKBACK_HOURS,
        "top_n": TOP_N,
        "method_note": (
            "실험용 비교점수이며 확률이 아님. v0.6.3 FULL PIPELINE DEBUG는 v0.6 점수/필터/순위를 변경하지 않고 "
            "전체 Upbit KRW 종목이 어느 단계에서 제외됐는지 추적한다. "
            "Launch TOP5는 실제 LAUNCH stage만 허용하고, 과거 가격미반응 강신호와 최근 거래량 재점화가 있었던 종목을 우대한다. "
            "24시간 이미 많이 오른 종목은 감점해 초기 Launch를 우선한다."
        ),
        # 기존 대시보드 호환용: top5는 Launch TOP5와 동일
        "top5": launch_top5,
        "launch_top5": launch_top5,
        "potential_top5": potential_top5,
        "ignition_watch": ignition_watch,
        "sleeper_watch": sleeper_watch,
        "ranked_top30": ranked[:RANK_N],
        # 기존 Debug UI 호환
        "debug_ranked": debug_rows[:60],
        # v0.6.3: 전체 KRW 검색용. BIO/RENDER/KAITO처럼 72h gate 이전 탈락도 보인다.
        "debug_all_krw": debug_all_krw,
        "debug_pipeline_summary": {
            "all_krw_markets": len(all_krw_markets),
            "markets_with_any_saved_signal": sum(
                1 for x in debug_all_krw if x.get("total_saved_signal_rows", 0) > 0
            ),
            "markets_with_recent_72h_signal": sum(
                1 for x in debug_all_krw if x.get("recent_signal_rows_72h", 0) > 0
            ),
            "scored_markets": len(ranked),
            "excluded_recent_72h_gate": sum(
                1 for x in debug_all_krw
                if x.get("pipeline_fail_stage") == "RECENT_72H_SIGNAL_GATE"
            ),
        },
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
            "transition=", x["transition_score"],
            "ret=", (x["latest_ret5"], x["latest_ret15"], x["latest_ret30"], x["latest_ret60"])
        )

    print("=== POTENTIAL TOP5 ===")
    for i, x in enumerate(potential_top5, 1):
        print(
            i, x["market"],
            "potential=", x["potential_score"],
            "launch=", x["launch_score"],
            "transition=", x["transition_score"],
            "ret=", (x["latest_ret5"], x["latest_ret15"], x["latest_ret30"], x["latest_ret60"])
        )

if __name__ == "__main__":
    main()
