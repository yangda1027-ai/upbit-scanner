import os, sys, json, time, statistics
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
from scipy.spatial import cKDTree

# V6.3의 검증된 신호/체결 함수 재사용: 점수 공식과 runner 규칙은 수정하지 않는다.
sys.path.append(str(Path(__file__).resolve().parent))
import backtest_strategy as v63

TRAIN_DAYS = int(os.getenv("TRAIN_DAYS", "120"))
TEST_DAYS = int(os.getenv("TEST_DAYS", "60"))
K = int(os.getenv("K_NEIGHBORS", "50"))
DELAY = float(os.getenv("REQUEST_DELAY", "0.12"))
OUT = Path("docs/data/backtest_replication_latest.json")
HOLDOUT_OFFSET_DAYS = int(os.getenv("HOLDOUT_OFFSET_DAYS", "181"))

# V6.3 모듈의 API 지연/이웃 수만 환경변수와 맞춘다.
v63.DELAY = DELAY
v63.K = K


def clamp(x, a, b):
    return max(a, min(b, x))


def scale(x, lo, hi):
    if hi == lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)


def mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def completed_index(rows, decision_time):
    """결정시각 직전까지 완전히 끝난 캔들만 사용한다."""
    return v63.li(rows, decision_time - timedelta(seconds=1))


def regime_snapshot(data, decision_time):
    """
    결과를 보기 전에 고정한 V6.4 Market Regime 규칙.

    구성(0~100):
      BTC 4h 모멘텀 20
      BTC 24h 모멘텀 15
      BTC EMA20>=EMA50 15
      알트 1h 상승 breadth 20
      알트 4h 상승 breadth 15
      알트 거래대금 확산 15

    GREEN >=70 / YELLOW 50~69.99 / RED <50
    단, BTC 1h <= -2.5%, BTC 4h <= -5%, 알트 1h 상승비율 <=25% 중 하나면 Emergency RED.

    이 임계값은 이번 결과를 본 뒤 재조정하지 않는 것을 전제로 한다.
    """
    btc = data.get("KRW-BTC")
    if not btc:
        return None

    bi = completed_index(btc["h1"], decision_time)
    if bi < 50:
        return None

    br = btc["h1"]
    bc = [float(x["trade_price"]) for x in br[max(0, bi - 70):bi + 1]]
    btc_now = float(br[bi]["trade_price"])
    btc_ret1 = (btc_now / float(br[bi - 1]["trade_price"]) - 1) * 100
    btc_ret4 = (btc_now / float(br[bi - 4]["trade_price"]) - 1) * 100
    btc_ret24 = (btc_now / float(br[bi - 24]["trade_price"]) - 1) * 100
    btc_ema_bull = v63.ema(bc[-30:], 20) >= v63.ema(bc[-60:], 50)

    pos1 = 0
    pos4 = 0
    vol_expand = 0
    valid = 0
    ret1_vals = []

    for mk, d in data.items():
        if mk == "KRW-BTC":
            continue
        i = completed_index(d["h1"], decision_time)
        if i < 24:
            continue
        rows = d["h1"]
        now = float(rows[i]["trade_price"])
        r1 = (now / float(rows[i - 1]["trade_price"]) - 1) * 100
        r4 = (now / float(rows[i - 4]["trade_price"]) - 1) * 100
        cur_v = float(rows[i]["candle_acc_trade_price"])
        prev_v = [float(x["candle_acc_trade_price"]) for x in rows[i - 24:i]]
        vr = cur_v / (mean(prev_v) or 1.0)
        valid += 1
        ret1_vals.append(r1)
        if r1 > 0:
            pos1 += 1
        if r4 > 0:
            pos4 += 1
        if vr >= 1.20:
            vol_expand += 1

    if valid < 30:
        return None

    breadth1 = 100 * pos1 / valid
    breadth4 = 100 * pos4 / valid
    volume_share = 100 * vol_expand / valid
    median_alt_ret1 = float(np.median(ret1_vals)) if ret1_vals else 0.0

    market_score = 0.0
    market_score += 20 * scale(btc_ret4, -3.0, 3.0)
    market_score += 15 * scale(btc_ret24, -8.0, 8.0)
    market_score += 15 if btc_ema_bull else 0
    market_score += 20 * scale(breadth1, 35.0, 70.0)
    market_score += 15 * scale(breadth4, 35.0, 70.0)
    market_score += 15 * scale(volume_share, 25.0, 60.0)
    market_score = clamp(market_score, 0.0, 100.0)

    emergency = (btc_ret1 <= -2.5) or (btc_ret4 <= -5.0) or (breadth1 <= 25.0)
    if emergency:
        regime = "RED"
    elif market_score >= 70:
        regime = "GREEN"
    elif market_score >= 50:
        regime = "YELLOW"
    else:
        regime = "RED"

    return {
        "regime": regime,
        "market_score": round(market_score, 2),
        "emergency_red": emergency,
        "btc_ret1": round(btc_ret1, 3),
        "btc_ret4": round(btc_ret4, 3),
        "btc_ret24": round(btc_ret24, 3),
        "btc_ema_bull": bool(btc_ema_bull),
        "breadth1": round(breadth1, 2),
        "breadth4": round(breadth4, 2),
        "volume_expansion_share": round(volume_share, 2),
        "median_alt_ret1": round(median_alt_ret1, 3),
        "market_count": valid,
    }


def build_training_and_test_data(train, test, end):
    data = {}
    X, G, MK = [], [], []

    ms = v63.markets()
    for n, m in enumerate(ms, 1):
        try:
            h1 = v63.fetch(m["market"], 60, train, end)
            if len(h1) < 200:
                continue

            # 학습 예시는 해당 1시간봉이 끝난 뒤의 상태라고 해석하며,
            # 결과는 그 다음 6개 1시간봉의 고가만 사용한다.
            for i in range(30, len(h1) - 6, 6):
                t = v63.dt(h1[i]["candle_date_time_utc"])
                if t >= test:
                    break
                f = v63.feat(h1, i)
                if not f:
                    continue
                base = float(h1[i]["trade_price"])
                hi = max(float(x["high_price"]) for x in h1[i + 1:i + 7])
                X.append([f[k] for k in v63.F])
                G.append((hi / base - 1) * 100)
                MK.append(m["market"])

            r5 = v63.fetch(m["market"], 5, test - timedelta(days=3), end + timedelta(days=1))
            if len(r5) < 500:
                continue
            data[m["market"]] = {
                "name": m["name"],
                "h1": h1,
                "r5": r5,
                "r15": v63.agg(r5, 15),
            }
            print(n, m["market"], len(r5))
        except Exception as e:
            print("ERR", m["market"], e)

    return data, np.asarray(X, float), np.asarray(G, float), MK


def main():
    current_end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    end = current_end - timedelta(days=HOLDOUT_OFFSET_DAYS)
    test = end - timedelta(days=TEST_DAYS)
    train = test - timedelta(days=TRAIN_DAYS)

    data, X, G, MK = build_training_and_test_data(train, test, end)
    if "KRW-BTC" not in data:
        raise RuntimeError("KRW-BTC 데이터가 없어 Market Regime을 계산할 수 없습니다.")
    if len(X) < 100:
        raise RuntimeError("학습 예시가 부족합니다.")

    med = np.median(X, 0)
    mad = np.median(np.abs(X - med), 0) * 1.4826
    std = np.std(X, 0)
    sca = np.where(mad > 1e-9, mad, np.where(std > 1e-9, std, 1))
    tree = cKDTree(((X - med) / sca) * v63.W)

    # 시간별 regime을 먼저 고정해 캐시한다.
    regime_cache = {}
    regime_hours = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    t = test + timedelta(hours=1)
    while t <= end - timedelta(hours=24):
        rs = regime_snapshot(data, t)
        if rs:
            regime_cache[t] = rs
            regime_hours[rs["regime"]] += 1
        t += timedelta(hours=1)

    signals = []
    t = test + timedelta(hours=1)
    while t <= end - timedelta(hours=24):
        rs = regime_cache.get(t)
        if not rs:
            t += timedelta(hours=1)
            continue

        # 중요: decision_time 직전의 완결 캔들만 사용한다.
        cut = t - timedelta(seconds=1)
        for mk, d in data.items():
            i1 = v63.li(d["h1"], cut)
            i5 = v63.li(d["r5"], cut)
            i15 = v63.li(d["r15"], cut)
            if min(i1, i5, i15) < 30:
                continue

            f1 = v63.feat(d["h1"], i1)
            f5 = v63.feat(d["r5"], i5)
            f15 = v63.feat(d["r15"], i15)
            if not (f1 and f5 and f15):
                continue

            c1 = [float(x["trade_price"]) for x in d["h1"][max(0, i1 - 60):i1 + 1]]
            cur = float(d["r5"][i5]["trade_price"])
            hi24 = max(float(x["high_price"]) for x in d["h1"][max(0, i1 - 23):i1 + 1])
            m = {
                "vol5": f5["vol_ratio"], "vol15": f15["vol_ratio"], "vol1h": f1["vol_ratio"],
                "rsi15": f15["rsi"], "rsi1h": f1["rsi"], "ret1h": f1["ret1"],
                "ret3h": f1["ret3"], "ret6h": f1["ret6"], "dist24": (cur / hi24 - 1) * 100,
                "ema2050": v63.ema(c1[-30:], 20) >= v63.ema(c1[-60:], 50),
            }

            q = ((np.asarray([f1[k] for k in v63.F]) - med) / sca) * v63.W
            dd, ii = tree.query(q, k=min(400, len(X)))
            if np.isscalar(ii):
                ii, dd = [ii], [dd]

            neigh, used = [], set()
            for dis, jj in zip(dd, ii):
                tm = MK[int(jj)]
                if tm == mk or tm in used:
                    continue
                neigh.append((float(dis), int(jj)))
                used.add(tm)
                if len(neigh) >= K:
                    break
            if not neigh:
                continue

            ng = [G[jj] for _, jj in neigh]
            p3 = 100 * sum(x >= 3 for x in ng) / len(ng)
            p5 = 100 * sum(x >= 5 for x in ng) / len(ng)
            p10 = 100 * sum(x >= 10 for x in ng) / len(ng)
            sim = 100 / (1 + mean([x for x, _ in neigh]))

            base7 = min(float(x["trade_price"]) for x in d["h1"][max(0, i1 - 7 * 24):i1 + 1])
            high7 = max(float(x["high_price"]) for x in d["h1"][max(0, i1 - 7 * 24):i1 + 1])
            recent = (high7 / base7 - 1) * 100 >= 20

            # 실험 격리: B점수에 실제 BTC regime을 넣지 않는다.
            # V6.3과 동일한 btc=2 근사로 점수를 만든 뒤, regime은 별도 필터로만 평가한다.
            sv = v63.score(m, p3, p5, p10, sim, 2, recent)
            if sv >= 70:
                signals.append({
                    "time": t, "market": mk, "score": float(sv), "i5": i5,
                    "entry": cur, "regime": rs["regime"], "market_score": rs["market_score"],
                })
        t += timedelta(hours=1)

    signals.sort(key=lambda x: x["time"])

    thresholds = [70, 75, 80]
    group_defs = {
        "ALL": lambda s: True,
        "GREEN": lambda s: s["regime"] == "GREEN",
        "YELLOW": lambda s: s["regime"] == "YELLOW",
        "RED": lambda s: s["regime"] == "RED",
        "NON_RED": lambda s: s["regime"] in ("GREEN", "YELLOW"),
    }
    strategies = {
        "fixed_5_3_6h": "즉시 100% · +5% 익절 / -3% 손절 / 6h",
        "runner70_80only": "즉시 100% · 80점+만 +5%에서 30% 익절 · 70% 3% 트레일링 / 최대 24h",
    }

    results = {str(th): {} for th in thresholds}

    for th in thresholds:
        base_eligible = [s for s in signals if s["score"] >= th]
        for gkey, gfun in group_defs.items():
            eligible = [s for s in base_eligible if gfun(s)]
            results[str(th)][gkey] = {}

            for skey in strategies:
                rows = []
                busy_until = {}
                for s in eligible:
                    mk = s["market"]
                    if mk in busy_until and s["time"] <= busy_until[mk]:
                        continue
                    d = data[mk]
                    r5 = d["r5"]
                    if skey == "fixed_5_3_6h":
                        tr = v63.fixed_trade(r5, s["i5"], s["entry"], 6)
                    else:
                        tr = v63.runner_trade(r5, s["i5"], s["entry"], s["score"], .70, 3.0, 24, True)
                    exit_t = v63.dt(r5[tr["exit_i"]]["candle_date_time_utc"])
                    busy_until[mk] = exit_t
                    rows.append({**tr, "market": mk, "time": s["time"].isoformat(), "score": round(s["score"], 2)})

                results[str(th)][gkey][skey] = v63.summarize(rows)

    signal_regimes = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    for s in signals:
        signal_regimes[s["regime"]] += 1

    out = {
        "version": "V6.5 Historical Replication Backtest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "markets": len(data),
            "k_neighbors": K,
            "lookahead_alignment": "completed_candles_only",
            "holdout_offset_days": HOLDOUT_OFFSET_DAYS,
            "train_start_utc": train.isoformat(),
            "test_start_utc": test.isoformat(),
            "test_end_utc": end.isoformat(),
        },
        "regime_rules": {
            "weights": {
                "btc_4h_momentum": 20,
                "btc_24h_momentum": 15,
                "btc_ema20_ge_ema50": 15,
                "alt_breadth_1h": 20,
                "alt_breadth_4h": 15,
                "alt_volume_expansion_share": 15,
            },
            "green": "market_score >= 70",
            "yellow": "50 <= market_score < 70",
            "red": "market_score < 50",
            "emergency_red": "BTC 1h <= -2.5% OR BTC 4h <= -5% OR alt 1h breadth <= 25%",
            "volume_expansion": "현재 1h 거래대금 >= 직전 24h 평균의 1.20배인 알트 비율",
        },
        "regime_hours": regime_hours,
        "signal_regimes": signal_regimes,
        "strategy_labels": strategies,
        "results": results,
        "notes": [
            "V6.4 결과를 본 뒤 규칙을 바꾸지 않고, 동일한 B점수·Regime·체결 규칙을 과거의 별도 60일 구간에 그대로 복제",
            "V6.4에서 사용한 최신 180일(학습 120일 + 검증 60일)보다 더 오래된 구간을 사용하며, 1일 간격을 두어 데이터 중복을 피함",
            "이번 테스트의 목적은 수익률 최적화가 아니라 RED 회피 효과와 GREEN/YELLOW 우위의 재현성 확인",
            "이번 결과를 본 뒤 GREEN/YELLOW/RED 경계값, B70/75/80 기준, Runner 파라미터를 조정하지 않는 것을 권장",
            "B점수 공식은 V6.4와 동일하며 score() 내부 btc=2 근사를 유지하여 Market Regime 필터 효과를 분리",
            "결정시각 직전까지 완전히 끝난 캔들만 사용",
            "같은 코인 포지션 보유 중 중복진입 제외",
            "동일 5분봉에서 +5%와 -3%가 동시에 닿으면 손절 우선",
            "runner는 +5% 도달 뒤 다음 5분봉부터 최고가 대비 -3% 트레일링, 최소 스탑은 진입가",
            "수수료·슬리피지·호가체결 영향은 아직 미반영",
            "현재 상장 종목 목록을 과거에 적용하므로 상장폐지/신규상장에 따른 survivorship bias 가능성은 남아 있음",
            "이 과거 복제 테스트도 진짜 미래 데이터인 forward test를 대체하지 않음"
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
