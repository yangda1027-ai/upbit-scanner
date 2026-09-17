import json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

API = "https://api.upbit.com"
OUT = Path("docs/data/v7_6_signals.json")
LATEST = Path("docs/data/v7_6_latest.json")
LIVE = Path("docs/data/v7_6_live.json")
TOP20 = Path("docs/data/v7_6_top20.json")
MARKET_SCAN = Path("docs/data/market_scan_latest.json")
HISTORY_DIR = Path("docs/data/v7_6_history")
SNAPSHOT_HISTORY_DIR = Path("docs/data/v7_6_rank_history")
# Preserve rank snapshots long enough for prospective five-slot validation.
SNAPSHOT_RETENTION_DAYS = 180

STABLE = {"USDT", "USDC", "DAI", "USD1", "USDE", "FDUSD", "TUSD"}
TOP_N = 10
HISTORY_RANK_N = 30
EARLY_DIAG_SCORE = 65
EPISODE_GAP_MIN = 60
MAX_ROWS = 50000
LIVE_TOP_N = 20
PER_MARKET_HISTORY_MAX = 250
IGNITION_HORIZON_HOURS = 3
IGNITION_SUCCESS_PCT = 1.5

S = requests.Session()
S.headers.update({"User-Agent": "upbit-v7-6-all-krw/1.0"})


def get(path, params=None, tries=4):
    last = None
    for i in range(tries):
        try:
            r = S.get(API + path, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(1.0 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.6 * (i + 1))
    raise last


def f(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def pct(a, b):
    return (a / b - 1) * 100 if b else 0.0


def clamp(x, a, b):
    return max(a, min(b, x))


def markets():
    xs = get("/v1/market/all", {"is_details": "false"})
    return [
        x["market"]
        for x in xs
        if x["market"].startswith("KRW-")
        and x["market"].split("-", 1)[1] not in STABLE
    ]


def candles(m, unit, count):
    """Return only fully closed minute candles, oldest first."""
    raw = get(
        f"/v1/candles/minutes/{unit}",
        {"market": m, "count": min(count + 2, 200)},
    )
    now = datetime.now(timezone.utc)
    closed = []
    for row in reversed(raw):
        try:
            started = datetime.fromisoformat(
                str(row["candle_date_time_utc"]).replace("Z", "+00:00")
            )
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if started + timedelta(minutes=unit) <= now:
                closed.append(row)
        except Exception:
            continue
    return closed[-count:]


def features(m):
    # V7.3ÃÂÃÂ¬ÃÂÃÂÃÂÃÂ ÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¼ÃÂÃÂ­ÃÂÃÂÃÂÃÂ 5ÃÂÃÂ«ÃÂÃÂ¶ÃÂÃÂÃÂÃÂ«ÃÂÃÂ´ÃÂÃÂ feature / score / EARLY ÃÂÃÂªÃÂÃÂ·ÃÂÃÂÃÂÃÂ¬ÃÂÃÂ¹ÃÂÃÂ
    c = candles(m, 5, 30)
    if len(c) < 25:
        return None

    cl = [f(x["trade_price"]) for x in c]
    tv = [f(x["candle_acc_trade_price"]) for x in c]

    r5 = pct(cl[-1], cl[-2])
    r15 = pct(cl[-1], cl[-4])
    r30 = pct(cl[-1], cl[-7])
    r60 = pct(cl[-1], cl[-13])

    v5 = tv[-1] / (sum(tv[-7:-1]) / 6 or 1)
    a15 = (sum(tv[-3:]) / 3) / (sum(tv[-9:-3]) / 6 or 1)
    a30 = (sum(tv[-6:]) / 6) / (sum(tv[-18:-6]) / 12 or 1)

    score = 35.0
    score += (
        clamp((v5 - 1) * 12, -8, 18)
        + clamp((a15 - 1) * 18, -10, 25)
        + clamp((a30 - 1) * 10, -6, 14)
    )
    score += clamp(r5 * 3, -8, 8) + clamp(r15 * 1.8, -8, 10)

    over = 0.0
    if r5 > 2.5:
        over += (r5 - 2.5) * 5
    if r15 > 4.0:
        over += (r15 - 4.0) * 4
    if r60 > 7.0:
        over += (r60 - 7.0) * 2
    score -= clamp(over, 0, 35)

    chase = (r5 >= 3.0) or (r15 >= 5.0) or (r60 >= 8.0)
    early = (not chase) and (a15 >= 1.25) and (v5 >= 1.15)
    label = "CHASE" if chase else ("EARLY" if early else "WATCH")

    return {
        "market": m,
        "price": cl[-1],
        "closed_candle_time_utc": c[-1].get("candle_date_time_utc"),
        "closed_candle_high": f(c[-1].get("high_price")),
        "closed_candle_low": f(c[-1].get("low_price")),
        "score": round(clamp(score, 0, 100), 2),
        "label": label,
        "ret_5m": round(r5, 3),
        "ret_15m": round(r15, 3),
        "ret_30m": round(r30, 3),
        "ret_60m": round(r60, 3),
        "value_ratio_5m": round(v5, 3),
        "value_accel_15m": round(a15, 3),
        "value_accel_30m": round(a30, 3),
    }


def btc_state():
    c = candles("KRW-BTC", 5, 24)
    cl = [f(x["trade_price"]) for x in c]
    return {
        "ret_15m": round(pct(cl[-1], cl[-4]), 3),
        "ret_60m": round(pct(cl[-1], cl[-13]), 3),
    }


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default



def iso_dt(x):
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except Exception:
        return None


def nearest_price_change(history_rows, market, now, current_price, hours):
    """Approximate return versus the stored episode closest to N hours ago."""
    target = now.timestamp() - hours * 3600
    best = None
    best_gap = None
    for r in history_rows:
        if r.get("market") != market:
            continue
        dt = iso_dt(r.get("ts"))
        if not dt:
            continue
        gap = abs(dt.timestamp() - target)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = r
    # Ignore very stale substitutes: tolerance grows with the requested horizon.
    tolerance = max(45 * 60, hours * 3600 * 0.35)
    if not best or best_gap is None or best_gap > tolerance:
        return None
    old_price = f(best.get("price"))
    if not old_price:
        return None
    return round(pct(current_price, old_price), 3)


def summarize_market_history(history_rows, market, now, current_price):
    """Build a compact 72h signal/episode/ignition summary for monitoring."""
    items = []
    for r in history_rows:
        if r.get("market") != market:
            continue
        dt = iso_dt(r.get("ts"))
        if dt:
            items.append((dt, r))
    items.sort(key=lambda x: x[0])

    windows = {}
    for h in (1, 3, 6, 12, 24, 72):
        cutoff = now.timestamp() - h * 3600
        rs = [r for dt, r in items if dt.timestamp() >= cutoff]
        windows[str(h)] = {
            "signals": len(rs),
            "A": sum(bool(r.get("A")) for r in rs),
            "B": sum(bool(r.get("B")) for r in rs),
            "C": sum(bool(r.get("C")) for r in rs),
            "ABC_ALL": sum(bool(r.get("ABC_ALL")) for r in rs),
            "avg_score": round(sum(f(r.get("score")) for r in rs) / len(rs), 2) if rs else None,
        }

    last72 = [(dt, r) for dt, r in items if (now - dt).total_seconds() <= 72 * 3600]
    latest_signal_age_min = None
    if items:
        latest_signal_age_min = round((now - items[-1][0]).total_seconds() / 60, 1)

    # Each stored history row is already episode-deduplicated by the main script.
    episode_count_72h = len(last72)

    # Historical ignition quality from future stored episode prices within 3h.
    success = fail = 0
    reactions = []
    seq = items[-300:]
    for i, (dt, r) in enumerate(seq):
        p0 = f(r.get("price"))
        if not p0:
            continue
        end_ts = dt.timestamp() + IGNITION_HORIZON_HOURS * 3600
        future_prices = [f(rr.get("price")) for d2, rr in seq[i+1:] if d2.timestamp() <= end_ts and f(rr.get("price")) > 0]
        if not future_prices:
            continue
        reaction = pct(max(future_prices), p0)
        reactions.append(reaction)
        if reaction >= IGNITION_SUCCESS_PCT:
            success += 1
        else:
            fail += 1

    total_eval = success + fail
    ignition = {
        "success": success,
        "fail": fail,
        "evaluated": total_eval,
        "rate": round(success / total_eval, 4) if total_eval else None,
        "success_threshold_pct": IGNITION_SUCCESS_PCT,
        "horizon_hours": IGNITION_HORIZON_HOURS,
        "avg_max_reaction_pct": round(sum(reactions) / len(reactions), 3) if reactions else None,
        "median_max_reaction_pct": round(sorted(reactions)[len(reactions)//2], 3) if reactions else None,
    }

    price_changes = {
        f"{h}h": nearest_price_change(history_rows, market, now, current_price, h)
        for h in (1, 3, 6, 12, 24, 72)
    }

    return {
        "windows": windows,
        "independent_episodes_72h": episode_count_72h,
        "latest_signal_age_min": latest_signal_age_min,
        "ignition": ignition,
        "price_change": price_changes,
    }


def daily_extension(m):
    """Fetch 31 daily candles only for shortlisted names and calculate extension."""
    try:
        ds = list(reversed(get("/v1/candles/days", {"market": m, "count": 31})))
        closes = [f(x.get("trade_price")) for x in ds]
        lows = [f(x.get("low_price")) for x in ds]
        if len(closes) < 2:
            return {}
        cur = closes[-1]
        def ret_days(n):
            if len(closes) <= n or not closes[-1-n]:
                return None
            return round(pct(cur, closes[-1-n]), 3)
        low14 = min(lows[-14:]) if len(lows) >= 14 else min(lows)
        low30 = min(lows[-30:]) if len(lows) >= 30 else min(lows)
        return {
            "ret_3d": ret_days(3),
            "ret_7d": ret_days(7),
            "ret_14d": ret_days(14),
            "ret_30d": ret_days(30),
            "from_14d_low_pct": round(pct(cur, low14), 3) if low14 else None,
            "from_30d_low_pct": round(pct(cur, low30), 3) if low30 else None,
            "low_14d": low14,
            "low_30d": low30,
        }
    except Exception as e:
        return {"error": str(e)}


def monitor_score(row, hist_summary):
    """Monitoring priority, not a trading signal. Penalizes weak repeated ignitions."""
    score = f(row.get("score"))
    ign = hist_summary.get("ignition", {})
    rate = ign.get("rate")
    evaluated = ign.get("evaluated", 0) or 0
    episodes = hist_summary.get("independent_episodes_72h", 0) or 0
    abc72 = hist_summary.get("windows", {}).get("72", {}).get("ABC_ALL", 0) or 0

    if rate is not None and evaluated >= 3:
        score += (rate - 0.5) * 24
        if rate < 0.30 and episodes >= 4:
            score -= 12
    score += min(abc72, 5) * 1.5
    if row.get("label") == "CHASE":
        score -= 20
    return round(clamp(score, 0, 100), 2)



def sma(xs, n):
    return sum(xs[-n:]) / n if len(xs) >= n else None


def rsi14(xs):
    if len(xs) < 15:
        return None
    gains, losses = [], []
    for a, b in zip(xs[-15:-1], xs[-14:]):
        d = b - a
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains) / 14
    al = sum(losses) / 14
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    rs = ag / al
    return 100 - 100 / (1 + rs)


def tf_scan(m, unit, count=80):
    c = candles(m, unit, count)
    if len(c) < 35:
        return None
    cl = [f(x.get("trade_price")) for x in c]
    hi = [f(x.get("high_price")) for x in c]
    lo = [f(x.get("low_price")) for x in c]
    tv = [f(x.get("candle_acc_trade_price")) for x in c]
    ma5, ma10, ma20 = sma(cl,5), sma(cl,10), sma(cl,20)
    ma60 = sma(cl,60)
    recent_value = sum(tv[-3:]) / 3
    base_value = sum(tv[-15:-3]) / 12 or 1
    value_accel = recent_value / base_value
    prev_high = max(hi[-13:-1])
    support = min(lo[-7:])
    resistance_dist = pct(prev_high, cl[-1]) if cl[-1] else 0
    range6 = (max(hi[-6:]) - min(lo[-6:])) / cl[-1] * 100 if cl[-1] else 99
    ret3 = pct(cl[-1], cl[-4])
    ret12 = pct(cl[-1], cl[-13])
    ma_reclaim = cl[-1] >= ma5 and cl[-1] >= ma10
    aligned = ma5 >= ma10 >= ma20
    return {
        "price": cl[-1], "rsi": round(rsi14(cl),2),
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
        "ma_reclaim": ma_reclaim, "ma_aligned": aligned,
        "value_accel": round(value_accel,3),
        "prev_high": prev_high, "support": support,
        "resistance_distance_pct": round(resistance_dist,3),
        "compression_pct": round(range6,3),
        "ret_3bars_pct": round(ret3,3), "ret_12bars_pct": round(ret12,3),
    }


def independent_market_scan(ms, rank24, value24, now):
    """Independent 5m/15m/1h chart scanner. Does NOT alter V7.6 scoring."""
    candidates = []
    for idx, m in enumerate(ms, 1):
        try:
            t5 = tf_scan(m, 5)
            t15 = tf_scan(m, 15)
            t60 = tf_scan(m, 60)
            if not (t5 and t15 and t60):
                continue
            price = t5["price"]
            # Exclude obvious chase conditions. We want pre-breakout/recovery structures.
            chase = (t5["ret_12bars_pct"] > 8 or t15["ret_12bars_pct"] > 14 or
                     t60["ret_12bars_pct"] > 25 or t5["rsi"] >= 76 or t15["rsi"] >= 74)
            if chase:
                continue
            score = 0.0
            # volume/value acceleration
            score += clamp((t5["value_accel"]-1)*16, -8, 24)
            score += clamp((t15["value_accel"]-1)*12, -6, 18)
            score += clamp((t60["value_accel"]-1)*6, -3, 9)
            # trend/reclaim across timeframes
            score += 7 if t5["ma_reclaim"] else -4
            score += 8 if t15["ma_reclaim"] else -5
            score += 9 if t60["ma_reclaim"] else -6
            score += 5 if t5["ma_aligned"] else 0
            score += 6 if t15["ma_aligned"] else 0
            score += 6 if t60["ma_aligned"] else 0
            # near resistance but not already extended
            if -0.5 <= t5["resistance_distance_pct"] <= 2.5: score += 8
            if -0.5 <= t15["resistance_distance_pct"] <= 3.5: score += 7
            # compression
            if t5["compression_pct"] <= 2.5: score += 6
            if t15["compression_pct"] <= 4.5: score += 4
            # RSI: constructive, not overheated
            for t, bonus in ((t5,5),(t15,5),(t60,4)):
                if 45 <= t["rsi"] <= 68: score += bonus
                elif t["rsi"] >= 72: score -= bonus
            score = round(clamp(score, 0, 100), 2)
            # Need at least a credible multi-timeframe setup.
            if score < 45:
                continue
            trigger = max(t5["prev_high"], t15["prev_high"])
            support = min(t5["support"], t15["support"])
            # observation zone around current/support, deliberately not a blind buy range
            obs_low = max(support, price * 0.975)
            obs_high = price * 1.005
            invalid = min(t15["support"], price * 0.965)
            state = "ì§ì ê²í " if (score >= 65 and t5["ma_reclaim"] and t15["ma_reclaim"] and t5["value_accel"] >= 1.15) else "ë§¤ììë¦¬ íì± ì¤"
            reasons=[]
            if t5["value_accel"] >= 1.2: reasons.append(f"5m ê±°ëëê¸ {t5['value_accel']:.2f}x")
            if t15["value_accel"] >= 1.15: reasons.append(f"15m ê±°ëëê¸ {t15['value_accel']:.2f}x")
            if t5["ma_aligned"]: reasons.append("5m MA ì ë°°ì´")
            elif t5["ma_reclaim"]: reasons.append("5m MA5/10 íë³µ")
            if t15["ma_aligned"]: reasons.append("15m MA ì ë°°ì´")
            elif t15["ma_reclaim"]: reasons.append("15m MA5/10 íë³µ")
            if t60["ma_reclaim"]: reasons.append("1h MA5/10 ì")
            if t5["compression_pct"] <= 2.5: reasons.append("5m ê°ê²© ìì¶")
            candidates.append({
                "market":m, "symbol":m.split('-',1)[1], "price":price,
                "status":state, "scan_score":score,
                "trade_value_rank_24h":rank24.get(m), "trade_value_24h":round(value24.get(m,0),2),
                "entry_observe_low":round(obs_low,8), "entry_observe_high":round(obs_high,8),
                "trigger_price":round(trigger,8), "invalidation_price":round(invalid,8),
                "reasons":reasons[:6], "5m":t5, "15m":t15, "1h":t60,
            })
        except Exception as e:
            print("market-scan skip", m, e)
        time.sleep(0.08)
        if idx % 25 == 0 or idx == len(ms):
            print(f"independent market scan {idx}/{len(ms)}")
    candidates.sort(key=lambda x:(-x["scan_score"], x.get("trade_value_rank_24h") or 9999))
    top = candidates[:5]
    payload={
        "generated_at":now.isoformat(),
        "version":"INDEPENDENT_MARKET_SCAN_V1",
        "note":"Independent from V7.6 score/ABC logic. Closed candles only; candidates are observation/trigger setups, not guaranteed buys.",
        "universe_count":len(ms), "qualified_count":len(candidates), "top5":top,
    }
    MARKET_SCAN.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return top


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    ms = markets()

    # 24h ÃÂÃÂªÃÂÃÂ±ÃÂÃÂ°ÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂªÃÂÃÂ¸ÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂ "ÃÂÃÂ­ÃÂÃÂÃÂÃÂÃÂÃÂ­ÃÂÃÂÃÂÃÂ°"ÃÂÃÂªÃÂÃÂ°ÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂ¼ ÃÂÃÂªÃÂÃÂ¸ÃÂÃÂ°ÃÂÃÂ«ÃÂÃÂ¡ÃÂÃÂ/ÃÂÃÂ«ÃÂÃÂ¹ÃÂÃÂÃÂÃÂªÃÂÃÂµÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ©ÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¼ÃÂÃÂ«ÃÂÃÂ¡ÃÂÃÂÃÂÃÂ«ÃÂÃÂ§ÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¬ÃÂÃÂ¬ÃÂÃÂÃÂÃÂ©
    tick = []
    for i in range(0, len(ms), 100):
        tick += get("/v1/ticker", {"markets": ",".join(ms[i:i + 100])})
        time.sleep(0.12)

    ranked = sorted(
        tick,
        key=lambda x: f(x.get("acc_trade_price_24h")),
        reverse=True,
    )
    rank24 = {x["market"]: i + 1 for i, x in enumerate(ranked)}
    value24 = {x["market"]: f(x.get("acc_trade_price_24h")) for x in ranked}

    # ÃÂÃÂ­ÃÂÃÂÃÂÃÂµÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¬ ÃÂÃÂ«ÃÂÃÂ³ÃÂÃÂÃÂÃÂªÃÂÃÂ²ÃÂÃÂ½ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂ:
    # V7.3 = ÃÂÃÂªÃÂÃÂ±ÃÂÃÂ°ÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂªÃÂÃÂ¸ÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ 120ÃÂÃÂªÃÂÃÂ°ÃÂÃÂÃÂÃÂ«ÃÂÃÂ§ÃÂÃÂ feature ÃÂÃÂªÃÂÃÂ³ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ°
    # V7.6 = KRW ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂÃÂÃÂ¬ÃÂÃÂ²ÃÂÃÂ´ ÃÂÃÂ¬ÃÂÃÂ¢ÃÂÃÂÃÂÃÂ«ÃÂÃÂªÃÂÃÂ©ÃÂÃÂ¬ÃÂÃÂÃÂÃÂ feature ÃÂÃÂªÃÂÃÂ³ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ°
    rows = []
    total = len(ms)

    for idx, m in enumerate(ms, 1):
        try:
            z = features(m)
            if z:
                r = rank24.get(m)
                z["trade_value_rank_24h"] = r
                z["trade_value_24h"] = round(value24.get(m, 0.0), 2)
                z["outside_top120"] = bool(r and r > 120)

                # A/B/CÃÂÃÂ«ÃÂÃÂÃÂÃÂ V7.5ÃÂÃÂ¬ÃÂÃÂÃÂÃÂ ÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¼ÃÂÃÂ­ÃÂÃÂÃÂÃÂ ÃÂÃÂªÃÂÃÂ´ÃÂÃÂÃÂÃÂ¬ÃÂÃÂ°ÃÂÃÂ° ÃÂÃÂ¬ÃÂÃÂ¡ÃÂÃÂ°ÃÂÃÂªÃÂÃÂ±ÃÂÃÂ´
                is_early = z["label"] == "EARLY"
                not_chase = z["label"] != "CHASE"

                z["A"] = bool(
                    is_early and not_chase
                    and z["score"] >= 85
                    and z["value_accel_15m"] >= 3
                    and z["value_accel_30m"] >= 2
                )

                z["B"] = bool(
                    is_early and not_chase
                    and z["score"] >= 80
                    and z["value_accel_15m"] >= 1.5
                    and z["value_accel_30m"] >= 3
                )

                z["C"] = bool(
                    is_early and not_chase
                    and z["score"] >= 70
                    and z["value_accel_15m"] >= 2
                    and z["value_accel_30m"] >= 2
                    and -1.5 <= z["ret_5m"] <= 0.5
                    and -2.0 <= z["ret_15m"] <= 1.0
                    and z["ret_30m"] <= 0.5
                )

                z["ABC_ALL"] = bool(z["A"] and z["B"] and z["C"])
                rows.append(z)

        except Exception as e:
            print("skip", m, e)

        # Upbit public API ÃÂÃÂ«ÃÂÃÂ¶ÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂ´ ÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ­ÃÂÃÂÃÂÃÂ
        time.sleep(0.07)

        if idx % 25 == 0 or idx == total:
            print(f"scanned {idx}/{total}")

    priority = {"EARLY": 0, "WATCH": 1, "CHASE": 2}
    rows.sort(
        key=lambda x: (
            priority[x["label"]],
            -x["score"],
            -x["value_accel_15m"],
        )
    )

    # ---- Full-universe rank snapshot history (logging only; V7.6 logic unchanged) ----
    # The scanner may run every 10 minutes, but a full-market snapshot is large.
    # Keep the full diagnostic snapshot hourly; fast candidate events are still
    # retained below whenever they first appear or their A/B/C state changes.
    SNAPSHOT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_rows = []
    for rank, z in enumerate(rows, 1):
        q = dict(z)
        q["rank"] = rank
        snapshot_rows.append(q)

    day_key = now.strftime("%Y-%m-%d")
    snapshot_path = SNAPSHOT_HISTORY_DIR / f"{day_key}.jsonl"
    full_snapshot_written = now.minute < 10
    if full_snapshot_written:
        snapshot_record = {
            "ts": now.isoformat(),
            "market_count": len(snapshot_rows),
            "rows": snapshot_rows,
        }
        with snapshot_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(snapshot_record, ensure_ascii=False, separators=(",", ":")) + "\n")

    # Keep disk/GitHub growth bounded. This only deletes old diagnostic snapshots.
    retention_cutoff = (now - timedelta(days=SNAPSHOT_RETENTION_DAYS)).date()
    for old_path in SNAPSHOT_HISTORY_DIR.glob("*.jsonl"):
        try:
            old_day = datetime.strptime(old_path.stem, "%Y-%m-%d").date()
            if old_day < retention_cutoff:
                old_path.unlink()
        except Exception:
            pass

    top = rows[:TOP_N]
    btc = btc_state()

    # ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂÃÂÃÂ¬ÃÂÃÂ²ÃÂÃÂ´ ÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ TOP120 ÃÂÃÂ«ÃÂÃÂ°ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¸ÃÂÃÂ«ÃÂÃÂÃÂÃÂ° ÃÂÃÂ¬ÃÂÃÂ¡ÃÂÃÂ°ÃÂÃÂªÃÂÃÂ±ÃÂÃÂ´ÃÂÃÂ¬ÃÂÃÂÃÂÃÂ ÃÂÃÂ­ÃÂÃÂÃÂÃÂµÃÂÃÂªÃÂÃÂ³ÃÂÃÂ¼ÃÂÃÂ­ÃÂÃÂÃÂÃÂ ÃÂÃÂ­ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂ³ÃÂÃÂ´ÃÂÃÂ«ÃÂÃÂ¥ÃÂÃÂ¼ ÃÂÃÂ«ÃÂÃÂ³ÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¥
    outside_candidates = [
        x for x in rows
        if x.get("outside_top120")
        and x["label"] == "EARLY"
        and (x["A"] or x["B"] or x["C"])
    ]

    # V7.6 history ÃÂÃÂ­ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¥:
    # 1) ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂÃÂÃÂ¬ÃÂÃÂ²ÃÂÃÂ´ ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂÃÂÃÂ«ÃÂÃÂ ÃÂÃÂ¬ TOP30
    # 2) ÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ ÃÂÃÂ«ÃÂÃÂ¬ÃÂÃÂ´ÃÂÃÂªÃÂÃÂ´ÃÂÃÂÃÂÃÂ­ÃÂÃÂÃÂÃÂÃÂÃÂªÃÂÃÂ²ÃÂÃÂ A/B/C ÃÂÃÂ­ÃÂÃÂÃÂÃÂµÃÂÃÂªÃÂÃÂ³ÃÂÃÂ¼ ÃÂÃÂ­ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂ³ÃÂÃÂ´ ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂÃÂÃÂ«ÃÂÃÂ¶ÃÂÃÂ
    # 3) ÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ ÃÂÃÂ«ÃÂÃÂ¬ÃÂÃÂ´ÃÂÃÂªÃÂÃÂ´ÃÂÃÂÃÂÃÂ­ÃÂÃÂÃÂÃÂÃÂÃÂªÃÂÃÂ²ÃÂÃÂ EARLY + score>=65 ÃÂÃÂ¬ÃÂÃÂ§ÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂ¨ ÃÂÃÂ­ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂ³ÃÂÃÂ´
    # ÃÂÃÂ«ÃÂÃÂ¥ÃÂÃÂ¼ ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¥ ÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¼ÃÂÃÂ«ÃÂÃÂ¡ÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¡ÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂ¤.
    selected = []
    for rank, z in enumerate(rows, 1):
        in_top30 = rank <= HISTORY_RANK_N
        abc_candidate = z["label"] == "EARLY" and (z["A"] or z["B"] or z["C"])
        early_diag = z["label"] == "EARLY" and z["score"] >= EARLY_DIAG_SCORE
        if not (in_top30 or abc_candidate or early_diag):
            continue

        q = dict(z)
        sources = []
        if in_top30:
            sources.append("TOP30")
        if abc_candidate:
            sources.append("ABC")
        if early_diag:
            sources.append("EARLY65")

        q.update({
            "ts": now.isoformat(),
            "rank": rank,
            "selection_source": "+".join(sources),
            "btc": btc,
            "h1": None,
            "h3": None,
            "h6": None,
            "h24": None,
        })
        selected.append(q)

    hist = load_json(OUT, [])
    if not isinstance(hist, list):
        hist = []

    # ÃÂÃÂªÃÂÃÂ°ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂ¢ÃÂÃÂÃÂÃÂ«ÃÂÃÂªÃÂÃÂ©/ÃÂÃÂªÃÂÃÂ°ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ A-B-C ÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ­ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂ¥ÃÂÃÂ¼ 5ÃÂÃÂ«ÃÂÃÂ¶ÃÂÃÂÃÂÃÂ«ÃÂÃÂ§ÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂ¤ ÃÂÃÂ¬ÃÂÃÂ¤ÃÂÃÂÃÂÃÂ«ÃÂÃÂ³ÃÂÃÂµ ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ­ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂ§ÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂªÃÂÃÂ³ÃÂÃÂ 
    # 60ÃÂÃÂ«ÃÂÃÂ¶ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ ÃÂÃÂ­ÃÂÃÂÃÂÃÂ ÃÂÃÂ«ÃÂÃÂ²ÃÂÃÂÃÂÃÂ«ÃÂÃÂ§ÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂÃÂÃÂ episodeÃÂÃÂ«ÃÂÃÂ¡ÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¥ÃÂÃÂ­ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂ¤.
    # ÃÂÃÂ«ÃÂÃÂÃÂÃÂ¨, A/B/C ÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ­ÃÂÃÂÃÂÃÂÃÂÃÂªÃÂÃÂ°ÃÂÃÂ ÃÂÃÂ«ÃÂÃÂ°ÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂ©ÃÂÃÂ´ ÃÂÃÂªÃÂÃÂ°ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ 60ÃÂÃÂ«ÃÂÃÂ¶ÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂÃÂÃÂ ÃÂÃÂªÃÂÃÂ¸ÃÂÃÂ°ÃÂÃÂ«ÃÂÃÂ¡ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ ÃÂÃÂ«ÃÂÃÂÃÂÃÂ¨ÃÂÃÂªÃÂÃÂ¸ÃÂÃÂ´ÃÂÃÂ«ÃÂÃÂÃÂÃÂ¤.
    cutoff = now - timedelta(minutes=EPISODE_GAP_MIN)
    recent_keys = set()
    for old in hist:
        try:
            ots = datetime.fromisoformat(str(old.get("ts", "")).replace("Z", "+00:00"))
            if ots >= cutoff:
                recent_keys.add((
                    old.get("market"),
                    bool(old.get("A")),
                    bool(old.get("B")),
                    bool(old.get("C")),
                ))
        except Exception:
            pass

    recs = []
    for q in selected:
        key = (q.get("market"), bool(q.get("A")), bool(q.get("B")), bool(q.get("C")))
        if key in recent_keys:
            continue
        recs.append(q)
        recent_keys.add(key)

    hist = recs + hist
    hist = hist[:MAX_ROWS]
    OUT.write_text(
        json.dumps(hist, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    latest = {
        "updated_at": now.isoformat(),
        "version": "V7.6 ALL_KRW",
        "scan_universe": "ALL_KRW",
        "scanned_markets": len(rows),
        "market_count": len(ms),
        "top_n": TOP_N,
        "history_rank_n": HISTORY_RANK_N,
        "early_diag_score": EARLY_DIAG_SCORE,
        "episode_gap_min": EPISODE_GAP_MIN,
        "new_history_records": len(recs),
        "rank_history": {
            "enabled": True,
            "directory": str(SNAPSHOT_HISTORY_DIR),
            "file": str(snapshot_path),
            "format": "jsonl",
            "retention_days": SNAPSHOT_RETENTION_DAYS,
            "markets_logged_this_run": len(snapshot_rows) if full_snapshot_written else 0,
            "full_snapshot_written": full_snapshot_written,
            "full_snapshot_interval_min": 60,
            "note": "Full-market diagnostic snapshot is hourly; fast candidate history remains enabled. This does not alter V7.6 scoring, labels, ranking, or TOP selection.",
        },
        "btc": btc,
        "top": top,
        "outside_top120_candidates": outside_candidates[:30],
        "outside_top120_candidate_count": len(outside_candidates),
        "note": (
            "V7.3 score/EARLY ÃÂÃÂªÃÂÃÂ·ÃÂÃÂÃÂÃÂ¬ÃÂÃÂ¹ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ ÃÂÃÂªÃÂÃÂ·ÃÂÃÂ¸ÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂ¡ÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂ§ÃÂÃÂ. "
            "24h ÃÂÃÂªÃÂÃÂ±ÃÂÃÂ°ÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂªÃÂÃÂ¸ÃÂÃÂ TOP120 ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂÃÂÃÂ­ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂ§ÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂÃÂÃÂªÃÂÃÂ±ÃÂÃÂ°. "
            "historyÃÂÃÂ«ÃÂÃÂÃÂÃÂ TOP30 + A/B/C ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂÃÂÃÂ¬ÃÂÃÂ²ÃÂÃÂ´ + EARLY score>=65ÃÂÃÂ«ÃÂÃÂ¥ÃÂÃÂ¼ 60ÃÂÃÂ«ÃÂÃÂ¶ÃÂÃÂ episodeÃÂÃÂ«ÃÂÃÂ¡ÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂ ÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ¥. "
            "trade_value_rank_24hÃÂÃÂ«ÃÂÃÂÃÂÃÂ ÃÂÃÂ­ÃÂÃÂÃÂÃÂÃÂÃÂ­ÃÂÃÂÃÂÃÂ°ÃÂÃÂªÃÂÃÂ°ÃÂÃÂ ÃÂÃÂ¬ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂÃÂÃÂ«ÃÂÃÂÃÂÃÂ¼ ÃÂÃÂ«ÃÂÃÂ¹ÃÂÃÂÃÂÃÂªÃÂÃÂµÃÂÃÂÃÂÃÂ¬ÃÂÃÂÃÂÃÂ© ÃÂÃÂªÃÂÃÂ¸ÃÂÃÂ°ÃÂÃÂ«ÃÂÃÂ¡ÃÂÃÂ."
        ),
    }

    LATEST.write_text(
        json.dumps(latest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- Compact monitoring outputs (designed to stay small) ----
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    by_market = {}
    for r in hist:
        m = r.get("market")
        if m:
            by_market.setdefault(m, []).append(r)

    live_coins = {}
    for z in rows:
        m = z["market"]
        hs = summarize_market_history(hist, m, now, f(z.get("price")))
        compact = {
            "price": z.get("price"),
            "score": z.get("score"),
            "label": z.get("label"),
            "A": z.get("A"),
            "B": z.get("B"),
            "C": z.get("C"),
            "ABC_ALL": z.get("ABC_ALL"),
            "ret_5m": z.get("ret_5m"),
            "ret_15m": z.get("ret_15m"),
            "ret_30m": z.get("ret_30m"),
            "ret_60m": z.get("ret_60m"),
            "value_ratio_5m": z.get("value_ratio_5m"),
            "value_accel_15m": z.get("value_accel_15m"),
            "value_accel_30m": z.get("value_accel_30m"),
            "trade_value_rank_24h": z.get("trade_value_rank_24h"),
            "trade_value_24h": z.get("trade_value_24h"),
            **hs,
        }
        compact["monitor_score"] = monitor_score(z, hs)
        live_coins[m] = compact

    live = {
        "generated_at": now.isoformat(),
        "version": "V7.6 MONITOR_COMPACT",
        "market_count": len(live_coins),
        "ignition_definition": {
            "success_threshold_pct": IGNITION_SUCCESS_PCT,
            "horizon_hours": IGNITION_HORIZON_HOURS,
            "note": "Based on future stored episode prices; for ranking/quality filtering, not exact OHLCV backtest.",
        },
        "coins": live_coins,
    }
    LIVE.write_text(json.dumps(live, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    ranked_monitor = sorted(
        live_coins.items(),
        key=lambda kv: (
            -f(kv[1].get("monitor_score")),
            -f(kv[1].get("score")),
            -f(kv[1].get("value_accel_15m")),
        ),
    )[:LIVE_TOP_N]

    top20_items = []
    for m, data in ranked_monitor:
        q = dict(data)
        q["market"] = m
        q["extension"] = daily_extension(m)
        top20_items.append(q)
        time.sleep(0.08)

    top20_payload = {
        "generated_at": now.isoformat(),
        "version": "V7.6 MONITOR_TOP20",
        "count": len(top20_items),
        "ranking_note": "monitor_score rewards current signal strength and historical ignition quality; CHASE/weak-repeat patterns are penalized.",
        "items": top20_items,
    }
    TOP20.write_text(json.dumps(top20_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Independent chart/supply-demand scan. Failure here must never break V7.6.
    try:
        market_top5 = independent_market_scan(ms, rank24, value24, now)
        print("independent market scan top5:", [x["market"] for x in market_top5])
    except Exception as e:
        print("independent market scan failed:", e)

    # Per-market episode files: only markets present in retained history.
    for m, items in by_market.items():
        items = sorted(items, key=lambda r: str(r.get("ts", "")), reverse=True)[:PER_MARKET_HISTORY_MAX]
        safe_name = m.replace("KRW-", "") + ".json"
        (HISTORY_DIR / safe_name).write_text(
            json.dumps({
                "market": m,
                "generated_at": now.isoformat(),
                "episodes": items,
            }, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    print("done")
    print("markets:", len(ms))
    print("feature rows:", len(rows))
    print("outside top120 A/B/C candidates:", len(outside_candidates))
    print("history selected:", len(selected), "new records:", len(recs))
    print(
        "rank snapshot:",
        snapshot_path,
        "markets:",
        len(snapshot_rows) if full_snapshot_written else 0,
    )
    for x in top:
        print(
            x["market"],
            x["label"],
            x["score"],
            "rank24=", x.get("trade_value_rank_24h"),
            "OUT120" if x.get("outside_top120") else "",
        )


if __name__ == "__main__":
    main()
