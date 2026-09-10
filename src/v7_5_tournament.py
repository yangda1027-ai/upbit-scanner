#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V7.5 Project Phoenix - 3 Strategy Forward Tournament + ABC overlap
- V7.3 로직/점수는 수정하지 않음
- A/B/C 전략은 그대로 유지
- ABC 동시충족은 "별도 통계 태그"로만 집계
- 첫 실행 시각 이전 = Historical(In-sample)
- 첫 실행 시각 이후 = Official Forward
- 표시용 시간은 KST
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "data" / "v7_3_signals.json"
OUT = ROOT / "docs" / "data" / "v7_5_tournament.json"

KST = timezone(timedelta(hours=9))
EPISODE_GAP_MINUTES = 60


def f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def to_kst_string(dt: datetime) -> str:
    return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def is_chase(r: Dict[str, Any]) -> bool:
    return (
        f(r.get("ret_5m")) >= 3.0
        or f(r.get("ret_15m")) >= 5.0
        or f(r.get("ret_60m")) >= 8.0
    )


def strategy_a(r: Dict[str, Any]) -> bool:
    return (
        r.get("label") == "EARLY"
        and not is_chase(r)
        and f(r.get("score")) >= 85.0
        and f(r.get("value_accel_15m")) >= 3.0
        and f(r.get("value_accel_30m")) >= 2.0
    )


def strategy_b(r: Dict[str, Any]) -> bool:
    return (
        r.get("label") == "EARLY"
        and not is_chase(r)
        and f(r.get("score")) >= 80.0
        and f(r.get("value_accel_15m")) >= 1.5
        and f(r.get("value_accel_30m")) >= 3.0
    )


def strategy_c(r: Dict[str, Any]) -> bool:
    r5 = f(r.get("ret_5m"))
    r15 = f(r.get("ret_15m"))
    r30 = f(r.get("ret_30m"))

    return (
        r.get("label") == "EARLY"
        and not is_chase(r)
        and f(r.get("score")) >= 70.0
        and f(r.get("value_accel_15m")) >= 2.0
        and f(r.get("value_accel_30m")) >= 2.0
        and -1.5 <= r5 <= 0.5
        and -2.0 <= r15 <= 1.0
        and r30 <= 0.5
    )


def strategy_abc(r: Dict[str, Any]) -> bool:
    return strategy_a(r) and strategy_b(r) and strategy_c(r)


STRATEGIES: Dict[str, Dict[str, Any]] = {
    "A": {
        "name": "Legacy 85/3/2",
        "description": "SCORE≥85 · 15m가속≥3 · 30m가속≥2",
        "fn": strategy_a,
    },
    "B": {
        "name": "30m Flow",
        "description": "SCORE≥80 · 15m가속≥1.5 · 30m가속≥3",
        "fn": strategy_b,
    },
    "C": {
        "name": "Compression + Flow",
        "description": "SCORE≥70 · 가격눌림/정체 · 15m가속≥2 · 30m가속≥2",
        "fn": strategy_c,
    },
}


def outcome(r: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    v = r.get(key)
    if not isinstance(v, dict):
        return None

    return {
        "close_pct": v.get("close_pct"),
        "max_pct": v.get("max_pct"),
        "min_pct": v.get("min_pct"),
        "hit_3": v.get("hit_3"),
        "hit_5": v.get("hit_5"),
        "hit_10": v.get("hit_10"),
    }


def classify(x: Dict[str, Any]) -> str:
    h6 = x.get("h6")
    h24 = x.get("h24")

    if not isinstance(h6, dict) or h6.get("max_pct") is None:
        return "TRACKING"

    h6max = f(h6.get("max_pct"))

    if h6max >= 3.0:
        return "FAST_MOVE"

    if not isinstance(h24, dict) or h24.get("max_pct") is None:
        return "SLEEPING"

    h24max = f(h24.get("max_pct"))

    if h24max >= 10.0:
        return "PHOENIX_10"
    if h24max >= 5.0:
        return "PHOENIX_5"
    if h24max >= 3.0:
        return "PHOENIX_3"

    return "FAILED_LATE"


def row_to_signal(r: Dict[str, Any], strategy_id: str) -> Dict[str, Any]:
    ts = parse_ts(r["ts"])

    x = {
        "strategy": strategy_id,
        "market": r.get("market"),
        "price": r.get("price"),
        "score": r.get("score"),
        "source_label": r.get("label"),
        "ret_5m": r.get("ret_5m"),
        "ret_15m": r.get("ret_15m"),
        "ret_30m": r.get("ret_30m"),
        "ret_60m": r.get("ret_60m"),
        "value_ratio_5m": r.get("value_ratio_5m"),
        "value_accel_15m": r.get("value_accel_15m"),
        "value_accel_30m": r.get("value_accel_30m"),
        "ts": r.get("ts"),
        "ts_kst": to_kst_string(ts),
        "rank": r.get("rank"),
        "btc": r.get("btc"),
        "abc_all": strategy_abc(r),
        "h1": outcome(r, "h1"),
        "h3": outcome(r, "h3"),
        "h6": outcome(r, "h6"),
        "h24": outcome(r, "h24"),
    }

    x["status"] = classify(x)
    return x


def build_episodes(
    rows: List[Dict[str, Any]],
    sid: str,
    fn: Callable[[Dict[str, Any]], bool]
) -> List[Dict[str, Any]]:

    matched = [r for r in rows if isinstance(r, dict) and fn(r)]
    matched.sort(key=lambda r: parse_ts(r["ts"]))

    episodes: List[Dict[str, Any]] = []
    last_by_market: Dict[str, datetime] = {}

    for r in matched:
        market = r.get("market")
        ts_raw = r.get("ts")

        if not market or not ts_raw:
            continue

        ts = parse_ts(ts_raw)
        prev = last_by_market.get(market)

        if prev is not None and (ts - prev) <= timedelta(minutes=EPISODE_GAP_MINUTES):
            last_by_market[market] = ts
            continue

        last_by_market[market] = ts
        episodes.append(row_to_signal(r, sid))

    return episodes


def pct(n: int, d: int) -> Optional[float]:
    return round(100.0 * n / d, 2) if d else None


def stats_for(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    h6_done = [
        x for x in signals
        if isinstance(x.get("h6"), dict)
        and x["h6"].get("max_pct") is not None
    ]

    fast = [
        x for x in h6_done
        if f(x["h6"].get("max_pct")) >= 3.0
    ]

    sleeper_eval = [
        x for x in signals
        if isinstance(x.get("h6"), dict)
        and x["h6"].get("max_pct") is not None
        and f(x["h6"].get("max_pct")) < 3.0
        and isinstance(x.get("h24"), dict)
        and x["h24"].get("max_pct") is not None
    ]

    hit3 = [x for x in sleeper_eval if f(x["h24"].get("max_pct")) >= 3.0]
    hit5 = [x for x in sleeper_eval if f(x["h24"].get("max_pct")) >= 5.0]
    hit10 = [x for x in sleeper_eval if f(x["h24"].get("max_pct")) >= 10.0]

    return {
        "episodes": len(signals),
        "h6_matured": len(h6_done),
        "fast_move_n": len(fast),
        "fast_move_rate_pct": pct(len(fast), len(h6_done)),
        "sleeper_evaluated_n": len(sleeper_eval),
        "sleeper_hit3_n": len(hit3),
        "sleeper_hit3_pct": pct(len(hit3), len(sleeper_eval)),
        "sleeper_hit5_n": len(hit5),
        "sleeper_hit5_pct": pct(len(hit5), len(sleeper_eval)),
        "sleeper_hit10_n": len(hit10),
        "sleeper_hit10_pct": pct(len(hit10), len(sleeper_eval)),
    }


def load_or_create_forward_start() -> str:
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text(encoding="utf-8"))
            fs = old.get("meta", {}).get("forward_start_utc")
            if fs:
                return fs
        except Exception:
            pass

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"V7.3 데이터가 없습니다: {SRC}")

    rows = json.loads(SRC.read_text(encoding="utf-8"))

    if not isinstance(rows, list):
        raise ValueError("v7_3_signals.json 최상위 구조가 list가 아닙니다.")

    forward_start_utc = load_or_create_forward_start()
    forward_start = parse_ts(forward_start_utc)

    strategy_output: Dict[str, Any] = {}
    all_signals: List[Dict[str, Any]] = []

    for sid, info in STRATEGIES.items():
        episodes = build_episodes(rows, sid, info["fn"])

        historical = [
            x for x in episodes
            if parse_ts(x["ts"]) < forward_start
        ]

        forward = [
            x for x in episodes
            if parse_ts(x["ts"]) >= forward_start
        ]

        strategy_output[sid] = {
            "name": info["name"],
            "description": info["description"],
            "historical_in_sample": stats_for(historical),
            "official_forward": stats_for(forward),
        }

        all_signals.extend(episodes)

    # ABC overlap: separate observation only, does not change A/B/C logic
    abc_episodes = build_episodes(rows, "ABC", strategy_abc)

    abc_historical = [
        x for x in abc_episodes
        if parse_ts(x["ts"]) < forward_start
    ]

    abc_forward = [
        x for x in abc_episodes
        if parse_ts(x["ts"]) >= forward_start
    ]

    abc_stats = {
        "historical_in_sample": stats_for(abc_historical),
        "official_forward": stats_for(abc_forward),
    }

    all_signals.sort(
        key=lambda x: parse_ts(x["ts"]),
        reverse=True
    )

    eligible = []

    for sid, s in strategy_output.items():
        st = s["official_forward"]

        if (
            st["sleeper_evaluated_n"] >= 30
            and st["sleeper_hit5_pct"] is not None
        ):
            eligible.append((st["sleeper_hit5_pct"], sid))

    leader = None

    if eligible:
        eligible.sort(reverse=True)
        leader = eligible[0][1]

    now_utc = datetime.now(timezone.utc).replace(microsecond=0)

    payload = {
        "meta": {
            "generated_at_utc": now_utc.isoformat(),
            "generated_at_kst": to_kst_string(now_utc),
            "forward_start_utc": forward_start_utc,
            "forward_start_kst": to_kst_string(forward_start),
            "episode_gap_minutes": EPISODE_GAP_MINUTES,
            "source": "docs/data/v7_3_signals.json",
            "note": "Official Forward는 forward_start 이후 신호만 포함.",
            "h24_note": "h24는 V7.3 evaluator가 기록한 기존 정의를 그대로 사용.",
            "leader_rule": "공식 sleeper_evaluated_n >= 30인 전략만 +5% 적중률로 임시 리더 선정",
            "leader": leader,
            "abc_note": "ABC는 A/B/C 세 조건을 모두 만족한 별도 관찰 집계이며 A/B/C 전략 자체는 변경하지 않음.",
        },
        "strategies": strategy_output,
        "abc_overlap": abc_stats,
        "signals": all_signals[:1000],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)

    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("=== V7.5 Phoenix Tournament ===")
    print("Generated KST:", payload["meta"]["generated_at_kst"])
    print("Forward start KST:", payload["meta"]["forward_start_kst"])
    print("Leader:", leader)
    print("ABC forward episodes:", abc_stats["official_forward"]["episodes"])
    print("Output:", OUT)


if __name__ == "__main__":
    main()
