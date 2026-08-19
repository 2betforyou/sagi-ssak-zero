"""사기싹제로 위험도 스코어링 엔진.

설계 원칙 (docs 참고): LLM은 evidence(근거) 추출만 담당하고, 점수 계산은
`risk_config.json`을 읽는 이 결정론적 엔진이 전담한다. LLM이 점수를 직접
산출하지 않도록 분리해 재현성과 설명가능성을 확보한다.

evidence 형식: {signal_id: {"present": bool, "source": str, ...extra}}
  - source는 risk_config.json의 confidence_by_source 키 중 하나여야 한다.
  - "No Evidence, No Accusation" 원칙에 따라 signal이 허용하지 않는 출처
    (allowed_sources)로 들어온 evidence는 엔진이 거부한다 — 근거 등급이
    낮은 정보로 identity mismatch 같은 강한 주장을 하지 못하게 막기 위함.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).parent / "risk_config.json"


@dataclass
class SignalContribution:
    signal_id: str
    label_ko: str
    category: str
    source: str
    confidence: float
    base_weight: float
    severity_multiplier: float
    contribution: float  # base_weight * confidence * severity_multiplier
    detail: str | None = None
    citation: str | None = None


@dataclass
class RiskResult:
    score: float
    level: str
    raw_score: float
    hard_gate_triggered: list[str] = field(default_factory=list)
    contributions: list[SignalContribution] = field(default_factory=list)


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _confidence(config: dict, source: str) -> float:
    table = config["confidence_by_source"]
    if source not in table:
        raise ValueError(f"알 수 없는 evidence source: {source!r} (허용값: {sorted(table)})")
    return table[source]


def _severity_multiplier(signal_cfg: dict, evidence_entry: dict) -> float:
    tiers = signal_cfg.get("severity_tiers")
    if not tiers:
        return 1.0

    deviation = evidence_entry.get("deviation_pct_below_median")
    if deviation is None:
        raise ValueError(
            f"signal '{signal_cfg['id']}'은 severity_tiers를 사용하므로 "
            "evidence에 deviation_pct_below_median이 필요합니다."
        )

    applicable = [t for t in tiers if deviation >= t["deviation_pct_below_median"]]
    if not applicable:
        return 0.0  # 최소 임계치 미만이면 이상치로 보지 않음
    best = max(applicable, key=lambda t: t["deviation_pct_below_median"])
    return best["weight_multiplier"]


def _level_for(config: dict, score: float) -> str:
    for band in config["risk_levels"]:
        if band["min"] <= score <= band["max"]:
            return band["level"]
    raise ValueError(f"score {score}에 해당하는 risk_level 구간이 없습니다.")


def _score_weighted_signals(
    config: dict, evidence: dict[str, dict], include_disabled: bool
) -> tuple[float, list[SignalContribution]]:
    signal_by_id = {s["id"]: s for s in config["weighted_signals"]}
    unknown = set(evidence) - signal_by_id.keys() - {g["id"] for g in config["hard_gates"]}
    if unknown:
        raise ValueError(f"risk_config.json에 정의되지 않은 signal_id: {sorted(unknown)}")

    raw = 0.0
    contributions: list[SignalContribution] = []
    for signal_id, signal_cfg in signal_by_id.items():
        entry = evidence.get(signal_id)
        if not entry or not entry.get("present"):
            continue
        if not signal_cfg.get("mvp", True) and not include_disabled:
            continue  # MVP 비활성 signal은 evidence가 있어도 무시

        source = entry["source"]
        allowed = signal_cfg["allowed_sources"]
        if source not in allowed:
            raise ValueError(
                f"signal '{signal_id}'는 {allowed} 출처만 허용하는데 "
                f"'{source}'로 들어온 evidence는 근거 등급 미달로 거부됩니다."
            )

        confidence = _confidence(config, source)
        multiplier = _severity_multiplier(signal_cfg, entry)
        contribution = signal_cfg["base_weight"] * confidence * multiplier
        raw += contribution

        contributions.append(
            SignalContribution(
                signal_id=signal_id,
                label_ko=signal_cfg["label_ko"],
                category=signal_cfg["category"],
                source=source,
                confidence=confidence,
                base_weight=signal_cfg["base_weight"],
                severity_multiplier=multiplier,
                contribution=round(contribution, 2),
                detail=entry.get("detail"),
                citation=entry.get("citation"),
            )
        )

    return raw, contributions


def _check_hard_gates(config: dict, evidence: dict[str, dict]) -> tuple[list[str], float]:
    triggered = []
    floor = 0.0
    for gate in config["hard_gates"]:
        entry = evidence.get(gate["id"])
        if not entry or not entry.get("present"):
            continue
        if entry["source"] != gate["required_source"]:
            raise ValueError(
                f"hard gate '{gate['id']}'는 '{gate['required_source']}' 출처로만 "
                f"발동 가능한데 '{entry['source']}'로 들어왔습니다."
            )
        triggered.append(gate["id"])
        floor = max(floor, gate["min_score"])
    return triggered, floor


def evaluate(
    evidence: dict[str, dict],
    config: dict | None = None,
    include_disabled: bool = False,
) -> RiskResult:
    config = config or load_config()

    raw, contributions = _score_weighted_signals(config, evidence, include_disabled)
    gates_triggered, floor = _check_hard_gates(config, evidence)

    lo, hi = config["score_range"]["min"], config["score_range"]["max"]
    score = max(min(raw, hi), lo)
    if gates_triggered:
        score = max(score, floor)
    score = round(score, 1)

    return RiskResult(
        score=score,
        level=_level_for(config, score),
        raw_score=round(raw, 2),
        hard_gate_triggered=gates_triggered,
        contributions=sorted(contributions, key=lambda c: c.contribution, reverse=True),
    )


def counterfactuals(
    evidence: dict[str, dict],
    config: dict | None = None,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """'이 신호가 해소되면 위험도가 얼마나 낮아지는가'를 계산.

    현재 위험을 높이고 있는(contribution > 0) signal을 하나씩 제거했을 때
    점수/등급이 어떻게 바뀌는지 비교해, 영향이 큰 순서로 반환한다.
    """
    config = config or load_config()
    base = evaluate(evidence, config)

    scenarios = []
    for c in base.contributions:
        if c.contribution <= 0:
            continue
        hypothetical = {k: v for k, v in evidence.items() if k != c.signal_id}
        result = evaluate(hypothetical, config)
        scenarios.append(
            {
                "signal_id": c.signal_id,
                "label_ko": c.label_ko,
                "score_delta": round(result.score - base.score, 1),
                "score_if_resolved": result.score,
                "level_if_resolved": result.level,
            }
        )

    scenarios.sort(key=lambda s: s["score_delta"])
    return scenarios[:top_n]


if __name__ == "__main__":
    example_evidence = {
        "identity_mismatch": {
            "present": True,
            "source": "official_api",
            "detail": "스토어 표시명 '전자랜드' vs 국세청 등록 사업자명 'ABC상사'",
        },
        "price_anomaly_unexplained": {
            "present": True,
            "source": "web_search_cited",
            "deviation_pct_below_median": 22.9,
            "detail": "동일 SKU 시장 중앙값 489,000원 대비 377,000원",
            "citation": "https://example.com/price-compare",
        },
        "business_very_new": {
            "present": True,
            "source": "official_api",
            "detail": "사업자등록 21일 경과",
        },
        "platform_payment_protection_used": {
            "present": True,
            "source": "platform_listed_info",
        },
    }

    result = evaluate(example_evidence)
    print(f"score={result.score} level={result.level} raw={result.raw_score}")
    for c in result.contributions:
        print(f"  {c.signal_id:<32} {c.contribution:+.1f}  ({c.source}, x{c.severity_multiplier})")

    print("\n반사실 설명 (counterfactual):")
    for cf in counterfactuals(example_evidence):
        print(f"  '{cf['label_ko']}' 해소 시 → {cf['score_if_resolved']} ({cf['level_if_resolved']}), Δ{cf['score_delta']}")
