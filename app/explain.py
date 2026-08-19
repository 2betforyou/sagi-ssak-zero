"""사기싹제로 XAI 리포트 생성.

risk_engine의 RiskResult(+원본 evidence)를 소비자가 이해할 수 있는 설명으로
변환한다. ANTHROPIC_API_KEY가 설정되어 있으면 Claude가 근거를 바탕으로
자연어 요약문을 생성하고, 없으면 evidence detail을 그대로 이어붙인 템플릿
으로 폴백한다 (해커톤 MVP는 키 없이도 항상 동작해야 함).
"""
from __future__ import annotations

import os

from app.risk_engine import RiskResult, counterfactuals

_ACTION_BY_LEVEL = {
    "LOW": "특별한 조치 없이 진행해도 좋습니다. 다만 결제는 항상 플랫폼 안전결제로 진행하세요.",
    "GUARDED": "몇 가지 확인되지 않은 정보가 있습니다. 판매자에게 사업자정보를 추가로 요청해보는 걸 권장합니다.",
    "CAUTION": "여러 이상 신호가 함께 발견되었습니다. 결제 전 판매자에게 근거 자료(공식 유통 계약서 등)를 요청하고, 가능하면 플랫폼 안전결제만 이용하세요.",
    "HIGH": "위험 신호가 다수 확인되었습니다. 결제를 보류하고, 추가 확인 전까지 계좌이체 등 직접 결제는 하지 마세요.",
    "CRITICAL": "확정적 위험 신호가 확인되었습니다. 결제를 중단하고 플랫폼 고객센터 또는 관련 기관(금융감독원 1332, 경찰 112)에 신고를 권장합니다.",
}


def _template_summary(result: RiskResult) -> str:
    if not result.contributions:
        return "현재까지 확인된 위험 신호가 없습니다."

    top = [c for c in result.contributions if c.contribution > 0][:3]
    if not top:
        return f"확인된 신호는 대부분 긍정적이며, 종합 위험도는 {result.score}점({result.level})입니다."

    reasons = ", ".join(c.label_ko for c in top)
    return f"{reasons} 등의 이유로 종합 위험도가 {result.score}점({result.level})으로 평가되었습니다."


def _llm_summary(result: RiskResult) -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        evidence_lines = "\n".join(
            f"- {c.label_ko} (근거: {c.detail or '상세 없음'})"
            for c in result.contributions
            if c.contribution > 0
        )
        protective_lines = "\n".join(
            f"- {c.label_ko}" for c in result.contributions if c.contribution < 0
        )
        prompt = (
            "너는 소비자 보호 AI야. 아래는 온라인 판매자에 대한 사기 위험도 분석 결과야. "
            "제공된 근거만 사용하고(근거에 없는 내용은 절대 추측하지 말고) 비전문가가 이해할 수 있는 "
            "한국어로 왜 위험한지 4문장 이내로 설명해줘. '사기입니다' 같은 단정적 표현 대신 "
            "'~가능성이 있습니다', '~확인이 필요합니다' 같은 신중한 표현을 사용해.\n\n"
            f"종합 위험도: {result.score}점 ({result.level})\n"
            f"위험 신호:\n{evidence_lines or '해당 없음'}\n\n"
            f"안심 신호:\n{protective_lines or '해당 없음'}"
        )
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return None  # 키가 유효하지 않거나 호출 실패 시 템플릿으로 폴백


def build_report(evidence: dict, result: RiskResult) -> dict:
    summary = _llm_summary(result)
    source = "llm" if summary else "template"
    summary = summary or _template_summary(result)

    return {
        "summary": summary,
        "source": source,
        "score": result.score,
        "level": result.level,
        "action": _ACTION_BY_LEVEL[result.level],
        "hard_gate_triggered": result.hard_gate_triggered,
        "evidence": [
            {
                "signal_id": c.signal_id,
                "label_ko": c.label_ko,
                "category": c.category,
                "contribution": c.contribution,
                "detail": c.detail,
                "citation": c.citation,
                "source": c.source,
            }
            for c in result.contributions
        ],
        "counterfactuals": counterfactuals(evidence),
    }
