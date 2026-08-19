"""사기싹제로 evidence 추출.

판매자 리스팅 원본 정보를 `risk_engine.evaluate()`가 소비하는 evidence
딕셔너리로 변환한다. 확인할 수 없는 신호는 절대 추측하지 않고 present=False
로 남긴다 ("No Evidence, No Accusation" — 근거 없이 의심하지 않는다).

외부 연동(국세청 사업자등록 상태조회 API)은 explain.py의 ANTHROPIC_API_KEY
패턴과 동일하게, 키가 있으면 실제 API를 시도하고 없거나 실패하면 로컬에서
계산 가능한 것만(체크섬 검증 등) 폴백으로 사용한다. 웹 검색이 필요한 신호
(external_fraud_mentions, official_brand_relation_unclear, low_external_footprint,
price_anomaly의 시장가)는 이 모듈 밖에서 검색을 수행한 뒤 구조화된 결과를
넣어주는 것을 전제로 한다 — 이 모듈이 직접 웹을 검색하지는 않는다.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class SellerListing:
    platform: str
    product_name: str
    price: int
    store_display_name: str
    category: str = ""
    description_text: str = ""
    business_name: str | None = None  # 플랫폼 사업자정보 고시에 공개된 상호
    business_reg_number: str | None = None
    business_address: str | None = None  # 플랫폼 사업자정보 고시에 공개된 사업장 주소
    market_median_price: int | None = None  # 외부에서 가격비교 후 주입
    external_reports: list[dict] = field(default_factory=list)
    external_search_performed: bool = False


def _normalize(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", s or "").lower()


# ---------------------------------------------------------------- identity --

def check_identity_mismatch(store_display_name: str, business_name: str | None) -> dict:
    """플랫폼에 사업자정보로 공개된 상호와 스토어 표시명을 비교.

    business_name이 아예 공개되지 않은 경우(플랫폼이 정보를 안 주는 경우)는
    '불일치'가 아니라 '판단 보류'이므로 present=False로 둔다.
    """
    if not business_name:
        return {"present": False}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        llm_result = _identity_mismatch_via_llm(store_display_name, business_name, api_key)
        if llm_result is not None:
            return llm_result

    a, b = _normalize(store_display_name), _normalize(business_name)
    mismatch = a not in b and b not in a
    return {
        "present": mismatch,
        "source": "platform_listed_info",
        "detail": f"스토어 표시명 '{store_display_name}' vs 공개 사업자명 '{business_name}' (문자열 비교)",
    }


def _identity_mismatch_via_llm(store_name: str, business_name: str, api_key: str) -> dict | None:
    """단순 문자열 비교로는 '삼성공식몰' vs '㈜삼성전자판매' 같은 정상 케이스와
    '삼성공식몰' vs 'ABC상사' 같은 위험 케이스를 구분하기 어렵다. LLM에게
    상표/법인 관계 추론을 맡기되, 비교 대상 문자열 자체는 여전히
    platform_listed_info 출처이므로 source는 그대로 유지한다."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "아래 두 이름이 같은 사업 주체를 가리키는지 판단해줘. "
            "브랜드명의 공식 대리점/유통점처럼 합리적으로 연결될 수 있으면 mismatch=false, "
            "아무 관련성을 추론할 수 없으면 mismatch=true로 판단해. "
            'JSON만 출력: {"mismatch": bool, "reason": "한 문장"}\n\n'
            f"스토어 표시명: {store_name}\n"
            f"플랫폼 공개 사업자명: {business_name}"
        )
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = json.loads(resp.content[0].text)
        return {
            "present": bool(parsed["mismatch"]),
            "source": "platform_listed_info",
            "detail": f"{parsed.get('reason', '')} (스토어명 '{store_name}' / 사업자명 '{business_name}')",
        }
    except Exception:
        return None  # 파싱/호출 실패 시 호출부가 문자열 비교 폴백으로 처리


# ------------------------------------------------------------------ address --

# 아파트/빌라 특유의 "OO동 OO호" 표기나 주거용 건물 명칭 — 사업장 주소로는 부자연스러움
_RESIDENTIAL_ADDRESS_PATTERNS = [
    r"\d+\s*동\s*\d+\s*호",
    r"빌라\s*\d*\s*호?",
    r"연립주택",
    r"다세대주택",
    r"아파트",
]
# 이 표현이 함께 있으면 오피스텔/건물이라도 상업용 호실일 가능성이 높아 위험 판정에서 제외
_COMMERCIAL_ADDRESS_HINTS = [r"상가", r"사무실", r"오피스\b", r"지식산업센터", r"층\s*\d+\s*호"]


def check_address_risk(address: str | None) -> dict:
    """사업장 주소가 주거용 건물로 추정되면 위험 신호로 본다.

    MVP는 텍스트 패턴만으로 판별한다 — 실제 건물 용도(상업/주거)는 건축물대장
    API 등으로 확인해야 정확하지만, 아직 그 연동이 없으므로 지나치게 확신하지
    않도록 confidence가 낮은 platform_listed_info 등급으로만 취급한다.
    """
    if not address:
        return {"present": False}

    if any(re.search(p, address) for p in _COMMERCIAL_ADDRESS_HINTS):
        return {"present": False}

    if any(re.search(p, address) for p in _RESIDENTIAL_ADDRESS_PATTERNS):
        return {
            "present": True,
            "source": "platform_listed_info",
            "detail": f"등록 사업장 주소가 주거용 건물로 추정됨: {address}",
        }

    return {"present": False}


# --------------------------------------------------------- business registry --

_BIZ_NO_WEIGHTS = [1, 3, 7, 1, 3, 7, 1, 3, 5]


def _valid_biz_no_checksum(reg_number: str) -> bool:
    """사업자등록번호 체크섬 검증 (공개된 표준 알고리즘). 형식 유효성만 확인할
    뿐, 실제로 등록/유지 중인지는 알 수 없다 — 그래서 이 결과만으로는
    business_registration_invalid hard gate를 발동시키지 않는다."""
    digits = re.sub(r"\D", "", reg_number or "")
    if len(digits) != 10:
        return False
    nums = [int(d) for d in digits]
    total = sum(d * w for d, w in zip(nums[:9], _BIZ_NO_WEIGHTS))
    total += (nums[8] * 5) // 10
    check = (10 - total % 10) % 10
    return check == nums[9]


def _call_nts_status_api(reg_number: str, api_key: str) -> dict:
    """국세청 사업자등록정보 상태조회 API (data.go.kr 서비스키 필요).

    주의: 이 함수는 아직 실제 네트워크로 검증되지 않았다. 데모 전에 진짜
    서비스키로 응답 필드명(b_stt, b_stt_cd 등)을 반드시 재확인할 것.
    """
    digits = re.sub(r"\D", "", reg_number)
    url = f"https://api.odcloud.kr/api/nts-businessman/v1/status?serviceKey={api_key}"
    body = json.dumps({"b_no": [digits]}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        payload = json.loads(resp.read())
    return payload["data"][0]


def check_business_registration(reg_number: str | None) -> dict:
    """hard gate(business_registration_invalid)는 official_api로 '폐업/등록없음'이
    실제 확인된 경우에만 채운다. 체크섬 실패(오타 가능성 등)만으로는 CRITICAL을
    강제하지 않고 참고용 진단 정보(valid_format)로만 남긴다."""
    if not reg_number:
        return {"valid_format": None}

    valid_format = _valid_biz_no_checksum(reg_number)

    api_key = os.environ.get("NTS_API_KEY")
    if api_key:
        try:
            data = _call_nts_status_api(reg_number, api_key)
            status = data.get("b_stt")  # 예: "계속사업자" / "휴업자" / "폐업자"
            return {
                "valid_format": valid_format,
                "verified_status": status,
                "verified": True,
                "invalid_confirmed": status in ("폐업자", None) or data.get("b_stt_cd") is None,
            }
        except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
            pass  # API 실패 시 체크섬 결과만으로 폴백

    return {"valid_format": valid_format, "verified_status": None, "verified": False}


# --------------------------------------------------------------------- price --

def check_price_anomaly(price: int, market_median_price: int | None) -> dict:
    if not market_median_price or market_median_price <= 0:
        return {"present": False}

    deviation = (market_median_price - price) / market_median_price * 100
    if deviation < 15:  # risk_config.json severity_tiers 최소 임계치
        return {"present": False}

    return {
        "present": True,
        "source": "web_search_cited",
        "deviation_pct_below_median": round(deviation, 1),
        "detail": f"시장 중앙값 {market_median_price:,}원 대비 {price:,}원 ({deviation:.1f}% 낮음)",
    }


# ------------------------------------------------------------------- payment --

_OFF_PLATFORM_PATTERNS = [
    r"계좌\s*이체", r"무통장\s*입금", r"카카오\s*(톡|페이)\s*(로|으로)?\s*(문의|연락|결제)",
    r"직\s*거래", r"선\s*입금", r"현금\s*(결제|거래)\s*(시|하면)\s*할인",
    r"안전결제\s*(없이|제외|불가)", r"개인\s*계좌", r"문자\s*(로|주시면)\s*(연락|문의)",
]
_PLATFORM_PROTECTION_PATTERNS = [r"안전결제", r"정품\s*안전결제", r"구매확정"]


def check_payment_risk(description_text: str) -> dict:
    text = description_text or ""
    off_platform = next((p for p in _OFF_PLATFORM_PATTERNS if re.search(p, text)), None)
    protection = any(re.search(p, text) for p in _PLATFORM_PROTECTION_PATTERNS)

    return {
        "off_platform_payment_request": {
            "present": off_platform is not None,
            "source": "platform_listed_info",
            "detail": "상품 설명에서 플랫폼 외부 결제 유도 문구 발견" if off_platform else None,
        },
        "platform_payment_protection_used": {
            "present": protection and off_platform is None,
            "source": "platform_listed_info",
        },
    }


# ------------------------------------------------------------------- external --

def check_external_reports(reports: list[dict], searched: bool) -> dict:
    """웹검색/에이전트 조사는 이 모듈 밖에서 수행하고 결과를
    [{"category": "fraud_mention" | "brand_relation", "citation": url, "detail": str,
      "unclear": bool}] 형태로 넣어준다는 전제. searched=False면(검색을 아예 안 했으면)
    '이력 없음'을 단정하지 않는다 — 안 찾아본 것과 찾아봤는데 없는 것은 다르다."""
    fraud = [r for r in reports if r.get("category") == "fraud_mention"]
    brand = [r for r in reports if r.get("category") == "brand_relation"]

    result: dict[str, dict] = {}
    if fraud:
        result["external_fraud_mentions"] = {
            "present": True,
            "source": "web_search_cited",
            "detail": fraud[0].get("detail"),
            "citation": fraud[0].get("citation"),
        }
    if brand:
        result["official_brand_relation_unclear"] = {
            "present": brand[0].get("unclear", True),
            "source": "web_search_cited",
            "detail": brand[0].get("detail"),
            "citation": brand[0].get("citation"),
        }
    if searched and not fraud:
        result["low_external_footprint"] = {
            "present": True,
            "source": "web_search_cited",
            "detail": "관련 검색에서 판매 이력·평판 정보를 찾지 못함",
        }
    return result


# -------------------------------------------------------------------- driver --

def extract_evidence(listing: SellerListing) -> dict[str, dict]:
    evidence: dict[str, dict] = {}

    evidence["identity_mismatch"] = check_identity_mismatch(
        listing.store_display_name, listing.business_name
    )

    evidence["registered_address_implausible"] = check_address_risk(listing.business_address)

    biz = check_business_registration(listing.business_reg_number)
    if biz.get("verified") and biz.get("invalid_confirmed"):
        evidence["business_registration_invalid"] = {
            "present": True,
            "source": "official_api",
            "detail": f"국세청 상태조회 결과: {biz.get('verified_status')}",
        }
    # valid_format=False(체크섬 실패)는 채점에 반영하지 않고 UI 참고용 경고로만 노출 권장
    # (biz["valid_format"]을 evidence와 별도로 리포트에 같이 넘기면 됨)

    evidence["price_anomaly_unexplained"] = check_price_anomaly(
        listing.price, listing.market_median_price
    )

    evidence.update(check_payment_risk(listing.description_text))

    evidence.update(
        check_external_reports(listing.external_reports, listing.external_search_performed)
    )

    return {k: v for k, v in evidence.items() if v.get("present")}


if __name__ == "__main__":
    from app.risk_engine import counterfactuals, evaluate

    listing = SellerListing(
        platform="롯데ON",
        product_name="정품 노트북 XYZ-15",
        price=377_000,
        store_display_name="전자랜드",
        business_name="ABC상사",
        business_reg_number="123-45-67890",
        business_address="서울시 강북구 미아동 대성빌라 2동 301호",
        market_median_price=489_000,
        description_text="빠른 거래 원하시면 계좌이체로 문의 주세요. 안전결제 없이 진행합니다.",
        external_reports=[
            {"category": "fraud_mention", "citation": "https://example.com/report1",
             "detail": "동일 상호로 유사 사례 제보 다수"},
        ],
        external_search_performed=True,
    )

    evidence = extract_evidence(listing)
    print("evidence:")
    for k, v in evidence.items():
        print(f"  {k}: {v}")

    result = evaluate(evidence)
    print(f"\nscore={result.score} level={result.level}")
    for c in result.contributions:
        print(f"  {c.signal_id:<32} {c.contribution:+.1f}")

    print("\n반사실 설명:")
    for cf in counterfactuals(evidence):
        print(f"  '{cf['label_ko']}' 해소 시 → {cf['score_if_resolved']} ({cf['level_if_resolved']})")
