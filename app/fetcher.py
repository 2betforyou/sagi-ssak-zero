"""구매 링크에서 판매자 리스팅 정보를 자동으로 가져오는 모듈.

전략 (우선순위):
1. 페이지 HTML의 JSON-LD(schema.org Product) / Open Graph 메타태그 파싱 —
   상품명·가격·사이트명 정도는 대부분의 커머스 사이트가 SEO 목적으로
   제공하므로 비교적 안정적으로 가져올 수 있다.
2. 사업자정보 고시(상호·사업자등록번호)는 표준 마크업이 없고 JS로 렌더링
   되는 경우가 많다. 정규식으로 먼저 시도하고, 실패하면 ANTHROPIC_API_KEY가
   있을 때 LLM에게 페이지 원문을 주고 구조화 추출을 맡긴다. 그래도 못 찾으면
   값을 비워두고 절대 추측하지 않는다 — evidence.py가 알아서 해당 signal을
   판단 보류(present=False) 처리한다.

한계: 쿠팡·네이버·롯데ON 등 다수의 국내 커머스 페이지는 JS 렌더링과 봇 차단이
있어 정적 GET 요청만으로는 본문을 거의 못 가져올 수 있다. 이 경우
fetch_listing()은 예외를 던지지 않고 확보된 필드만 채운 SellerListing을
반환하며, 나머지는 사용자가 폼에서 직접 보완하도록 UI에서 처리한다.
"""
from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.evidence import SellerListing

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

_BIZ_LABEL_PATTERNS = {
    "business_name": r"(?:상호(?:명)?|업체명)\s*[:：]\s*([^\n<]{2,40})",
    "business_reg_number": r"사업자\s*등록\s*번호\s*[:：]\s*([0-9\-]{10,12})",
    "business_address": r"(?:사업장\s*소재지|주소)\s*[:：]\s*([^\n<]{5,80})",
}


class FetchError(Exception):
    pass


def fetch_listing(url: str) -> SellerListing:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=8)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise FetchError(f"페이지를 가져오지 못했습니다: {e}") from e

    soup = BeautifulSoup(resp.text, "html.parser")
    platform = urlparse(url).netloc.replace("www.", "")

    product_name, price = _from_json_ld(soup)
    product_name = product_name or _og(soup, "og:title") or ""
    if price is None:
        price = _og_price(soup)

    store_display_name = _og(soup, "og:site_name") or platform
    page_text = soup.get_text("\n")

    business_name = _regex_extract(page_text, "business_name")
    business_reg_number = _regex_extract(page_text, "business_reg_number")
    business_address = _regex_extract(page_text, "business_address")

    if (
        (not business_name or not business_reg_number or not business_address)
        and os.environ.get("ANTHROPIC_API_KEY")
    ):
        llm_fields = _llm_extract_business_info(page_text)
        business_name = business_name or llm_fields.get("business_name")
        business_reg_number = business_reg_number or llm_fields.get("business_reg_number")
        business_address = business_address or llm_fields.get("business_address")

    return SellerListing(
        platform=platform,
        product_name=product_name,
        price=int(price) if price else 0,
        store_display_name=store_display_name,
        business_name=business_name,
        business_reg_number=business_reg_number,
        business_address=business_address,
        description_text=page_text[:3000],
    )


def _og(soup: BeautifulSoup, prop: str) -> str | None:
    tag = soup.find("meta", property=prop)
    return tag["content"].strip() if tag and tag.get("content") else None


def _og_price(soup: BeautifulSoup) -> float | None:
    for prop in ("product:price:amount", "og:price:amount"):
        tag = soup.find("meta", property=prop)
        if tag and tag.get("content"):
            try:
                return float(re.sub(r"[^\d.]", "", tag["content"]))
            except ValueError:
                pass
    return None


def _from_json_ld(soup: BeautifulSoup) -> tuple[str | None, float | None]:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict) and item.get("@type") == "Product":
                name = item.get("name")
                offers = item.get("offers")
                offers = offers[0] if isinstance(offers, list) and offers else offers
                price = offers.get("price") if isinstance(offers, dict) else None
                try:
                    price = float(price) if price is not None else None
                except (TypeError, ValueError):
                    price = None
                return name, price
    return None, None


def _regex_extract(text: str, field: str) -> str | None:
    m = re.search(_BIZ_LABEL_PATTERNS[field], text)
    return m.group(1).strip() if m else None


def fetch_listing_with_fallback(
    url: str,
) -> tuple[SellerListing | None, bytes | None, str | None]:
    """정적 요청으로 핵심 정보(상품명·사업자명)를 하나도 못 건졌을 때만
    헤드리스 브라우저 스크린샷 + vision 추출로 폴백한다 (느리고 무거우므로
    항상 쓰지 않고 필요할 때만). 반환값: (listing, screenshot_png, error_message).
    screenshot_png은 추출 성공 여부와 무관하게 캡처됐으면 항상 반환 — 실패해도
    사용자가 최소한 눈으로 페이지를 확인할 수 있게 하기 위함.
    """
    from app.screenshot import capture_screenshot, extract_from_screenshot

    listing: SellerListing | None = None
    error: str | None = None
    try:
        listing = fetch_listing(url)
    except FetchError as e:
        error = str(e)

    is_thin = listing is None or (not listing.product_name and not listing.business_name)
    if not is_thin:
        return listing, None, error

    screenshot = capture_screenshot(url)
    if not screenshot:
        return listing, None, error or "스크린샷 캡처에도 실패했습니다 (headless 브라우저 미설치 또는 접속 실패)."

    vision_fields = extract_from_screenshot(screenshot)
    if not vision_fields:
        return listing, screenshot, error  # 스크린샷은 있으니 사용자가 눈으로라도 확인 가능

    base = listing or SellerListing(
        platform=urlparse(url).netloc.replace("www.", ""),
        product_name="",
        price=0,
        store_display_name="",
    )
    merged = SellerListing(
        platform=base.platform,
        product_name=base.product_name or vision_fields.get("product_name") or "",
        price=base.price or int(vision_fields.get("price") or 0),
        store_display_name=base.store_display_name or vision_fields.get("store_display_name") or "",
        business_name=base.business_name or vision_fields.get("business_name"),
        business_reg_number=base.business_reg_number or vision_fields.get("business_reg_number"),
        business_address=base.business_address or vision_fields.get("business_address"),
        description_text=(base.description_text or "") + "\n" + (vision_fields.get("payment_notice_text") or ""),
    )
    return merged, screenshot, None


def _llm_extract_business_info(page_text: str) -> dict:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        prompt = (
            "아래는 쇼핑몰 상품 페이지에서 추출한 텍스트야. 여기서 '사업자정보 고시'에 "
            "해당하는 상호(business_name), 사업자등록번호(business_reg_number), "
            "사업장 주소(business_address)를 찾아줘. "
            "본문에 명시적으로 나와 있지 않으면 절대 추측하지 말고 null로 남겨.\n"
            'JSON만 출력해: {"business_name": string|null, "business_reg_number": string|null, '
            '"business_address": string|null}\n\n'
            f"{page_text[:6000]}"
        )
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(resp.content[0].text)
    except Exception:
        return {}
