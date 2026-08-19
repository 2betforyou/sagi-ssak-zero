"""정적 HTML 추출이 실패하거나 정보가 부족할 때(대부분 JS 렌더링 페이지)
헤드리스 브라우저로 실제 렌더링된 화면을 캡처해:
  1) 사용자에게 그대로 보여주고(최소한 눈으로는 확인 가능하게),
  2) ANTHROPIC_API_KEY가 있으면 Claude vision으로 정보를 추출하는 폴백.

playwright는 무거운 선택적 의존성이라, 설치가 안 돼 있거나 브라우저 바이너리가
없으면 조용히 None/빈 값을 반환한다 — 이 모듈이 없어도 앱의 나머지 기능은
정상 동작해야 한다.
"""
from __future__ import annotations

import base64
import json
import os


def capture_screenshot(url: str, timeout_ms: int = 15000) -> bytes | None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 2000})
                page.goto(url, timeout=timeout_ms, wait_until="networkidle")
                return page.screenshot(full_page=True)
            finally:
                browser.close()
    except Exception:
        return None


def extract_from_screenshot(png_bytes: bytes) -> dict:
    """화면에 실제로 보이는 정보만 뽑도록 강제 — 근거 없는 값은 절대 만들지 않는다."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {}
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        b64 = base64.b64encode(png_bytes).decode()
        prompt = (
            "이 이미지는 쇼핑몰 상품 페이지 스크린샷이야. 화면에 실제로 보이는 정보만 사용해서 "
            "다음 필드를 추출해줘. 화면에 없으면 절대 추측하지 말고 null로 남겨.\n"
            'JSON만 출력: {"product_name": string|null, "price": number|null, '
            '"store_display_name": string|null, "business_name": string|null, '
            '"business_reg_number": string|null, "business_address": string|null, '
            '"payment_notice_text": string|null}'
        )
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": b64},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return json.loads(resp.content[0].text)
    except Exception:
        return {}
