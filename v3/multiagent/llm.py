"""OpenAI 채팅 클라이언트 + 예산 연동.

v2와 달리 매 호출이 BudgetTracker를 거친다 —
비용 상한을 넘는 순간 파이프라인 전체가 멈춘다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from .config import Settings, get_settings
from .guardrails import BudgetTracker, with_retry

logger = logging.getLogger(__name__)


class LLMClient:
    """AsyncOpenAI 래퍼 — chat/chat_json + 사용량을 BudgetTracker에 기록.

    입력: Settings, BudgetTracker
    출력: chat()/chat_json()
    동작: _ensure_client()로 lazy 초기화, chat()에 재시도 데코레이터 적용.
    usage가 응답에 있으면 실제 토큰, 없으면 문자수/4로 추정한다.
    """

    def __init__(self, settings: Settings | None = None, budget: BudgetTracker | None = None) -> None:
        """입력: settings, budget / 출력: 인스턴스.

        동작: budget이 없으면 무제한 BudgetTracker를 만든다(테스트용).
        """
        self._settings = settings or get_settings()
        self.budget = budget or BudgetTracker(
            max_cost_usd=999.0, max_total_tokens=10_000_000, model=self._settings.agent_model
        )
        self._client: AsyncOpenAI | None = None

    def _ensure_client(self) -> AsyncOpenAI:
        """입력: 없음 / 출력: AsyncOpenAI. 최초 호출 시 생성."""
        if self._client is None:
            api_key = self._settings.openai_api_key.get_secret_value()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다 (.env 참고)")
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._settings.openai_base_url,
                timeout=90.0,
                max_retries=0,  # 재시도는 with_retry가 담당 (이중 재시도 방지)
            )
        return self._client

    @with_retry(max_retries=2)
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """입력: messages, 선택 파라미터 / 출력: assistant content.

        동작:
        1. budget.check() — 이미 상한이면 API 호출 없이 BudgetExceeded.
        2. chat.completions.create 호출 (with_retry로 일시 오류 재시도).
        3. usage를 budget.track()에 기록 — 초과 시 즉시 BudgetExceeded.
        """
        self.budget.check()  # 호출 전 예산 확인
        params: dict[str, Any] = {
            "model": self._settings.agent_model,
            "messages": messages,
            "temperature": self._settings.agent_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self._settings.agent_max_tokens,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        resp = await self._ensure_client().chat.completions.create(**params)

        # 사용량 기록 — usage가 없으면 추정
        usage = getattr(resp, "usage", None)
        if usage:
            self.budget.track(usage.prompt_tokens or 0, usage.completion_tokens or 0)
        else:
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            self.budget.track(total_chars // 4, len(resp.choices[0].message.content or "") // 4)

        return resp.choices[0].message.content or ""

    async def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """입력: messages / 출력: 파싱된 dict.

        동작: json_mode로 호출 후 파싱. 코드펜스 벗기기 포함.
        """
        text = await self.chat(messages, json_mode=True, **kwargs)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)

    async def aclose(self) -> None:
        """입력: 없음 / 출력: 없음."""
        if self._client is not None:
            await self._client.close()
