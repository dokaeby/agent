"""OpenAI 채팅/임베딩 클라이언트.

v1과 달리 여기서는 '문자열 in → 문자열 out' 수준의 단순 인터페이스를 쓴다.
LangGraph 노드는 JSON dict가 아니라 LLM의 최종 텍스트/구조화 결과만 필요로 하기 때문.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """AsyncOpenAI 래퍼 — 채팅 completion + 임베딩 + JSON 추출.

    입력: Settings(없으면 get_settings())
    출력: chat()/embed()/chat_json() 메서드
    동작: AsyncOpenAI는 lazy 생성 — API 키 없이 import 가능.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """입력: settings / 출력: 인스턴스(연결 없음).

        동작: _client는 None — 첫 API 호출 시 _ensure_client()가 생성.
        """
        self._settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None

    def _ensure_client(self) -> AsyncOpenAI:
        """입력: 없음 / 출력: 초기화된 AsyncOpenAI.

        동작: 최초 호출 시 생성 후 캐시. 키 없으면 RuntimeError.
        """
        if self._client is None:
            api_key = self._settings.openai_api_key.get_secret_value()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다 (.env 참고)")
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._settings.openai_base_url,
                timeout=120.0,
                max_retries=2,
            )
        return self._client

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """입력: OpenAI 형식 messages, 선택적 model/temp/max_tokens/json_mode.
        출력: assistant content 문자열.

        동작: chat.completions.create 1회 호출 후 content 반환.
        json_mode=True면 response_format=json_object를 붙인다 —
        프롬프트에 'JSON으로만 답하라' 지시가 함께 있어야 한다.
        """
        params: dict[str, Any] = {
            "model": model or self._settings.agent_model,
            "messages": messages,
            "temperature": self._settings.agent_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self._settings.agent_max_tokens,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        resp = await self._ensure_client().chat.completions.create(**params)
        return resp.choices[0].message.content or ""

    async def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """입력: messages + chat()의 kwargs / 출력: 파싱된 dict.

        동작: json_mode로 호출 후 JSON 파싱. 모델이 가끔 코드펜스를
        붙이므로 ```json ... ``` 블록도 벗겨낸다. 파싱 실패 시 예외 전파.
        """
        text = await self.chat(messages, json_mode=True, **kwargs)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """입력: 문자열 리스트 / 출력: 임베딩 벡터 리스트(입력과 같은 순서).

        동작: embeddings.create를 2048개 단위로 배치 호출한다
        (API 제한 대응). 빈 리스트는 빈 리스트 반환.
        """
        if not texts:
            return []
        vectors: list[list[float]] = []
        client = self._ensure_client()
        model = self._settings.embed_model
        for i in range(0, len(texts), 2048):
            batch = texts[i : i + 2048]
            resp = await client.embeddings.create(model=model, input=batch)
            vectors.extend(item.embedding for item in resp.data)
        return vectors

    async def aclose(self) -> None:
        """입력: 없음 / 출력: 없음. 내부 HTTP 세션을 닫는다."""
        if self._client is not None:
            await self._client.close()
