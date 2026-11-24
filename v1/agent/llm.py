"""OpenAI 채팅 클라이언트 래퍼.

역할 두 가지:
1. OpenAI 응답을 우리가 정의한 Pydantic 모델(LLMTurn)로 정규화한다.
   → 에이전트 루프는 OpenAI 타입에 의존하지 않아 테스트에서 가짜로 교체 가능.
2. 토큰 사용량을 누적하고 USD 비용을 계산한다.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .config import Settings, get_settings

logger = logging.getLogger(__name__)

# USD / 1M tokens: (input, output). 모델명이 prefix로 매칭된다.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
}
DEFAULT_PRICING = MODEL_PRICING["gpt-4o-mini"]


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """입력: model(모델명), prompt_tokens, completion_tokens.
    출력: USD 비용(float).

    동작: MODEL_PRICING을 긴 prefix부터 매칭해 단가를 찾는다.
    등록되지 않은 모델은 DEFAULT_PRICING(gpt-4o-mini)으로 계산하고
    경고 로그를 남긴다 — 미등록 모델의 비용을 0으로 처리하지 않기 위함.
    """
    pricing = DEFAULT_PRICING
    for prefix in sorted(MODEL_PRICING, key=len, reverse=True):
        if model.startswith(prefix):
            pricing = MODEL_PRICING[prefix]
            break
    else:
        logger.warning("가격표에 없는 모델 %s — gpt-4o-mini 가격으로 추정", model)
    in_price, out_price = pricing
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000


class ToolCallSpec(BaseModel):
    """모델이 요청한 툴 호출 하나 (정규화된 형태).

    입력: OpenAI tool_calls 항목 → _normalize()가 변환
    출력: id/name/arguments를 가진 검증된 객체
    동작: arguments는 JSON '문자열'로 유지한다 — execute_tool이 파싱 책임을 진다.
    """

    id: str
    name: str
    arguments: str = "{}"  # JSON 문자열 원문


class LLMTurn(BaseModel):
    """정규화된 모델 응답 한 턴.

    입력: OpenAI ChatCompletion message + usage + finish_reason
    출력: content(텍스트) XOR tool_calls(호출 목록) + 토큰 사용량
    동작: content와 tool_calls는 동시에 올 수 있다(설명 텍스트+호출).
    에이전트 루프는 이 객체만 보고 분기하므로 OpenAI 타입 의존이 없다.
    """

    content: str | None = None
    tool_calls: list[ToolCallSpec] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None


# --- 스트리밍 이벤트 ---


@dataclass
class TextDelta:
    """스트림으로 도착한 텍스트 조각 하나.

    입력: OpenAI chunk의 delta.content
    출력: delta 문자열
    동작: yield 단위 이벤트. 누적은 호출자(Agent.run_stream)가 한다.
    """

    delta: str


@dataclass
class ToolCallDelta:
    """스트림으로 도착한 툴 호출 조각 (인자는 분할되어 온다).

    입력: OpenAI chunk의 delta.tool_calls[i]
    출력: index/name/arguments_delta 조각
    동작: 같은 index끼리 누적해야 완성된 호출이 된다.
    TurnComplete 시점에는 이미 조립된 ToolCallSpec 목록이 나온다.
    """

    index: int
    name: str | None
    arguments_delta: str


@dataclass
class TurnComplete:
    """스트림 종료 이벤트. 조립이 끝난 LLMTurn을 운반한다.

    입력: 누적된 content 조각 + tool_acc + usage
    출력: 완성된 LLMTurn
    동작: stream()의 마지막 yield. Agent는 이 이벤트에서만 턴을 확정한다.
    """

    turn: LLMTurn


StreamEvent = TextDelta | ToolCallDelta | TurnComplete


class ChatClient(Protocol):
    """Agent가 요구하는 최소 클라이언트 인터페이스 (테스트에서 Fake로 교체).

    이 Protocol이 DI의 핵심 — Agent는 OpenAIChatClient 구체 타입을 모르고
    complete()/stream() 두 메서드만 있으면 된다.
    """

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMTurn:
        """입력: messages(OpenAI 형식 dict 목록), tools(스키마, None이면 비활성).
        출력: 정규화된 LLMTurn 한 개.
        동작: 비스트리밍 1회 호출. 네트워크/인증 오류는 예외로 전파한다.
        """
        ...

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[StreamEvent]:
        """입력: complete()와 동일.
        출력: TextDelta/ToolCallDelta 조각들 + 마지막에 TurnComplete.
        동작: async generator. 반드시 TurnComplete로 끝나야 한다는 계약.
        """
        ...


class OpenAIChatClient:
    """AsyncOpenAI 래퍼. 응답을 LLMTurn으로 정규화한다.

    AsyncOpenAI는 첫 호출 시 lazy하게 생성한다 — API 키 없이도
    모듈 import/서버 기동이 가능해야 하기 때문.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """입력: settings(없으면 get_settings() 싱글턴).
        출력: 클라이언트 인스턴스(아직 네트워크 연결 없음).
        동작: model명을 저장하고 _client는 None으로 둔다 —
        실제 AsyncOpenAI 생성은 _ensure_client()까지 지연된다.
        """
        self._settings = settings or get_settings()
        self.model = self._settings.agent_model
        self._client: AsyncOpenAI | None = None

    def _ensure_client(self) -> AsyncOpenAI:
        """입력: 없음 / 출력: 초기화된 AsyncOpenAI.

        동작: 최초 호출 시에만 생성하고 이후 캐시해 반환한다.
        API 키가 없으면 RuntimeError — /chat에서 503으로 변환된다.
        """
        if self._client is None:
            api_key = self._settings.openai_api_key.get_secret_value()
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다 (.env 참고)")
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._settings.openai_base_url,
                timeout=60.0,
                max_retries=2,
            )
        return self._client

    def _params(self, messages, tools, stream=False):
        """입력: messages, tools(스키마|None), stream(bool).
        출력: chat.completions.create()에 그대로 전달할 kwargs dict.

        동작: tools가 있으면 tool_choice="auto"를 붙인다.
        stream=True면 stream_options.include_usage로 마지막 chunk에
        usage가 오게 한다(비용 계산에 필요).
        """
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self._settings.agent_max_tokens,
            "temperature": self._settings.agent_temperature,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        if stream:
            params["stream"] = True
            params["stream_options"] = {"include_usage": True}
        return params

    @staticmethod
    def _normalize(message, usage, finish_reason) -> LLMTurn:
        """입력: OpenAI message 객체, usage 객체, finish_reason 문자열.
        출력: LLMTurn.

        동작: message.tool_calls를 ToolCallSpec 목록으로 변환한다.
        tc.id가 없으면 index 기반 가짜 id(call_{i})를 부여해
        tool_call_id 매칭 규약을 지킨다. usage 필드가 없으면 0으로 둔다.
        """
        tool_calls = [
            ToolCallSpec(
                id=tc.id or f"call_{i}",
                name=tc.function.name,
                arguments=tc.function.arguments or "{}",
            )
            for i, tc in enumerate(message.tool_calls or [])
        ]
        return LLMTurn(
            content=message.content,
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason=finish_reason,
        )

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMTurn:
        """입력: messages, tools / 출력: LLMTurn.

        동작: 비스트리밍 completions.create 1회 호출 후 _normalize.
        API 오류는 openai 예외로 전파 — 호출자(Agent)가 처리한다.
        """
        resp = await self._ensure_client().chat.completions.create(
            **self._params(messages, tools)
        )
        choice = resp.choices[0]
        return self._normalize(choice.message, resp.usage, choice.finish_reason)

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[StreamEvent]:
        """입력: complete()와 동일 / 출력: StreamEvent의 async iterator.

        동작: 텍스트 델타와 툴콜 조각을 실시간 yield한다.
        OpenAI는 툴콜도 스트림으로 쪼개서 보낸다:
        - tool_calls[i].function.name / arguments 가 조각으로 도착
        - index로 같은 호출끼리 묶어 누적한다(tool_acc).
        스트림이 끝나면 조립된 ToolCallSpec로 TurnComplete를 yield한다.
        usage는 include_usage 옵션으로 마지막 chunk에 도착한다.
        """
        content_parts: list[str] = []
        tool_acc: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        usage = None

        stream = await self._ensure_client().chat.completions.create(
            **self._params(messages, tools, stream=True)
        )
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            for choice in chunk.choices or []:
                delta = choice.delta
                if delta and delta.content:
                    content_parts.append(delta.content)
                    yield TextDelta(delta.content)
                for tc in getattr(delta, "tool_calls", None) or []:
                    idx = tc.index
                    acc = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        acc["id"] += tc.id
                    fn = tc.function
                    name_delta = getattr(fn, "name", None)
                    args_delta = getattr(fn, "arguments", None)
                    if name_delta:
                        acc["name"] += name_delta
                    if args_delta:
                        acc["arguments"] += args_delta
                    yield ToolCallDelta(index=idx, name=name_delta, arguments_delta=args_delta or "")
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

        tool_calls = [
            ToolCallSpec(
                id=acc["id"] or f"call_{idx}",
                name=acc["name"],
                arguments=acc["arguments"] or "{}",
            )
            for idx, acc in sorted(tool_acc.items())
        ]
        yield TurnComplete(
            LLMTurn(
                content="".join(content_parts) or None,
                tool_calls=tool_calls,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                finish_reason=finish_reason,
            )
        )

    async def aclose(self) -> None:
        """입력: 없음 / 출력: 없음.

        동작: 내부 AsyncOpenAI 세션을 닫는다. 클라이언트가 아직
        생성되지 않았으면(_client is None) 아무것도 하지 않는다.
        """
        if self._client is not None:
            await self._client.close()
