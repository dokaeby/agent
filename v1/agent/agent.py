"""ReAct 루프: LLM ↔ 툴 호출을 직접 구현한다.

흐름 (턴마다 반복):
  1. messages 전체를 모델에 보낸다 (tools 스키마 포함)
  2. 모델이 tool_calls를 반환하면 → 각 호출을 화이트리스트 검증 후 실행
     → 결과를 role="tool" 메시지로 붙이고 다시 1로
  3. 일반 텍스트를 반환하면 → 종료

LangChain 없이 이 루프만으로 tool-use 에이전트가 동작한다.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, get_settings
from .llm import (
    ChatClient,
    LLMTurn,
    OpenAIChatClient,
    TextDelta,
    ToolCallDelta,
    TurnComplete,
    estimate_cost_usd,
)
from .tools import execute_tool, tool_schemas

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

Rules:
- Use tools when they help answer accurately; never invent tool results.
- After receiving tool results, answer the user's question concisely.
- If a tool returns an error, fix the arguments and retry at most once,
  otherwise explain the limitation.
- Reply in the same language the user writes in."""

MAX_TOOL_RESULT_CHARS = 8_000
"""tool 결과가 이 길이를 넘으면 잘라서 모델에 보낸다 — 컨텍스트 폭주 방지."""


@dataclass
class AgentResult:
    """run()/run_stream()의 최종 반환값.

    입력: 루프가 누적한 content/iterations/토큰/비용/사용된 도구명
    출력: ChatResponse 직렬화와 CLI 출력에 쓰이는 요약 객체
    동작: 토큰은 턴마다 _accumulate로 합산되고, 비용은 루프 종료 시
    _finish_cost가 모델 가격표로 1회 계산한다.
    """

    content: str
    iterations: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    tool_names_used: list[str] = field(default_factory=list)


@dataclass
class StreamCallbacks:
    """스트리밍 도중 외부로 흘려보낼 이벤트를 모으는 버퍼.

    입력: run_stream이 yield하는 이벤트들
    출력: text_parts(누적 텍스트 조각), tool_events(도구 호출 기록)
    동작: 현재 run_stream은 yield로 직접 흘려보내므로 이 버퍼는
    외부 소비자(SSE/웹소켓)가 이벤트를 모을 때 쓰도록 남겨둔 확장점이다.
    """

    text_parts: list[str] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)


class Agent:
    """ReAct 루프 본체. client와 settings를 주입받아 독립적으로 동작한다.

    입력: ChatClient 프로토콜 구현체(실제는 OpenAIChatClient, 테스트는 Fake)
    출력: run()→AgentResult, run_stream()→이벤트 async iterator
    동작: messages를 스스로 관리하며 client.complete/stream을 반복 호출.
    """

    def __init__(
        self,
        client: ChatClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        """입력: client(None이면 OpenAIChatClient 생성), settings(None이면 싱글턴).
        출력: Agent 인스턴스.

        동작: tool_schemas()를 1회 호출해 self._tools에 캐시 —
        매 턴마다 레지스트리를 다시 읽지 않는다.
        """
        self.settings = settings or get_settings()
        self.client = client or OpenAIChatClient(self.settings)
        self._tools = tool_schemas()

    # ---------- 공통 ----------

    def _base_messages(self, user_message: str, history: list[dict] | None) -> list[dict]:
        """입력: user_message(이번 질문), history(이전 대화 [{"role","content"}...]).
        출력: LLM에 보낼 messages 리스트 — [system, *history, user].

        동작: SYSTEM_PROMPT를 항상 맨 앞에 고정하고, 클라이언트가 보낸
        history를 그대로 이어붙인다. 이 리스트는 루프 중 계속 append된다.
        """
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *(history or []),
            {"role": "user", "content": user_message},
        ]

    @staticmethod
    def _assistant_message(turn: LLMTurn) -> dict[str, Any]:
        """입력: LLMTurn / 출력: OpenAI 규약의 assistant 메시지 dict.

        동작: tool_calls가 있으면 {"id","type":"function","function":{...}}
        형태로 포함한다 — 이 형식이어야 다음 턴에서 모델이 자기 호출과
        tool 결과를 올바르게 연결한다. content는 없으면 빈 문자열.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": turn.content or ""}
        if turn.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in turn.tool_calls
            ]
        return msg

    @staticmethod
    def _accumulate(result: AgentResult, turn: LLMTurn) -> None:
        """입력: result(누적 대상), turn(이번 응답).
        출력: 없음(result를 in-place로 갱신).

        동작: prompt/completion 토큰을 턴마다 더한다.
        """
        result.prompt_tokens += turn.prompt_tokens
        result.completion_tokens += turn.completion_tokens

    def _finish_cost(self, result: AgentResult) -> None:
        """입력: 토큰이 누적된 result / 출력: 없음.

        동작: settings.agent_model의 가격표로 총 비용을 계산해
        result.cost_usd에 저장한다. 루프 종료 직전에 1회 호출된다.
        """
        result.cost_usd = estimate_cost_usd(
            self.settings.agent_model, result.prompt_tokens, result.completion_tokens
        )

    async def _run_tool_calls(self, messages: list[dict], turn: LLMTurn, used: list[str]) -> None:
        """입력: messages(누적 리스트), turn(tool_calls 포함), used(도구명 기록용).
        출력: 없음 — messages에 tool 메시지를 append한다.

        동작: turn.tool_calls를 순서대로 execute_tool에 넘기고,
        결과 JSON을 {"role":"tool","tool_call_id":call.id,"content":output}
        형태로 messages에 추가한다. tool_call_id 매칭이 OpenAI 규약의 핵심.
        결과가 MAX_TOOL_RESULT_CHARS를 넘으면 잘라서 컨텍스트를 보호한다.
        """
        for call in turn.tool_calls:
            used.append(call.name)
            logger.info("tool call: %s(%s)", call.name, call.arguments[:200])
            output = await execute_tool(call.name, call.arguments)
            if len(output) > MAX_TOOL_RESULT_CHARS:
                output = output[:MAX_TOOL_RESULT_CHARS] + "...[truncated]"
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": output}
            )

    # ---------- 비스트리밍 ----------

    async def run(
        self,
        user_message: str,
        history: list[dict] | None = None,
    ) -> AgentResult:
        """입력: user_message, history(선택) / 출력: AgentResult.

        동작: ReAct 루프 본체.
        - 매 반복: client.complete(messages, tools) 호출 → assistant 메시지 append
        - tool_calls가 있으면 → _run_tool_calls로 실행+tool 메시지 append → 반복
        - 텍스트만 오면 → content 확정 후 반환
        - agent_max_iterations 초과 시 → tools=None으로 최종 답변을 1회 더 요청
          (모델이 더 이상 도구를 호출하지 못하게 해 반드시 텍스트로 끝나게 함)
        """
        messages = self._base_messages(user_message, history)
        result = AgentResult(content="", iterations=0)
        used: list[str] = []

        while result.iterations < self.settings.agent_max_iterations:
            turn = await self.client.complete(messages, tools=self._tools)
            self._accumulate(result, turn)
            messages.append(self._assistant_message(turn))

            if not turn.tool_calls:
                result.content = turn.content or ""
                result.iterations += 1
                result.tool_names_used = used
                self._finish_cost(result)
                return result

            await self._run_tool_calls(messages, turn, used)
            result.iterations += 1

        # 반복 한도 초과 → 툴 없이 최종 답변을 한 번 더 요청
        logger.warning("max iterations reached; forcing final answer")
        turn = await self.client.complete(messages, tools=None)
        self._accumulate(result, turn)
        messages.append(self._assistant_message(turn))
        result.content = turn.content or "(도구 호출 한도에 도달했습니다)"
        result.tool_names_used = used
        self._finish_cost(result)
        return result

    # ---------- 스트리밍 ----------

    async def run_stream(
        self,
        user_message: str,
        history: list[dict] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """입력: run()과 동일 / 출력: 이벤트 dict의 async iterator.

        동작: client.stream의 조각 이벤트를 외부로 흘려보내며 루프를 돌린다.
        이벤트 형식:
          {"type": "text", "delta": "..."}                    — 텍스트 조각
          {"type": "tool_call_delta", ...}                    — 툴콜 스트림 조각(UI 진행 표시용)
          {"type": "tool_call", "name":..., "arguments":...}  — 툴 실행 직전
          {"type": "tool_result", "name":..., "ok": bool}      — 툴 실행 직후
          {"type": "done", "result": AgentResult}              — 최종 결과(항상 마지막)

        비스트리밍과의 차이: 텍스트는 TurnComplete까지 모았다가
        text_parts로 content를 확정한다(조각 누락 방지).
        """
        messages = self._base_messages(user_message, history)
        result = AgentResult(content="", iterations=0)
        used: list[str] = []

        while result.iterations < self.settings.agent_max_iterations:
            turn: LLMTurn | None = None
            text_parts: list[str] = []

            async for event in self.client.stream(messages, tools=self._tools):
                if isinstance(event, TextDelta):
                    text_parts.append(event.delta)
                    yield {"type": "text", "delta": event.delta}
                elif isinstance(event, ToolCallDelta):
                    # 스트림 조각은 UI 진행 표시용으로만 흘린다 (조립은 TurnComplete에서)
                    yield {
                        "type": "tool_call_delta",
                        "index": event.index,
                        "name": event.name,
                        "arguments_delta": event.arguments_delta,
                    }
                elif isinstance(event, TurnComplete):
                    turn = event.turn

            assert turn is not None  # stream은 반드시 TurnComplete로 끝난다
            self._accumulate(result, turn)
            messages.append(self._assistant_message(turn))
            result.iterations += 1

            if not turn.tool_calls:
                result.content = "".join(text_parts) or turn.content or ""
                result.tool_names_used = used
                self._finish_cost(result)
                yield {"type": "done", "result": result}
                return

            for call in turn.tool_calls:
                used.append(call.name)
                yield {"type": "tool_call", "name": call.name, "arguments": call.arguments}
                output = await execute_tool(call.name, call.arguments)
                ok = not output.lstrip().startswith('{"error"')
                yield {"type": "tool_result", "name": call.name, "ok": ok}
                if len(output) > MAX_TOOL_RESULT_CHARS:
                    output = output[:MAX_TOOL_RESULT_CHARS] + "...[truncated]"
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": output}
                )

        # 반복 한도 → 비스트리밍 최종 호출
        turn = await self.client.complete(messages, tools=None)
        self._accumulate(result, turn)
        result.content = turn.content or "(도구 호출 한도에 도달했습니다)"
        result.tool_names_used = used
        self._finish_cost(result)
        yield {"type": "done", "result": result}

    async def aclose(self) -> None:
        """입력: 없음 / 출력: 없음.

        동작: client에 aclose가 있으면 위임한다(OpenAIChatClient는
        내부 AsyncOpenAI 세션을 닫음). FakeClient처럼 aclose가 없는
        구현체에서도 안전하게 no-op이 되도록 getattr으로 확인한다.
        """
        close = getattr(self.client, "aclose", None)
        if close:
            await close()
