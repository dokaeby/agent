"""에이전트 역할 정의 — Planner/Researcher/Writer/Reviewer.

각 에이전트는 '역할 + 사용 가능 도구 + 시스템 프롬프트'만 정의한다.
실제 호출/타임아웃/예산은 orchestrator가 관리한다 —
에이전트는 '무엇을 하는가'만 알고 '언제 멈추는가'는 모른다(관심사 분리).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .guardrails import wrap_untrusted
from .llm import LLMClient
from .tools import execute_tool, tool_schemas

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3  # 에이전트당 도구 호출 루프 상한 — 무한 루프 방지


@dataclass
class AgentContext:
    """에이전트 실행 컨텍스트 — 공유 상태.

    입력: orchestrator가 생성
    출력: 각 에이전트가 참조
    동작: allow_write로 권한을, budget은 llm.budget으로 이미 연동됨.
    """

    allow_write: bool = False
    mcp: Any = None  # MCPClient (tools.execute_tool에 전달)


@dataclass
class AgentOutput:
    """에이전트 실행 결과.

    입력: content(텍스트/JSON), tool_calls_made(기록용)
    출력: 호출자에게 반환
    동작: researcher는 JSON 배열, writer/reviewer는 텍스트를 content에 담는다.
    """

    content: str
    tool_calls_made: list[str] = field(default_factory=list)
    iterations: int = 0


class BaseAgent:
    """에이전트 베이스 — 시스템 프롬프트 + 도구 + ReAct 루프.

    입력: name, system_prompt, allowed_tool_names, llm, context
    출력: run() → AgentOutput
    동작: 도구 호출 루프를 MAX_TOOL_ROUNDS로 제한.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        allowed_tool_names: list[str],
        llm: LLMClient,
        context: AgentContext,
    ) -> None:
        """입력: 역할 정의 / 출력: 인스턴스.

        동작: allowed_tool_names에 해당하는 스키마만 LLM에 노출한다 —
        역할별 권한 분리의 일부(연구원은 write 도구를 '보지도' 못한다).
        """
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.context = context
        all_schemas = {s["function"]["name"]: s for s in tool_schemas()}
        self._tools = [all_schemas[n] for n in allowed_tool_names if n in all_schemas]
        self._tool_names = set(allowed_tool_names)

    async def _call_llm(self, messages: list[dict[str, Any]], use_tools: bool = True) -> dict[str, Any]:
        """입력: messages / 출력: {"content": str, "tool_calls": [...]}.

        동작: chat()은 텍스트만 반환하므로, 도구 호출이 필요한 경우
        JSON 모드로 {"tool": name, "args": {...}} 또는 {"final": "..."} 형식을 받는다.
        실제 OpenAI function calling 대신 'JSON 프로토콜'로 단순화 —
        프레임워크 없이도 도구 루프가 동작함을 보이기 위함.
        """
        if use_tools and self._tools:
            tool_desc = "\n".join(
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in self._tools
            )
            prompt = (
                messages[-1]["content"]
                + "\n\n사용 가능한 도구:\n" + tool_desc +
                "\n\n도구를 쓰려면 {\"tool\": \"이름\", \"args\": {...}} JSON만 출력하라."
                " 최종 답변이면 {\"final\": \"답변\"} JSON만 출력하라."
            )
            messages[-1] = {"role": "user", "content": prompt}
            try:
                data = await self.llm.chat_json(messages, temperature=0.0, max_tokens=800)
                return {"content": "", "tool_calls": [], "json": data}
            except Exception:
                # JSON 파싱 실패 → 일반 텍스트로 처리
                text = await self.llm.chat(messages, temperature=0.0, max_tokens=800)
                return {"content": text, "tool_calls": [], "json": None}
        text = await self.llm.chat(messages)
        return {"content": text, "tool_calls": [], "json": None}

    async def _execute_tool_call(self, name: str, args: dict[str, Any], used: list[str]) -> str:
        """입력: 도구명, 인자, 기록 리스트 / 출력: 도구 결과 문자열.

        동작: 화이트리스트(_tool_names) 확인 → execute_tool 위임.
        결과는 이미 wrap_untrusted로 정제되어 있다.
        """
        if name not in self._tool_names:
            return json.dumps({"error": f"tool not allowed for {self.name}: {name}"})
        used.append(name)
        return await execute_tool(
            name, json.dumps(args, ensure_ascii=False),
            allow_write=self.context.allow_write, mcp=self.context.mcp,
        )

    async def run(self, task: str, extra_context: str = "") -> AgentOutput:
        """입력: task(역할별 작업 지시), extra_context(선택) / 출력: AgentOutput.

        동작: 시스템 프롬프트 + task로 ReAct 루프 실행.
        JSON 프로토콜로 도구 호출을 해석하고, MAX_TOOL_ROUNDS까지 반복.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if extra_context:
            ctx_msg = f"참고 자료:\n{wrap_untrusted(extra_context, 'context')}"
            messages.append({"role": "system", "content": ctx_msg})
        messages.append({"role": "user", "content": task})

        used: list[str] = []
        iterations = 0
        final_content = ""

        while iterations < MAX_TOOL_ROUNDS:
            iterations += 1
            resp = await self._call_llm(messages)
            data = resp.get("json")

            if data and isinstance(data, dict):
                if "final" in data:
                    final_content = str(data["final"])
                    break
                if "tool" in data:
                    tool_name = str(data["tool"])
                    tool_args = data.get("args", {})
                    if not isinstance(tool_args, dict):
                        tool_args = {}
                    result = await self._execute_tool_call(tool_name, tool_args, used)
                    messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
                    messages.append({"role": "user", "content": f"도구 결과: {result}"})
                    continue
                # 예상치 못한 JSON → 내용으로 취급
                final_content = json.dumps(data, ensure_ascii=False)
                break

            final_content = resp.get("content", "")
            break

        return AgentOutput(content=final_content, tool_calls_made=used, iterations=iterations)


# ---------- 역할별 시스템 프롬프트 ----------

PLANNER_PROMPT = """You are a research planner. Break the topic into 3-5 specific,
searchable sub-questions. Output ONLY a JSON array of strings,
e.g. ["sub-question 1", "sub-question 2"].
Each sub-question should be answerable by searching documents. No explanations."""

RESEARCHER_PROMPT = """You are a research analyst. Answer the sub-question using search_docs.
Rules:
- Search first, then answer based ONLY on what the tool returns.
- If the tool returns no useful info, say "정보 없음".
- Be concise: 2-4 sentences with key facts.
- Output format: {"final": "your answer"} or {"tool": "search_docs", "args": {"query": "..."}}"""

WRITER_PROMPT = """You are a technical writer. Write a well-structured report section in Markdown.
Rules:
- Use ONLY the provided research notes as source material.
- Structure: summary paragraph, then detailed sections with headers.
- Cite sources inline as [source: filename] when possible.
- Do NOT invent facts not in the notes.
- Output the markdown directly (not JSON)."""

REVIEWER_PROMPT = """You are a critical reviewer. Evaluate the draft report against these criteria:
1. Accuracy — does it match the research notes?
2. Completeness — does it cover all sub-questions?
3. Clarity — is it well-structured and readable?

Output ONLY JSON: {"score": 0.0-1.0, "feedback": "specific improvement suggestions", "passed": true/false}
Score >= 0.7 means passed. Be strict but fair."""

PUBLISHER_PROMPT = """You are a publisher. Format the final report for delivery.
Rules:
- Add a title header, table of contents if long, and a sources section.
- Ensure markdown is clean and well-formatted.
- Output the final markdown only."""
