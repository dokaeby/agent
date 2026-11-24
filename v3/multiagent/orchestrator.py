"""슈퍼바이저 오케스트레이터 — 멀티에이전트 패턴의 본체.

구현된 패턴:
- Supervisor: 이 클래스가 전체 흐름을 관리하고 각 에이전트에 위임
- Plan-and-Execute: planner가 하위 작업을 분해 → 실행 → 종합
- 병렬 서브에이전트: researcher들을 asyncio.gather로 동시 실행
- Reviewer/Critic: reviewer가 채점 → 미달 시 writer에게 피드백(handoff)
- Handoff: reviewer→writer 수정 요청, 최종 publisher에게 전달

운영 가드레일:
- BudgetTracker: LLMClient가 모든 호출을 추적, 상한 시 BudgetExceeded
- Timeout: 각 에이전트 호출에 asyncio.wait_for 적용
- Retry: LLMClient.chat 내부에서 with_retry로 재시도
- Permission: context.allow_write로 write 도구 차단
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .agents import (
    PLANNER_PROMPT,
    PUBLISHER_PROMPT,
    RESEARCHER_PROMPT,
    REVIEWER_PROMPT,
    WRITER_PROMPT,
    AgentContext,
    AgentOutput,
    BaseAgent,
)
from .config import Settings, get_settings
from .guardrails import BudgetExceeded, BudgetTracker, with_timeout
from .llm import LLMClient
from .mcp_client import MCPClient
from .tools import register_mcp_tools, set_mcp_client

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """파이프라인 실행 결과.

    입력: 각 단계의 산출물
    출력: 최종 리포트 + 메타데이터
    동작: budget_summary로 비용/토큰 사용량을 포함한다.
    """

    topic: str
    plan: list[str]
    research_notes: list[str]
    draft: str
    review_score: float
    review_rounds: int
    final_report: str
    elapsed_sec: float
    budget: dict[str, Any]
    tool_calls: dict[str, list[str]] = field(default_factory=dict)


class Orchestrator:
    """리서치 리포트 파이프라인 슈퍼바이저.

    입력: Settings(없으면 싱글턴)
    출력: run(topic) → PipelineResult
    동작: planner→researcher(병렬)→writer→reviewer 루프→publisher 순서로 실행.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """입력: settings / 출력: 인스턴스.

        동작: LLMClient + BudgetTracker + MCPClient + 에이전트들을 생성한다.
        MCP 서버는 stdio로 자동 spawn된다.
        """
        self.settings = settings or get_settings()
        self.budget = BudgetTracker(
            max_cost_usd=self.settings.max_cost_usd,
            max_total_tokens=self.settings.max_total_tokens,
            model=self.settings.agent_model,
        )
        self.llm = LLMClient(self.settings, self.budget)
        self.mcp = MCPClient()
        set_mcp_client(self.mcp)
        register_mcp_tools()

        self.context = AgentContext(allow_write=self.settings.allow_write_tools, mcp=self.mcp)
        self._agents = {
            "planner": BaseAgent("planner", PLANNER_PROMPT, [], self.llm, self.context),
            "researcher": BaseAgent(
                "researcher", RESEARCHER_PROMPT, ["search_docs", "get_current_time"], self.llm, self.context
            ),
            "writer": BaseAgent("writer", WRITER_PROMPT, [], self.llm, self.context),
            "reviewer": BaseAgent("reviewer", REVIEWER_PROMPT, [], self.llm, self.context),
            "publisher": BaseAgent("publisher", PUBLISHER_PROMPT, ["save_report"], self.llm, self.context),
        }

    async def _run_agent(self, name: str, task: str, extra_context: str = "") -> tuple[AgentOutput, float]:
        """입력: 에이전트명, 작업, 추가 컨텍스트 / 출력: (AgentOutput, 소요시간).

        동작: with_timeout으로 에이전트 호출을 제한하고 시간을 측정한다.
        """
        agent = self._agents[name]
        t0 = time.monotonic()
        output = await with_timeout(
            agent.run(task, extra_context=extra_context),
            self.settings.agent_timeout_sec,
            label=f"agent:{name}",
        )
        elapsed = time.monotonic() - t0
        logger.info("%s 완료 (%.1fs, %d tools)", name, elapsed, len(output.tool_calls_made))
        return output, elapsed

    async def _plan(self, topic: str) -> list[str]:
        """입력: 주제 / 출력: 하위 질문 리스트.

        동작: planner가 JSON 배열을 반환. 파싱 실패 시 주제를 그대로 사용.
        """
        out, _ = await self._run_agent("planner", f"리서치 주제: {topic}")
        try:
            # planner는 JSON 배열을 출력해야 함 — 텍스트에서 파싱
            text = out.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            plan = json.loads(text)
            if isinstance(plan, list) and all(isinstance(x, str) for x in plan):
                return plan[:5]  # 최대 5개 하위 작업
        except (json.JSONDecodeError, ValueError):
            logger.warning("planner 출력 파싱 실패, 원문 사용: %.100s", out.content)
        return [topic]

    async def _research_parallel(self, subtasks: list[str]) -> list[str]:
        """입력: 하위 질문 리스트 / 출력: 각 질문의 조사 노트.

        동작: 각 subtask마다 researcher를 생성해 asyncio.gather로 병렬 실행.
        이게 '병렬 서브에이전트' 패턴 — sequential 대비 wall-clock 시간 단축.
        """
        async def _one(subtask: str) -> tuple[str, AgentOutput]:
            """입력: 하위 질문 / 출력: (질문, 결과)."""
            out, _ = await self._run_agent("researcher", f"조사 질문: {subtask}")
            return subtask, out

        results = await asyncio.gather(*[_one(st) for st in subtasks])
        notes = []
        for subtask, out in results:
            notes.append(f"## {subtask}\n{out.content}")
        return notes

    async def _write_draft(self, topic: str, notes: list[str], feedback: str = "") -> tuple[str, AgentOutput]:
        """입력: 주제, 조사 노트, 피드백 / 출력: (초안, AgentOutput).

        동작: writer에게 노트를 컨텍스트로 전달해 초안 작성.
        피드백이 있으면 reviewer의 수정 요청을 포함한다(handoff).
        """
        context = "\n\n".join(notes)
        task = f"주제: {topic}\n\n위 조사 노트들을 바탕으로 리포트 초안을 작성하라."
        if feedback:
            task += f"\n\n이전 초안에 대한 리뷰어 피드백:\n{feedback}\n이를 반영해 수정하라."
        out, _ = await self._run_agent("writer", task, extra_context=context)
        return out.content, out

    async def _review(self, topic: str, draft: str) -> tuple[float, str, AgentOutput]:
        """입력: 주제, 초안 / 출력: (점수, 피드백, AgentOutput).

        동작: reviewer가 JSON {score, feedback, passed}를 반환.
        파싱 실패 시 점수 0.5로 간주(보수적).
        """
        task = f"주제: {topic}\n\n다음 초안을 평가하라:\n\n{draft[:3000]}"
        out, _ = await self._run_agent("reviewer", task)
        try:
            text = out.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            return float(data.get("score", 0.5)), str(data.get("feedback", "")), out
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("reviewer 출력 파싱 실패: %.100s", out.content)
            return 0.5, "파싱 실패", out

    async def _publish(self, topic: str, draft: str, notes: list[str]) -> tuple[str, AgentOutput]:
        """입력: 주제, 최종 초안, 노트 / 출력: (최종 리포트, AgentOutput).

        동작: publisher가 포맷팅. save_report 도구가 허용되면 파일로도 저장.
        """
        context = "\n\n".join(notes[:3])  # 출처 요약용으로 앞 3개만
        task = f"주제: {topic}\n\n최종 초안을 출판용으로 포맷팅하라."
        out, _ = await self._run_agent("publisher", task, extra_context=context)
        return out.content, out

    async def run(self, topic: str) -> PipelineResult:
        """입력: 리서치 주제 / 출력: PipelineResult.

        동작: 전체 파이프라인 실행.
        1. plan: planner가 하위 질문 분해
        2. research: researcher들을 병렬 실행
        3. write+review: writer 초안 → reviewer 채점 → 미달 시 피드백과 함께 재작성 (최대 MAX_REVIEW_ROUNDS)
        4. publish: publisher가 최종 포맷
        5. budget.summary()로 비용/토큰 반환
        """
        t0 = time.monotonic()
        tool_calls: dict[str, list[str]] = {}

        try:
            # 1. Plan-and-Execute: 계획 수립
            plan = await self._plan(topic)
            logger.info("계획: %d개 하위 작업", len(plan))

            # 2. 병렬 서브에이전트: 조사
            notes = await self._research_parallel(plan)
            logger.info("조사 완료: %d개 노트", len(notes))

            # 3. Writer + Reviewer 루프 (handoff: reviewer→writer)
            draft, writer_out = await self._write_draft(topic, notes)
            tool_calls["writer"] = writer_out.tool_calls_made
            review_score, feedback, rounds = 0.0, "", 0

            for round_num in range(self.settings.max_review_rounds + 1):
                review_score, feedback, reviewer_out = await self._review(topic, draft)
                tool_calls["reviewer"] = reviewer_out.tool_calls_made
                rounds = round_num + 1
                logger.info("리뷰 %d회차: score=%.2f", rounds, review_score)
                passed = review_score >= self.settings.review_pass_score
                if passed or round_num >= self.settings.max_review_rounds:
                    break
                # handoff: reviewer의 피드백을 writer에게 전달해 수정
                draft, writer_out = await self._write_draft(topic, notes, feedback=feedback)
                tool_calls["writer"] = writer_out.tool_calls_made

            # 4. Publish
            final_report, pub_out = await self._publish(topic, draft, notes)
            tool_calls["publisher"] = pub_out.tool_calls_made

            elapsed = time.monotonic() - t0
            return PipelineResult(
                topic=topic,
                plan=plan,
                research_notes=notes,
                draft=draft,
                review_score=review_score,
                review_rounds=rounds,
                final_report=final_report,
                elapsed_sec=round(elapsed, 2),
                budget=self.budget.summary(),
                tool_calls=tool_calls,
            )
        except BudgetExceeded as exc:
            elapsed = time.monotonic() - t0
            logger.error("예산 초과로 파이프라인 중단: %s", exc)
            return PipelineResult(
                topic=topic, plan=[], research_notes=[], draft="",
                review_score=0.0, review_rounds=0, final_report=f"[중단] {exc}",
                elapsed_sec=round(elapsed, 2), budget=self.budget.summary(),
            )
        finally:
            await self.mcp.aclose()
            await self.llm.aclose()
