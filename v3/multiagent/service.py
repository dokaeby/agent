"""FastAPI 서비스 — 멀티에이전트 리포트 파이프라인 HTTP 인터페이스."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .orchestrator import Orchestrator, PipelineResult


class ReportRequest(BaseModel):
    """POST /reports 요청 바디.

    입력: topic(리서치 주제, 필수), max_subtasks(선택)
    출력: 검증된 요청 객체
    """

    topic: str = Field(min_length=2, max_length=500)
    max_subtasks: int = Field(default=5, ge=1, le=10)


def _result_to_dict(r: PipelineResult) -> dict[str, Any]:
    """입력: PipelineResult / 출력: JSON 직렬화 가능한 dict.

    동작: dataclass를 dict로 변환해 FastAPI 응답으로 반환.
    """
    return {
        "topic": r.topic,
        "plan": r.plan,
        "research_notes": r.research_notes,
        "draft": r.draft,
        "review_score": r.review_score,
        "review_rounds": r.review_rounds,
        "final_report": r.final_report,
        "elapsed_sec": r.elapsed_sec,
        "budget": r.budget,
        "tool_calls": r.tool_calls,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    """입력: settings / 출력: FastAPI 앱.

    동작: 앱 팩토리. 오케스트레이터는 요청마다 생성한다 —
    MCP 클라이언트/예산 추적이 요청 단위로 독립적이어야 하기 때문.
    """
    config = settings or get_settings()
    app = FastAPI(title="Multi-Agent Report Service", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """입력: 없음 / 출력: 상태 dict."""
        return {"status": "ok", "llm_enabled": str(config.llm_enabled)}

    @app.post("/reports")
    async def create_report(body: ReportRequest) -> dict[str, Any]:
        """입력: ReportRequest / 출력: PipelineResult dict.

        동작: Orchestrator를 생성해 run(topic)을 실행한다.
        각 요청이 독립적인 budget/MCP 세션을 갖는다.
        """
        orch = Orchestrator(config)
        result = await orch.run(body.topic)
        return _result_to_dict(result)

    return app


app = create_app()
