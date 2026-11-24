"""FastAPI 서버: 채팅 엔드포인트 + 도구 목록 + 헬스체크."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import Agent
from .config import Settings, get_settings
from .llm import ChatClient
from .tools import all_tools


class ChatRequest(BaseModel):
    """POST /chat의 요청 바디.

    입력: message(필수, 1~50000자), history(선택, 이전 대화 dict 목록)
    출력: 검증된 요청 객체 — 길이 위반은 FastAPI가 자동으로 422 반환
    동작: history는 클라이언트가 세션을 유지할 때 이전 턴을 그대로 보낸다.
    서버는 무상태(stateless)라 대화 저장소가 없다.
    """

    message: str = Field(min_length=1, max_length=50_000)
    history: list[dict[str, Any]] | None = None  # [{"role": "user"|"assistant", "content": ...}]


class ChatResponse(BaseModel):
    """POST /chat의 응답 바디.

    입력: AgentResult의 필드들
    출력: reply + 반복 횟수 + 토큰 사용량 + 비용 + 사용된 도구명
    동작: response_model로 선언해 OpenAPI 스키마와 응답 검증을 자동화한다.
    """

    reply: str
    iterations: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    tools_used: list[str]


def create_app(client: ChatClient | None = None, settings: Settings | None = None) -> FastAPI:
    """입력: client(테스트용 가짜 주입 가능), settings(기본 get_settings()).
    출력: 라우트가 등록된 FastAPI 앱.

    동작: 앱 팩토리 패턴 — Agent를 app.state.agent에 1회 생성해
    모든 요청이 같은 인스턴스(=같은 _MEMORY 등 상태)를 공유한다.
    client를 주입하면 OpenAIChatClient 대신 Fake로 교체해
    API 키 없이 엔드포인트를 테스트할 수 있다.
    """
    config = settings or get_settings()
    app = FastAPI(title="Frame-free Agent", version="0.1.0")
    app.state.agent = Agent(client=client, settings=config)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """입력: 없음 / 출력: {"status":"ok","llm_enabled":"true|false"}.

        동작: 서버 생존 + LLM 키 설정 여부를 한 번에 확인하는 엔드포인트.
        llm_enabled가 false여도 서버는 정상 — /chat만 503을 반환한다.
        """
        return {"status": "ok", "llm_enabled": str(config.llm_enabled)}

    @app.get("/tools")
    async def list_tools() -> dict[str, Any]:
        """입력: 없음 / 출력: {"tools":[{name,description,parameters}...]}.

        동작: _REGISTRY에 등록된 도구 스키마를 그대로 노출한다.
        모델에 보내는 것과 동일한 JSON Schema라 디버깅/문서화에 유용.
        """
        return {
            "tools": [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in all_tools()
            ]
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest) -> ChatResponse:
        """입력: ChatRequest(message, history?) / 출력: ChatResponse.

        동작: app.state.agent.run()을 await해 ReAct 루프를 끝까지 돌린다.
        RuntimeError(API 키 없음 등)는 503으로 변환한다 —
        클라이언트가 "서버 다운"과 "설정 문제"를 구분할 수 있게.
        그 외 예외는 FastAPI 기본 500 처리에 맡긴다.
        """
        agent: Agent = app.state.agent
        try:
            result = await agent.run(body.message, history=body.history)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return ChatResponse(
            reply=result.content,
            iterations=result.iterations,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            tools_used=result.tool_names_used,
        )

    return app


app = create_app()
"""uvicorn이 import하는 앱 인스턴스. 모듈 레벨 생성이 가능했던 것은
OpenAIChatClient가 lazy 초기화라서 — API 키 없이도 import된다."""
