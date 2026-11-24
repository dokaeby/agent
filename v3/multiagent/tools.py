"""도구 레지스트리 — MCP 도구를 에이전트용으로 래핑 + 권한 분리.

권한 모델:
- read: 누구나 호출 가능 (search_docs, get_current_time, list_reports)
- write: ALLOW_WRITE_TOOLS=true일 때만 (save_report)

execute_tool은 권한 검사 → MCP 호출 → 결과를 wrap_untrusted로 정제해 반환.
모든 실패는 에러 JSON — 모델이 보고 재시도 가능.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .guardrails import READ, WRITE, check_permission, wrap_untrusted
from .mcp_client import MCPClient

logger = logging.getLogger(__name__)


@dataclass
class AgentTool:
    """에이전트가 사용할 수 있는 도구 정의.

    입력: name/description/parameters(JSON Schema)/permission/func
    출력: Tool 객체
    동작: func는 동기/async 모두 가능. MCP 도구는 mcp_client.call_tool을 감싼다.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    permission: str  # READ | WRITE
    func: Callable[..., Any]


_REGISTRY: dict[str, AgentTool] = {}


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    permission: str,
    func: Callable[..., Any],
) -> None:
    """입력: 도구 정의 / 출력: 없음.

    동작: _REGISTRY에 등록. 데코레이터 대신 명시적 함수로
    MCP 클라이언트 주입이 필요한 도구를 깔끔하게 등록한다.
    """
    _REGISTRY[name] = AgentTool(name, description, parameters, permission, func)


def all_tools() -> list[AgentTool]:
    """입력: 없음 / 출력: 등록된 도구 리스트."""
    return list(_REGISTRY.values())


def tool_schemas() -> list[dict[str, Any]]:
    """입력: 없음 / 출력: OpenAI tools 파라미터 형식.

    동작: AgentTool → {"type":"function","function":{...}} 변환.
    """
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in _REGISTRY.values()
    ]


async def execute_tool(
    name: str, arguments_json: str, *, allow_write: bool, mcp: MCPClient | None = None
) -> str:
    """입력: 도구명, JSON 인자, write 허용 여부, MCP 클라이언트 / 출력: JSON 문자열.

    동작:
    1. _REGISTRY 조회 — 없으면 unknown tool 에러.
    2. check_permission — write 도구 + allow_write=False면 permission denied.
    3. func(**args) 실행 — awaitable이면 await.
    4. 결과를 wrap_untrusted로 정제(인젝션 방어) 후 {"result": ...} JSON 반환.
    모든 예외는 {"error": ...}로 변환 — 루프를 죽이지 않는다.
    """
    tool = _REGISTRY.get(name)
    if tool is None:
        return json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)

    perm_error = check_permission(tool.permission, allow_write)
    if perm_error:
        return json.dumps({"error": perm_error}, ensure_ascii=False)

    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"invalid JSON arguments: {exc}"}, ensure_ascii=False)
    if not isinstance(args, dict):
        return json.dumps({"error": "arguments must be a JSON object"}, ensure_ascii=False)

    try:
        result = tool.func(**args)
        if inspect.isawaitable(result):
            result = await result
        # 도구 결과는 신뢰할 수 없는 데이터 — 인젝션 정제 후 반환
        safe_result = wrap_untrusted(str(result), source=f"tool:{name}")
        return json.dumps({"result": safe_result}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 - 도구 오류는 모델에 전달
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


# --- MCP 도구 래퍼 등록 ---

# MCP 클라이언트는 모듈 레벨 싱글턴 — orchestrator가 생성해 주입
_mcp_client: MCPClient | None = None


def set_mcp_client(client: MCPClient) -> None:
    """입력: MCPClient / 출력: 없음.

    동작: 도구 래퍼가 사용할 MCP 클라이언트를 설정한다.
    오케스트레이터 초기화 시 1회 호출.
    """
    global _mcp_client
    _mcp_client = client


def _require_mcp() -> MCPClient:
    """입력: 없음 / 출력: MCPClient.

    동작: _mcp_client가 없으면 RuntimeError — set_mcp_client 먼저 호출 필요.
    """
    if _mcp_client is None:
        raise RuntimeError("MCP client not initialized — call set_mcp_client() first")
    return _mcp_client


def register_mcp_tools() -> None:
    """입력: 없음 / 출력: 없음.

    동작: MCP 서버의 4개 도구를 AgentTool로 등록한다.
    각 func는 _mcp_client.call_tool을 호출하는 얇은 래퍼.
    """

    async def _call(name: str, **kwargs: Any) -> str:
        """입력: MCP tool 이름, 인자 / 출력: 결과 텍스트.

        동작: MCP 클라이언트로 tools/call 요청.
        """
        return await _require_mcp().call_tool(name, kwargs)

    register_tool(
        name="search_docs",
        description=(
            "Search the knowledge base for documents related to a query. "
            "Returns matching doc excerpts with scores."
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
        permission=READ,
        func=lambda query: _call("search_docs", query=query),
    )

    register_tool(
        name="get_current_time",
        description="Get the current date and time.",
        parameters={"type": "object", "properties": {}},
        permission=READ,
        func=lambda: _call("get_current_time"),
    )

    register_tool(
        name="save_report",
        description="Save a report to a file. Requires write permission.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Report title (used as filename)"},
                "content": {"type": "string", "description": "Markdown content"},
            },
            "required": ["title", "content"],
        },
        permission=WRITE,
        func=lambda title, content: _call("save_report", title=title, content=content),
    )

    register_tool(
        name="list_reports",
        description="List previously saved reports.",
        parameters={"type": "object", "properties": {}},
        permission=READ,
        func=lambda: _call("list_reports"),
    )
