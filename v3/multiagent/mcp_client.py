"""MCP 클라이언트 — stdio로 mcp_server를 띄워 tool을 호출한다.

에이전트의 tools.py가 이 클라이언트를 통해 MCP 도구를 사용한다.
서버가 죽어 있으면 호출이 실패하고 execute_tool이 에러 JSON으로 변환한다.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MCPClient:
    """FastMCP stdio 서버와의 세션 클라이언트.

    입력: server_module(실행할 모듈 경로)
    출력: call_tool()/list_tools()
    동작: lazy connect — 첫 호출 시 stdio_client로 서버 프로세스를 띄운다.
    """

    def __init__(self, server_module: str = "multiagent.mcp_server") -> None:
        """입력: 서버 모듈 경로 / 출력: 인스턴스(미연결 상태)."""
        self._server_module = server_module
        self._session = None
        self._exit_stack = None

    async def _ensure_session(self):
        """입력: 없음 / 출력: ClientSession.

        동작: 최초 호출 시 stdio_client로 서버를 spawn하고 세션을 연다.
        """
        if self._session is not None:
            return self._session
        import sys
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", self._server_module],
        )
        self._exit_stack = AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self._session

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """입력: MCP tool 이름, 인자 dict / 출력: 결과 텍스트.

        동작: tools/call 표준 메서드로 서버에 요청.
        content[0].text를 반환. 실패 시 예외 전파(호출자가 처리).
        """
        session = await self._ensure_session()
        result = await session.call_tool(name, arguments)
        # MCP CallToolResult: content 리스트에서 text 블록 추출
        parts = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else ""

    async def aclose(self) -> None:
        """입력: 없음 / 출력: 없음. 세션과 서버 프로세스를 정리한다."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
