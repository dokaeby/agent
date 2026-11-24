"""도구 레지스트리 + 권한 테스트 — API 키 불필요."""

import json

import pytest

from multiagent import tools as t
from multiagent.tools import execute_tool, register_tool, tool_schemas


class FakeMCP:
    """MCPClient 흉내 — call_tool만 구현.

    입력: responses(tool_name → 반환 문자열)
    출력: call_tool() 메서드
    동작: 등록된 응답을 반환하거나, 없으면 기본값.
    """

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    async def call_tool(self, name, arguments):
        """입력: tool 이름, 인자 / 출력: 등록된 응답 또는 기본 문자열."""
        self.calls.append((name, arguments))
        return self.responses.get(name, f"fake:{name}")


@pytest.fixture
def fake_mcp():
    """입력: 없음 / 출력: FakeMCP.

    동작: 테스트용 가짜 MCP 클라이언트.
    """
    return FakeMCP({"search_docs": "검색 결과입니다", "save_report": "saved: test.md"})


def test_register_and_schema():
    """입력: 커스텀 도구 등록 / 출력: 없음(단언).

    동작: register_tool 후 tool_schemas에 OpenAI 형식으로 나타나는지 확인.
    """
    register_tool(
        name="echo",
        description="Echo input",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        permission="read",
        func=lambda text: text,
    )
    schemas = tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "echo" in names


async def test_execute_read_tool(fake_mcp):
    """입력: read 도구 호출 / 출력: JSON 결과.

    동작: allow_write=False여도 read 도구는 실행되고,
    결과가 wrap_untrusted 태그로 감싸져 반환되는지 확인.
    """
    register_tool(
        name="test_read",
        description="test",
        parameters={"type": "object", "properties": {}},
        permission="read",
        func=lambda: "read result",
    )
    out = json.loads(await execute_tool("test_read", "{}", allow_write=False, mcp=fake_mcp))
    assert "read result" in out["result"]
    assert "untrusted" in out["result"]  # 인젝션 방어 태그


async def test_execute_write_tool_blocked(fake_mcp):
    """입력: write 도구 + allow_write=False / 출력: permission denied.

    동작: write 권한 도구가 차단되고 func가 호출되지 않는지 확인.
    """
    called = False

    def write_func():
        nonlocal called
        called = True
        return "should not happen"

    register_tool(
        name="test_write",
        description="test",
        parameters={"type": "object", "properties": {}},
        permission="write",
        func=write_func,
    )
    out = json.loads(await execute_tool("test_write", "{}", allow_write=False, mcp=fake_mcp))
    assert "permission denied" in out["error"]
    assert called is False  # func 미실행


async def test_execute_write_tool_allowed(fake_mcp):
    """입력: write 도구 + allow_write=True / 출력: 정상 결과."""
    register_tool(
        name="test_write2",
        description="test",
        parameters={"type": "object", "properties": {}},
        permission="write",
        func=lambda: "write ok",
    )
    out = json.loads(await execute_tool("test_write2", "{}", allow_write=True, mcp=fake_mcp))
    assert "write ok" in out["result"]


async def test_unknown_tool():
    """입력: 미등록 도구명 / 출력: unknown tool 에러."""
    out = json.loads(await execute_tool("no_such", "{}", allow_write=True))
    assert "unknown tool" in out["error"]


async def test_invalid_json_args():
    """입력: 깨진 JSON / 출력: invalid JSON 에러."""
    out = json.loads(await execute_tool("echo", "{bad", allow_write=True))
    assert "invalid JSON" in out["error"]


async def test_tool_exception_returns_error_json(fake_mcp):
    """입력: 예외를 던지는 도구 / 출력: error JSON(예외 전파 아님)."""
    def boom():
        raise RuntimeError("tool crashed")

    register_tool(
        name="boom_tool",
        description="test",
        parameters={"type": "object", "properties": {}},
        permission="read",
        func=boom,
    )
    out = json.loads(await execute_tool("boom_tool", "{}", allow_write=True, mcp=fake_mcp))
    assert "RuntimeError" in out["error"]
    assert "tool crashed" in out["error"]


def test_mcp_tools_registered():
    """입력: register_mcp_tools() / 출력: 없음(단언).

    동작: MCP 도구 4개가 레지스트리에 등록되고 권한이 올바른지 확인.
    """
    t.register_mcp_tools()
    tools = {tool.name: tool for tool in t.all_tools()}
    assert "search_docs" in tools
    assert "save_report" in tools
    assert tools["search_docs"].permission == "read"
    assert tools["save_report"].permission == "write"
