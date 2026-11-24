"""오케스트레이터 + 에이전트 테스트 — FakeLLM으로 API 키 불필요."""

import pytest

from multiagent import tools as tool_registry
from multiagent.agents import AgentContext, BaseAgent
from multiagent.config import Settings
from multiagent.guardrails import BudgetExceeded
from multiagent.orchestrator import Orchestrator


class FakeLLM:
    """스크립트된 응답을 반환하는 가짜 LLM.

    입력: responses(순서대로 반환할 문자열), json_responses(JSON 모드용),
          raise_exc(설정 시 chat/chat_json에서 발생시킬 예외)
    출력: chat()/chat_json()/aclose() 메서드
    동작: 호출마다 다음 스크립트를 pop. raise_exc가 있으면 즉시 raise.
    """

    def __init__(self, responses=None, json_responses=None, raise_exc=None):
        self._responses = list(responses or [])
        self._json = list(json_responses or [])
        self._raise = raise_exc
        self.chat_calls = 0

    async def chat(self, messages, **kwargs):
        """입력: messages / 출력: 다음 스크립트 문자열."""
        self.chat_calls += 1
        if self._raise:
            raise self._raise
        if self._responses:
            return self._responses.pop(0)
        return '{"final": "fake answer"}'

    async def chat_json(self, messages, **kwargs):
        """입력: messages / 출력: 다음 스크립트 dict."""
        if self._raise:
            raise self._raise
        if self._json:
            return self._json.pop(0)
        return {"final": "fake"}

    async def aclose(self):
        """입력: 없음 / 출력: 없음."""


class FakeMCP:
    """MCPClient 흉내 — call_tool/aclose만 구현."""

    async def call_tool(self, name, arguments):
        """입력: tool 이름, 인자 / 출력: 가짜 결과."""
        return f"fake result from {name}"

    async def aclose(self):
        """입력: 없음 / 출력: 없음."""


@pytest.fixture(autouse=True)
def _registered_mcp():
    """입력: 없음 / 출력: FakeMCP.

    동작: 모든 테스트 전에 MCP 도구를 등록하고 가짜 클라이언트를 주입한다.
    register_mcp_tools는 멱등하지 않지만 같은 name 덮어쓰기라 안전하다.
    """
    tool_registry.set_mcp_client(FakeMCP())
    tool_registry.register_mcp_tools()
    return FakeMCP()


def _settings(**overrides):
    """입력: Settings 필드 덮어쓰기 / 출력: 테스트용 Settings."""
    base = {
        "openai_api_key": "sk-test",
        "agent_model": "gpt-4o-mini",
        "max_cost_usd": 1.0,
        "max_total_tokens": 100000,
        "agent_timeout_sec": 5.0,
        "max_review_rounds": 1,
        "review_pass_score": 0.7,
        "allow_write_tools": False,
    }
    base.update(overrides)
    return Settings(**base)


def _swap_llm(orch: Orchestrator, llm: FakeLLM) -> None:
    """입력: 오케스트레이터, 가짜 LLM / 출력: 없음.

    동작: orch.llm과 모든 에이전트의 llm을 Fake로 교체한다.
    """
    orch.llm = llm
    for agent in orch._agents.values():
        agent.llm = llm


async def test_agent_run_with_tool():
    """입력: BaseAgent + FakeLLM / 출력: AgentOutput.

    동작: JSON 프로토콜로 도구 호출 → 결과 수신 → 최종 답변까지 확인.
    """
    llm = FakeLLM(
        json_responses=[
            {"tool": "search_docs", "args": {"query": "RAG"}},
            {"final": "RAG는 검색 기반 생성입니다."},
        ],
    )
    ctx = AgentContext(allow_write=False, mcp=None)
    agent = BaseAgent("test", "You are a test agent.", ["search_docs"], llm, ctx)
    result = await agent.run("RAG가 뭐야?")
    assert result.content == "RAG는 검색 기반 생성입니다."
    assert "search_docs" in result.tool_calls_made


async def test_agent_tool_not_in_whitelist():
    """입력: 허용되지 않은 도구 호출 시도 / 출력: 에러 JSON 후 진행.

    동작: _tool_names에 없는 도구는 실행되지 않고 에러 JSON이 반환된다.
    """
    llm = FakeLLM(
        json_responses=[
            {"tool": "save_report", "args": {"title": "x", "content": "y"}},
            {"final": "done"},
        ],
    )
    ctx = AgentContext(allow_write=False, mcp=None)
    # search_docs만 허용 — save_report는 화이트리스트에 없음
    agent = BaseAgent("test", "test", ["search_docs"], llm, ctx)
    result = await agent.run("save a report")
    assert result.content == "done"
    # save_report는 tool_calls_made에 기록되지 않아야 함 (실행 안 됨)
    assert "save_report" not in result.tool_calls_made


async def test_orchestrator_budget_exceeded():
    """입력: chat에서 BudgetExceeded를 던지는 FakeLLM / 출력: 중단된 PipelineResult.

    동작: planner 단계에서 BudgetExceeded → final_report에 '[중단]' 포함.
    실제 LLMClient 없이도 예산 초과 시나리오를 검증한다.
    """
    settings = _settings()
    orch = Orchestrator(settings)
    fake = FakeLLM(raise_exc=BudgetExceeded("토큰 상한 초과: 999 > 100"))
    _swap_llm(orch, fake)

    result = await orch.run("테스트 주제")
    assert "[중단]" in result.final_report
    assert result.review_score == 0.0


async def test_orchestrator_full_pipeline():
    """입력: FakeLLM + 전체 파이프라인 / 출력: PipelineResult.

    동작: planner→researcher×2→writer→reviewer→publisher 순서로 실행되고
    최종 리포트가 생성되는지 확인. MCP는 autouse fixture의 FakeMCP.
    """
    settings = _settings(max_review_rounds=0)  # 리뷰 1회만

    # chat() 경로를 쓰는 에이전트: planner(도구없음), writer, reviewer, publisher
    chat_responses = [
        '["Q1", "Q2"]',                          # planner
        "# 리포트 초안\n내용",                      # writer
        '{"score": 0.9, "feedback": "좋음", "passed": true}',  # reviewer
        "# 최종 리포트\n완료",                      # publisher
    ]
    # chat_json() 경로를 쓰는 에이전트: researcher(도구 있음) × 2 + publisher(save_report)
    json_responses = [
        {"final": "A에 대한 답"},
        {"final": "B에 대한 답"},
        {"final": "# 최종 리포트\n완료"},  # publisher — save_report 도구 보유
    ]

    llm = FakeLLM(responses=chat_responses, json_responses=json_responses)
    orch = Orchestrator(settings)
    _swap_llm(orch, llm)

    result = await orch.run("테스트 주제")
    assert result.topic == "테스트 주제"
    assert len(result.plan) == 2
    assert len(result.research_notes) == 2
    assert result.review_score == 0.9
    assert "최종 리포트" in result.final_report
    assert result.elapsed_sec >= 0


async def test_agent_max_tool_rounds():
    """입력: 계속 도구만 호출하는 FakeLLM / 출력: 제한된 반복.

    동작: MAX_TOOL_ROUNDS(3)를 넘지 않는다 — 무한 루프 방지.
    """

    class ToolLoopLLM(FakeLLM):
        async def chat_json(self, messages, **kwargs):
            """입력: messages / 출력: 항상 tool 호출 JSON."""
            return {"tool": "search_docs", "args": {"query": "x"}}

        async def chat(self, messages, **kwargs):
            """입력: messages / 출력: 항상 tool 호출 JSON."""
            return '{"tool": "search_docs", "args": {"query": "x"}}'

    llm = ToolLoopLLM()
    ctx = AgentContext(allow_write=False, mcp=None)
    agent = BaseAgent("test", "test", ["search_docs"], llm, ctx)
    result = await agent.run("계속 검색")
    assert result.iterations <= 3  # MAX_TOOL_ROUNDS
