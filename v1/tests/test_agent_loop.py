"""ReAct 루프 테스트: 가짜 ChatClient로 툴콜→최종응답 시나리오 검증."""

import pytest

from agent.agent import Agent
from agent.config import Settings
from agent.llm import LLMTurn, ToolCallSpec, estimate_cost_usd


def make_settings(**overrides) -> Settings:
    """입력: overrides(Settings 필드 덮어쓰기) / 출력: 테스트용 Settings.

    동작: 실제 .env를 읽지 않고 결정적 값으로 Settings를 만든다 —
    openai_api_key는 "sk-test"(실제 호출은 FakeClient가 가로챔).
    agent_max_iterations=5 등 루프 테스트에 필요한 기본값을 고정한다.
    """
    base = {
        "openai_api_key": "sk-test",
        "agent_model": "gpt-4o-mini",
        "agent_max_iterations": 5,
        "agent_max_tokens": 1024,
        "agent_temperature": 0.0,
    }
    base.update(overrides)
    return Settings(**base)


class FakeClient:
    """스크립트된 LLMTurn을 순서대로 반환하는 가짜 클라이언트.

    입력: turns(LLMTurn 리스트 — 호출 순서대로 소비)
    출력: ChatClient 프로토콜을 만족하는 객체
    동작: complete() 호출마다 _turns.pop(0)으로 다음 턴을 반환하고,
    호출 인자(messages/tools)를 calls에 기록한다 — 테스트가
    "두 번째 호출의 messages에 tool 결과가 들어갔는가"를 검증할 수 있다.
    """

    def __init__(self, turns: list[LLMTurn]):
        self._turns = list(turns)
        self.calls: list[dict] = []

    async def complete(self, messages, tools=None):
        """입력: messages, tools / 출력: 다음 스크립트된 LLMTurn.

        동작: messages는 list()로 복사해 저장한다 — 루프가 이후
        append를 계속해도 호출 '시점'의 스냅샷이 유지되어야 하기 때문.
        (참조 저장 시 나중에 messages[-1]이 최종 assistant가 되어
        tool 메시지 검증이 깨진다 — 실제로 겪은 버그)
        """
        self.calls.append({"messages": list(messages), "tools": tools})
        return self._turns.pop(0)

    def stream(self, messages, tools=None):  # pragma: no cover - 비스트림 테스트만 사용
        """입력: messages, tools / 출력: 없음.

        동작: 이 테스트 파일은 run()만 검증하므로 stream은 미구현.
        """
        raise NotImplementedError


async def test_single_tool_call_then_answer():
    """입력: 툴콜 턴 → 텍스트 턴 2개가 스크립트된 FakeClient.
    출력: 최종 답변이 두 번째 턴의 content.

    동작 검증: calculator 호출 → tool 메시지에 '"result": 42' 포함 →
    iterations=2, 토큰 합산(250/50), tool_names_used=["calculator"].
    """
    client = FakeClient(
        [
            LLMTurn(
                content=None,
                tool_calls=[
                    ToolCallSpec(id="c1", name="calculator", arguments='{"expression":"6*7"}')
                ],
                prompt_tokens=100,
                completion_tokens=20,
            ),
            LLMTurn(content="6 곱하기 7은 42입니다.", prompt_tokens=150, completion_tokens=30),
        ]
    )
    agent = Agent(client=client, settings=make_settings())
    result = await agent.run("6*7은 얼마야?")

    assert result.content == "6 곱하기 7은 42입니다."
    assert result.iterations == 2
    assert result.tool_names_used == ["calculator"]
    assert result.prompt_tokens == 250
    assert result.completion_tokens == 50

    # 두 번째 호출의 messages에 tool 결과("result": 42)가 포함됐는지 확인
    second_call_messages = client.calls[1]["messages"]
    tool_msgs = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert '"result": 42' in tool_msgs[0]["content"]


async def test_unknown_tool_returns_error_and_recovers():
    """입력: 미등록 도구(hack_tool) 호출 턴 → 텍스트 턴.
    출력: 두 번째 턴의 최종 답변.

    동작 검증: execute_tool이 예외 없이 에러 JSON을 반환하고,
    그 결과가 role="tool" 메시지로 두 번째 LLM 호출에 전달된다.
    루프가 죽지 않고 모델이 회복할 수 있음을 확인한다.
    """
    client = FakeClient(
        [
            LLMTurn(
                tool_calls=[ToolCallSpec(id="c1", name="hack_tool", arguments="{}")],
                prompt_tokens=10,
                completion_tokens=5,
            ),
            LLMTurn(content="그런 도구는 없어요.", prompt_tokens=20, completion_tokens=10),
        ]
    )
    agent = Agent(client=client, settings=make_settings())
    result = await agent.run("뭐든 해줘")
    assert result.content == "그런 도구는 없어요."
    tool_msg = client.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert "unknown tool" in tool_msg["content"]


async def test_max_iterations_forces_final_answer():
    """입력: 계속 툴만 호출하는 3턴 + 최종 텍스트 1턴, max_iterations=3.
    출력: 스크립트된 최종 답변.

    동작 검증: 반복 한도 도달 시 루프를 빠져나와 tools=None으로
    complete()를 1회 더 호출한다 — 모델이 더 이상 툴을 호출할 수
    없어 반드시 텍스트로 끝난다. 마지막 호출의 tools가 None인지 확인.
    """
    # 계속 툴만 호출하는 모델 → 한도 후 tools=None으로 최종 호출
    endless = LLMTurn(
        tool_calls=[ToolCallSpec(id="c", name="calculator", arguments='{"expression":"1+1"}')],
        prompt_tokens=10,
        completion_tokens=5,
    )
    client = FakeClient([endless] * 3 + [LLMTurn(content="정리하면 여기까지입니다.", prompt_tokens=5, completion_tokens=5)])
    agent = Agent(client=client, settings=make_settings(agent_max_iterations=3))
    result = await agent.run("계속 계산해")
    assert result.content == "정리하면 여기까지입니다."
    # 마지막 호출은 tools=None (강제 최종 답변)
    assert client.calls[-1]["tools"] is None


async def test_usage_and_cost_accumulate():
    """입력: 100만 입력+100만 출력 토큰을 쓰는 단일 턴.
    출력: cost_usd ≈ 0.75.

    동작 검증: gpt-4o-mini 가격($0.15/M 입력, $0.60/M 출력)으로
    (1M*0.15 + 1M*0.60)/1M = $0.75가 계산되는지 확인한다.
    """
    client = FakeClient([LLMTurn(content="hi", prompt_tokens=1_000_000, completion_tokens=1_000_000)])
    agent = Agent(client=client, settings=make_settings())
    result = await agent.run("hi")
    # gpt-4o-mini: $0.15/M input + $0.60/M output
    assert result.cost_usd == pytest.approx(0.75, abs=1e-6)


def test_estimate_cost_unknown_model_uses_default():
    """입력: 가격표에 없는 모델명 + 100만 입력 토큰.
    출력: cost ≈ 0.15.

    동작 검증: 미등록 모델은 gpt-4o-mini 가격으로 폴백 계산된다
    (비용을 0으로 두지 않는다).
    """
    cost = estimate_cost_usd("unknown-model-x", 1_000_000, 0)
    assert cost == pytest.approx(0.15)
