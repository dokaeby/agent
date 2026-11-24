"""스트리밍 루프 테스트."""

from agent.agent import Agent
from agent.config import Settings
from agent.llm import LLMTurn, TextDelta, ToolCallDelta, ToolCallSpec, TurnComplete


class FakeStreamClient:
    """스크립트된 이벤트 시퀀스를 스트리밍하는 가짜.

    입력: scripts(턴별 StreamEvent 리스트의 리스트)
    출력: ChatClient 프로토콜 구현체
    동작: stream() 호출마다 다음 스크립트를 꺼내 async generator로 반환.
    각 스크립트는 반드시 TurnComplete로 끝나야 Agent가 턴을 확정한다.
    complete()는 한도 초과 시 최종 호출용으로 "fallback"을 반환한다.
    """

    def __init__(self, scripts: list[list]):
        self._scripts = [list(s) for s in scripts]
        self.calls = 0

    async def complete(self, messages, tools=None):  # 최종 강제 호출용
        """입력: messages, tools / 출력: 고정 LLMTurn("fallback").

        동작: 이 테스트들은 한도 초과 경로를 타지 않으므로 호출되면 안 되지만,
        혹시 호출돼도 안전한 값을 반환한다.
        """
        self.calls += 1
        return LLMTurn(content="fallback", prompt_tokens=0, completion_tokens=0)

    def stream(self, messages, tools=None):
        """입력: messages, tools / 출력: async generator.

        동작: scripts에서 다음 이벤트 리스트를 pop해 yield하는 generator를 반환.
        """
        events = self._scripts.pop(0)

        async def gen():
            """입력: 없음(클로저의 events) / 출력: 이벤트를 순서대로 yield.

            동작: 스크립트된 StreamEvent를 하나씩 흘려보낸다.
            """
            for e in events:
                yield e

        return gen()


def make_settings(**kw):
    """입력: Settings 필드 덮어쓰기 / 출력: 테스트용 Settings.

    동작: 결정적 기본값(api_key="sk-test", max_iterations=4)으로 생성.
    """
    return Settings(openai_api_key="sk-test", agent_model="gpt-4o-mini", agent_max_iterations=4, **kw)


async def test_stream_text_only():
    """입력: TextDelta 2개 + TurnComplete 1개짜리 스크립트.
    출력: text 이벤트 2개 + done 이벤트 1개.

    동작 검증: 델타가 순서대로("Hello"," world") yield되고,
    done.result.content가 조각을 합친 "Hello world"인지 확인한다.
    """
    client = FakeStreamClient(
        [
            [
                TextDelta("Hello"),
                TextDelta(" world"),
                TurnComplete(LLMTurn(content="Hello world", prompt_tokens=10, completion_tokens=5)),
            ]
        ]
    )
    agent = Agent(client=client, settings=make_settings())
    events = [e async for e in agent.run_stream("hi")]

    texts = [e["delta"] for e in events if e["type"] == "text"]
    assert texts == ["Hello", " world"]
    done = events[-1]
    assert done["type"] == "done"
    assert done["result"].content == "Hello world"


async def test_stream_tool_call_then_answer():
    """입력: 툴콜 스트림(조각 2개+TurnComplete) → 텍스트 스트림.
    출력: tool_call_delta/tool_call/tool_result/text/done 이벤트들.

    동작 검증: ToolCallDelta 조각은 UI 표시용으로 흘러가고,
    TurnComplete의 조립된 ToolCallSpec으로 실제 execute_tool이 실행된다
    (calculator 2+2=4). 최종 content는 두 번째 턴의 "4입니다.".
    """
    client = FakeStreamClient(
        [
            # 턴 1: 툴콜 스트림 (조각 도착)
            [
                ToolCallDelta(index=0, name="calculator", arguments_delta='{"exp'),
                ToolCallDelta(index=0, name=None, arguments_delta='ression":"2+2"}'),
                TurnComplete(
                    LLMTurn(
                        content=None,
                        tool_calls=[
                            ToolCallSpec(id="c1", name="calculator", arguments='{"expression":"2+2"}')
                        ],
                        prompt_tokens=10,
                        completion_tokens=5,
                    )
                ),
            ],
            # 턴 2: 최종 텍스트
            [
                TextDelta("4입니다."),
                TurnComplete(LLMTurn(content="4입니다.", prompt_tokens=20, completion_tokens=8)),
            ],
        ]
    )
    agent = Agent(client=client, settings=make_settings())
    events = [e async for e in agent.run_stream("2+2는?")]

    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert events[-1]["type"] == "done"
    assert events[-1]["result"].content == "4입니다."
    assert events[-1]["result"].tool_names_used == ["calculator"]
