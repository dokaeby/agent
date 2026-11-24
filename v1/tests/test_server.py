"""FastAPI 서버 테스트 (가짜 클라이언트 주입)."""

from fastapi.testclient import TestClient

from agent.config import Settings
from agent.llm import LLMTurn, ToolCallSpec
from agent.server import create_app


class FakeClient:
    """스크립트된 LLMTurn을 순서대로 반환하는 가짜 클라이언트.

    입력: turns(LLMTurn 리스트)
    출력: ChatClient 프로토콜 구현체
    동작: complete()마다 pop(0)으로 다음 턴 반환. stream은 미구현 —
    /chat 엔드포인트는 run()(비스트림)만 사용한다.
    """

    def __init__(self, turns):
        self._turns = list(turns)

    async def complete(self, messages, tools=None):
        """입력: messages, tools / 출력: 다음 LLMTurn.

        동작: 스크립트 소비만 하고 호출 기록은 하지 않는다(이 파일의
        테스트는 HTTP 레이어만 검증).
        """
        return self._turns.pop(0)

    def stream(self, messages, tools=None):
        """입력: messages, tools / 출력: 없음.

        동작: 미구현 — NotImplementedError.
        """
        raise NotImplementedError


def _settings() -> Settings:
    """입력: 없음 / 출력: 테스트용 Settings.

    동작: 실제 .env와 무관하게 결정적 값으로 생성한다.
    """
    return Settings(openai_api_key="sk-test", agent_model="gpt-4o-mini")


def test_healthz():
    """입력: FakeClient+settings로 만든 앱 / 출력: 없음(단언).

    동작: GET /healthz가 200과 status="ok"를 반환하는지 확인한다.
    """
    app = create_app(client=FakeClient([]), settings=_settings())
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_tools_endpoint():
    """입력: 앱 인스턴스 / 출력: 없음(단언).

    동작: GET /tools가 등록된 도구 목록을 반환하고 calculator가
    포함되는지 확인한다.
    """
    app = create_app(client=FakeClient([]), settings=_settings())
    client = TestClient(app)
    resp = client.get("/tools")
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()["tools"]}
    assert "calculator" in names


def test_chat_with_tool():
    """입력: 툴콜 턴→텍스트 턴 스크립트 + POST /chat {"message":"3+4는?"}.
    출력: 없음(단언).

    동작: HTTP 요청이 ReAct 루프 전체를 통과해 ChatResponse로
    직렬화되는지 확인한다 — reply, tools_used, 토큰 합산(10+15=25),
    cost_usd>0까지 엔드투엔드로 검증.
    """
    fake = FakeClient(
        [
            LLMTurn(
                tool_calls=[
                    ToolCallSpec(id="c1", name="calculator", arguments='{"expression":"3+4"}')
                ],
                prompt_tokens=10,
                completion_tokens=5,
            ),
            LLMTurn(content="3+4는 7입니다.", prompt_tokens=15, completion_tokens=8),
        ]
    )
    app = create_app(client=fake, settings=_settings())
    client = TestClient(app)
    resp = client.post("/chat", json={"message": "3+4는?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "3+4는 7입니다."
    assert body["tools_used"] == ["calculator"]
    assert body["prompt_tokens"] == 25
    assert body["cost_usd"] > 0


def test_chat_validation():
    """입력: message="" (빈 문자열) / 출력: 없음(단언).

    동작: ChatRequest의 min_length=1 검증이 FastAPI 레벨에서
    422를 반환하는지 확인한다 — 앱 코드까지 도달하지 않는다.
    """
    app = create_app(client=FakeClient([]), settings=_settings())
    client = TestClient(app)
    assert client.post("/chat", json={"message": ""}).status_code == 422
