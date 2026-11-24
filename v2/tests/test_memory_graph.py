"""메모리 + 그래프 구조 테스트 — API 키 불필요 (FakeLLM 사용)."""


from rag_agent.graph import build_graph
from rag_agent.memory import LongTermMemory, extract_facts_prompt
from rag_agent.state import AgentState


class FakeLLM:
    """스크립트된 응답을 반환하는 가짜 LLM.

    입력: chat_responses(순서대로 반환할 문자열), chat_json_responses
    출력: chat()/chat_json()/embed() 메서드
    동작: 호출마다 다음 스크립트를 pop. embed는 고정 더미 벡터 반환.
    """

    def __init__(self, chat_responses=None, chat_json_responses=None):
        self._chat = list(chat_responses or ["답변"])
        self._json = list(chat_json_responses or [{}])
        self.chat_calls = 0

    async def chat(self, messages, **kwargs):
        """입력: messages / 출력: 다음 스크립트 문자열."""
        self.chat_calls += 1
        return self._chat.pop(0) if self._chat else "답변"

    async def chat_json(self, messages, **kwargs):
        """입력: messages / 출력: 다음 스크립트 dict."""
        return self._json.pop(0) if self._json else {}

    async def embed(self, texts):
        """입력: 텍스트 리스트 / 출력: 더미 벡터 리스트."""
        return [[0.1] * 8 for _ in texts]


class FakeRetriever:
    """고정 청크를 반환하는 가짜 리트리버."""

    def __init__(self, chunks=None):
        self._chunks = chunks or [
            {"id": "c1", "document": "연차는 11일입니다", "metadata": {"source": "vacation.md"}, "rrf": 0.5}
        ]

    async def retrieve(self, query):
        """입력: 쿼리 / 출력: 고정 청크 리스트."""
        return self._chunks


def test_longterm_memory(tmp_path):
    """입력: 임시 db 경로 / 출력: 없음(단언).

    동작: remember→recall_all→forget이 sqlite에서 동작하는지 확인.
    """
    mem = LongTermMemory(tmp_path / "lt.db")
    mem.remember("user_name", "김철수")
    mem.remember("pref", "간결한 답변")
    assert mem.recall_all() == {"user_name": "김철수", "pref": "간결한 답변"}
    mem.remember("user_name", "이영희")  # 갱신
    assert mem.recall_all()["user_name"] == "이영희"
    assert mem.forget("pref") is True
    assert "pref" not in mem.recall_all()
    mem.close()


def test_extract_facts_prompt():
    """입력: 사용자/어시스턴트 메시지 / 출력: 프롬프트 문자열.

    동작: 프롬프트에 두 메시지가 포함되고 JSON 형식 지시가 있는지 확인.
    """
    p = extract_facts_prompt("내 이름은 김철수야", "반갑습니다 김철수님")
    assert "김철수" in p
    assert "JSON" in p


async def test_graph_runs_to_completion(tmp_path):
    """입력: FakeLLM+FakeRetriever / 출력: 없음(단언).

    동작: use_hitl=False로 그래프를 실행해 answer까지 도달하는지 확인.
    각 노드(rewrite→retrieve→grade→generate→memory_update→extract_facts)가
    순서대로 실행되고 최종 answer가 반환되는지 검증.
    """
    llm = FakeLLM(
        chat_responses=[
            "연차 일수",          # rewrite
            "yes",                # grade
            "연차는 11일입니다.",  # generate
            "대화 요약",           # memory_update (메시지 부족으로 스킵될 수 있음)
            "[]",                 # extract_facts (chat_json)
        ],
        chat_json_responses=[{"scores": []}, []],
    )
    retriever = FakeRetriever()
    longterm = LongTermMemory(tmp_path / "lt.db")

    graph = build_graph(llm, retriever, longterm, use_hitl=False)
    state: AgentState = {
        "messages": [{"role": "user", "content": "연차 며칠이야?"}],
        "question": "연차 며칠이야?",
        "iteration": 0,
        "tokens_used": 0,
    }
    result = await graph.ainvoke(state)
    assert result["answer"] == "연차는 11일입니다."
    assert result["sources"] == ["vacation.md"]
    longterm.close()


async def test_graph_hitl_interrupt(tmp_path):
    """입력: use_hitl=True 그래프 / 출력: 없음(단언).

    동작: approve 노드에서 interrupt 발생 → __interrupt__에
    approval_request가 담기는지 확인. 이후 Command(resume=approved)
    로 재개해 END까지 도달하는지 검증.
    """
    llm = FakeLLM(
        chat_responses=["검색어", "yes", "답변입니다"],
        chat_json_responses=[[]],
    )
    retriever = FakeRetriever()
    longterm = LongTermMemory(tmp_path / "lt.db")

    # interrupt 재개(Command resume)에는 checkpointer가 필수 — 인메모리로 주입
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(llm, retriever, longterm, use_hitl=True, checkpointer=InMemorySaver())
    state: AgentState = {
        "messages": [{"role": "user", "content": "질문"}],
        "question": "질문",
        "iteration": 0,
        "tokens_used": 0,
    }
    # 첫 실행: approve에서 interrupt
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": "t1"}})
    assert "__interrupt__" in result
    interrupt_data = result["__interrupt__"][0].value
    assert interrupt_data["type"] == "approval_request"
    assert interrupt_data["draft_answer"] == "답변입니다"

    # 재개: 승인
    from langgraph.types import Command
    final = await graph.ainvoke(
        Command(resume={"approved": True, "feedback": ""}),
        config={"configurable": {"thread_id": "t1"}},
    )
    assert "__interrupt__" not in final or final.get("answer")
    longterm.close()
