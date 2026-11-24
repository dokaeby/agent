"""LangGraph RAG 파이프라인 — 노드/엣지/조건부 분기/HITL.

그래프 흐름:
  START → rewrite → retrieve → grade ─┬─(관련없음 & 재시도 가능)→ retrieve
                                      └─(관련있음)→ generate → approve(HITL)
                                                                       │
                                            ┌──────────────────────────┤
                                            │ approved → END           │
                                            │ rejected → rewrite(피드백 반영)
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .llm import LLMClient
from .memory import LongTermMemory, extract_facts_prompt, summarize_conversation
from .retrieval import Retriever
from .state import AgentState

logger = logging.getLogger(__name__)

MAX_RETRIEVE_ITERATIONS = 2  # grade 실패 시 retrieve 재시도 상한


def _msg_role(m: Any) -> str:
    """입력: 메시지(dict 또는 LangChain Message 객체) / 출력: 역할 문자열.

    동작: add_messages 리듀서가 dict 입력을 HumanMessage/AIMessage 객체로
    변환하므로, state["messages"]는 객체가 섞여 있을 수 있다.
    dict는 "role" 키, 객체는 .type(human/ai/system/tool)을 읽어
    OpenAI 역할명(user/assistant/system/tool)으로 정규화한다.
    """
    if isinstance(m, dict):
        return str(m.get("role", ""))
    t = getattr(m, "type", "")
    return {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(t, str(t))


def _msg_content(m: Any) -> str:
    """입력: 메시지(dict 또는 Message 객체) / 출력: content 문자열.

    동작: dict는 ["content"], 객체는 .content 속성을 읽는다.
    content가 list(멀티모달)면 문자열로 변환한다.
    """
    if isinstance(m, dict):
        c = m.get("content", "")
    else:
        c = getattr(m, "content", "")
    return c if isinstance(c, str) else str(c)


def _msg_dicts(messages: list[Any]) -> list[dict[str, str]]:
    """입력: 메시지 리스트 / 출력: {"role","content"} dict 리스트.

    동작: summarize_conversation 등 dict를 기대하는 함수에 넘기기 위한 변환.
    """
    return [{"role": _msg_role(m), "content": _msg_content(m)} for m in messages]


def build_graph(
    llm: LLMClient,
    retriever: Retriever,
    longterm: LongTermMemory,
    *,
    use_hitl: bool = True,
    checkpointer: Any = None,
):
    """입력: LLM/Retriever/LongTermMemory + HITL 여부 + checkpointer.
    출력: 컴파일된 LangGraph.

    동작: 노드 6개 + 조건부 엣지 2개로 StateGraph를 구성한다.
    use_hitl=False면 approve 노드를 건너뛰고 generate→END로 직행한다
    (평가/테스트 시 HITL 없이 자동 실행하기 위함).
    checkpointer를 넘기면 thread_id별 상태가 영속화된다 —
    SqliteSaver를 주면 프로세스 재시작 후에도 대화가 이어진다.
    """

    # ---------- 노드 ----------

    async def rewrite(state: AgentState) -> dict[str, Any]:
        """입력: state(messages/question/approval.feedback).
        출력: {"rewritten_query": ...}

        동작: 사용자 질문을 검색에 최적인 쿼리로 재작성한다.
        HITL에서 거절+피드백이 있으면 피드백을 반영해 재작성.
        """
        feedback = (state.get("approval") or {}).get("feedback", "")
        last_user = next(
            (_msg_content(m) for m in reversed(state["messages"]) if _msg_role(m) == "user"),
            state["question"],
        )
        prompt = (
            "사용자 질문을 문서 검색에 최적인 검색 쿼리로 재작성하라. "
            "핵심 명사와 의도를 유지하고, 답변은 쓰지 마라. 쿼리만 출력하라.\n\n"
            f"질문: {last_user}"
        )
        if feedback:
            prompt += f"\n\n이전 답변이 거절됐다. 피드백: {feedback}"
        rewritten = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=100)
        return {"rewritten_query": rewritten.strip() or last_user}

    async def retrieve(state: AgentState) -> dict[str, Any]:
        """입력: state(rewritten_query, iteration).
        출력: {"retrieved": [...], "iteration": +1}

        동작: Retriever.retrieve()로 하이브리드 검색+리랭크.
        iteration을 증가시켜 grade의 재시도 상한을 관리한다.
        """
        chunks = await retriever.retrieve(state["rewritten_query"])
        return {"retrieved": chunks, "iteration": state.get("iteration", 0) + 1}

    async def grade(state: AgentState) -> dict[str, Any]:
        """입력: state(rewritten_query, retrieved).
        출력: {"graded_relevant": bool}

        동작: 검색된 청크가 질문에 답할 수 있는지 LLM이 판정한다.
        retrieved가 비어있으면 False. 판정 실패 시 True로 간주
        (검색 결과를 버리는 것보다 생성 단계로 보내는 게 낫다).
        """
        chunks = state.get("retrieved", [])
        if not chunks:
            return {"graded_relevant": False}
        context = "\n---\n".join(c["document"][:500] for c in chunks[:3])
        prompt = (
            "다음 문서 조각들이 질문에 답하는 데 충분한 정보를 담고 있는지 "
            "yes/no로만 답하라.\n\n"
            f"질문: {state['rewritten_query']}\n\n문서:\n{context}"
        )
        try:
            verdict = await llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10
            )
            return {"graded_relevant": verdict.strip().lower().startswith("yes")}
        except Exception:  # noqa: BLE001 - 판정 실패 시 생성으로 진행
            logger.warning("grade 판정 실패 — relevant로 간주")
            return {"graded_relevant": True}

    async def generate(state: AgentState) -> dict[str, Any]:
        """입력: state(rewritten_query, retrieved, messages, context_summary, user_facts).
        출력: {"answer": ..., "sources": [...], "tokens_used": ...}

        동작: 검색된 청크를 컨텍스트로 답변 생성.
        - 단기 메모리(context_summary)와 장기 메모리(user_facts)를 프롬프트에 주입
        - 관련 청크가 없으면 '모르겠다'고 명시 (환각 방지)
        - sources에 인용된 문서의 출처(metadata.source/title)를 기록
        """
        chunks = state.get("retrieved", [])
        context = "\n\n".join(
            f"[문서 {i+1}] {c['document']}" for i, c in enumerate(chunks)
        )
        system_parts = [
            "너는 문서 기반 답변 어시스턴트다. 주어진 문서만 근거로 답하고, "
            "문서에 없는 내용은 '제공된 문서에서 확인할 수 없다'고 명시하라.",
        ]
        if state.get("context_summary"):
            system_parts.append(f"대화 요약: {state['context_summary']}")
        if state.get("user_facts"):
            facts = ", ".join(f"{k}={v}" for k, v in state["user_facts"].items())
            system_parts.append(f"사용자 정보: {facts}")
        system_parts.append(f"참고 문서:\n{context if context else '(없음)'}")

        messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
        # 최근 대화 4개만 유지해 토큰 절약 (Message 객체/dict 모두 처리)
        messages += [
            {"role": _msg_role(m), "content": _msg_content(m)}
            for m in state["messages"][-4:]
            if _msg_role(m) in ("user", "assistant")
        ]
        answer = await llm.chat(messages)
        sources = [
            c["metadata"].get("source", c["id"]) for c in chunks
        ]
        return {"answer": answer, "sources": sources, "tokens_used": len(answer) // 4}

    async def memory_update(state: AgentState) -> dict[str, Any]:
        """입력: state(messages).
        출력: {"context_summary": ...} (필요 시)

        동작: 메시지가 임계치(SUMMARIZE_THRESHOLD)를 넘으면 대화를 요약해
        단기 메모리에 저장한다. 장기 메모리는 answer 생성 후 별도 추출.
        """
        msgs = _msg_dicts(state.get("messages", []))
        if len(msgs) < 8:  # 요약할 만큼 길지 않으면 스킵
            return {}
        try:
            summary = await summarize_conversation(llm, msgs)
            return {"context_summary": summary}
        except Exception as exc:  # noqa: BLE001 - 요약 실패는 치명적이지 않음
            logger.warning("대화 요약 실패: %s", exc)
            return {}

    async def extract_user_facts(state: AgentState) -> dict[str, Any]:
        """입력: state(messages, answer).
        출력: {"user_facts": 갱신된 dict}

        동작: 마지막 user/assistant 교환에서 개인 정보를 추출해
        LongTermMemory에 저장하고 state에도 반영한다.
        """
        msgs = [m for m in _msg_dicts(state.get("messages", [])) if m["role"] in ("user", "assistant")]
        if len(msgs) < 2:
            return {}
        user_msg = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        asst_msg = state.get("answer") or next(
            (m["content"] for m in reversed(msgs) if m["role"] == "assistant"), ""
        )
        try:
            facts_text = extract_facts_prompt(user_msg, asst_msg)
            data = await llm.chat_json(
                [{"role": "user", "content": facts_text}], temperature=0.0, max_tokens=300
            )
            new_facts = {f["key"]: f["value"] for f in data if isinstance(f, dict) and "key" in f}
            for k, v in new_facts.items():
                longterm.remember(k, v)
            if new_facts:
                merged = longterm.recall_all()
                return {"user_facts": merged}
        except Exception as exc:  # noqa: BLE001 - 사실 추출 실패는 무시
            logger.warning("사용자 사실 추출 실패: %s", exc)
        return {}

    async def approve(state: AgentState) -> dict[str, Any]:
        """입력: state(answer, sources).
        출력: {"approval": {"approved": bool, "feedback": str}}

        동작: HITL — interrupt()로 실행을 일시정지하고 사람의 결정을 기다린다.
        resume 값은 Command(resume={...})로 들어온다.
        use_hitl=False면 이 노드는 그래프에 포함되지 않는다.
        """
        decision = interrupt(
            {
                "type": "approval_request",
                "question": state["question"],
                "draft_answer": state["answer"],
                "sources": state.get("sources", []),
            }
        )
        # decision은 Command(resume=...)의 값 — {"approved": bool, "feedback": str}
        return {"approval": decision or {"approved": True, "feedback": ""}}

    # ---------- 조건부 라우팅 ----------

    def route_after_grade(state: AgentState) -> str:
        """입력: state(graded_relevant, iteration).
        출력: 다음 노드 이름.

        동작: 관련 없고 재시도 횟수 남았으면 retrieve로, 아니면 generate로.
        """
        if state.get("graded_relevant"):
            return "generate"
        if state.get("iteration", 0) < MAX_RETRIEVE_ITERATIONS:
            return "retrieve"
        return "generate"  # 재시도 소진 → 있는 자료로 생성(없으면 '모른다')

    def route_after_approve(state: AgentState) -> str:
        """입력: state(approval).
        출력: "END" 또는 "rewrite".

        동작: 승인됐으면 종료, 거절됐으면 피드백을 반영해 rewrite부터 재시도.
        """
        approval = state.get("approval") or {}
        return "END" if approval.get("approved", True) else "rewrite"

    # ---------- 그래프 조립 ----------

    graph = StateGraph(AgentState)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("generate", generate)
    graph.add_node("memory_update", memory_update)
    graph.add_node("extract_facts", extract_user_facts)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", route_after_grade, {"retrieve": "retrieve", "generate": "generate"}
    )
    graph.add_edge("generate", "memory_update")
    graph.add_edge("memory_update", "extract_facts")

    if use_hitl:
        graph.add_node("approve", approve)
        graph.add_edge("extract_facts", "approve")
        graph.add_conditional_edges(
            "approve", route_after_approve, {"END": END, "rewrite": "rewrite"}
        )
    else:
        graph.add_edge("extract_facts", END)

    return graph.compile(checkpointer=checkpointer)
