"""LangGraph 버전 — 같은 파이프라인을 StateGraph로 구현.

orchestrator.py(프레임워크 프리)와 비교하기 위한 참조 구현.
차이점:
- 상태 관리를 LangGraph의 TypedDict + add_messages가 담당
- 병렬 실행은 Send API로 표현
- 조건부 분기(리뷰 루프)는 add_conditional_edges
- 체크포인터로 thread_id별 상태 영속화 가능

실행: python -m frameworks.langgraph_impl "주제"
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from multiagent.config import get_settings
from multiagent.guardrails import BudgetTracker
from multiagent.llm import LLMClient

logger = logging.getLogger(__name__)


# ---------- 상태 스키마 ----------

class ReportState(TypedDict):
    """LangGraph 상태 — orchestrator의 PipelineResult에 대응.

    입력: topic
    출력: plan/subtasks/notes/draft/review/final_report
    동작: langgraph가 노드 반환 dict를 state에 merge한다.
    """

    topic: str
    plan: list[str]
    subtasks: list[str]
    notes: list[str]
    draft: str
    review_score: float
    review_feedback: str
    review_round: int
    final_report: str


# ---------- LLM 헬퍼 ----------

async def _chat_json(llm: LLMClient, prompt: str) -> dict:
    """입력: LLMClient, 프롬프트 / 출력: 파싱된 dict.

    동작: JSON 모드로 호출 후 파싱. 실패 시 빈 dict.
    """
    try:
        return await llm.chat_json([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=800)
    except Exception:
        return {}


# ---------- 노드 ----------

async def plan_node(state: ReportState, llm: LLMClient) -> dict:
    """입력: state(topic) / 출력: {plan, subtasks}.

    동작: 주제를 3-5개 하위 질문으로 분해.
    """
    prompt = (
        f"리서치 주제: {state['topic']}\n\n"
        "이를 3-5개의 구체적 검색 질문으로 분해하라. JSON 배열로만 답하라."
    )
    data = await _chat_json(llm, prompt)
    plan = data.get("plan", data) if isinstance(data, dict) else data
    if not isinstance(plan, list):
        plan = [state["topic"]]
    plan = [str(p) for p in plan[:5]]
    return {"plan": plan, "subtasks": plan}


async def research_node(state: ReportState, llm: LLMClient, subtask: str) -> dict:
    """입력: state, LLM, 개별 subtask(Send API로 주입) / 출력: {notes: [note]}.

    동작: 하나의 하위 질문을 조사해 노트 1개를 반환.
    Send API로 여러 researcher가 병렬 실행된다.
    """
    prompt = (
        f"조사 질문: {subtask}\n\n"
        "지식베이스를 검색해 답하라. 간결하게 2-4문장."
    )
    text = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=500)
    return {"notes": [f"## {subtask}\n{text}"]}


async def write_node(state: ReportState, llm: LLMClient) -> dict:
    """입력: state(topic, notes, review_feedback) / 출력: {draft}.

    동작: 조사 노트를 바탕으로 초안 작성. 리뷰 피드백이 있으면 반영.
    """
    notes_text = "\n\n".join(state.get("notes", []))
    feedback = state.get("review_feedback", "")
    prompt = (
        f"주제: {state['topic']}\n\n조사 노트:\n{notes_text}\n\n"
        "위 내용으로 리포트 초안을 작성하라."
    )
    if feedback:
        prompt += f"\n\n리뷰어 피드백:\n{feedback}\n이를 반영해 수정하라."
    draft = await llm.chat([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=1500)
    return {"draft": draft}


async def review_node(state: ReportState, llm: LLMClient) -> dict:
    """입력: state(topic, draft) / 출력: {review_score, review_feedback, review_round}.

    동작: 초안을 0-1로 채점하고 피드백을 생성.
    """
    prompt = (
        f"주제: {state['topic']}\n\n초안:\n{state['draft'][:3000]}\n\n"
        "정확성/완성도/명확성을 평가하라. JSON: {\"score\": 0.0-1.0, \"feedback\": \"...\"}"
    )
    data = await _chat_json(llm, prompt)
    score = float(data.get("score", 0.5))
    feedback = str(data.get("feedback", ""))
    return {
        "review_score": score,
        "review_feedback": feedback,
        "review_round": state.get("review_round", 0) + 1,
    }


async def publish_node(state: ReportState, llm: LLMClient) -> dict:
    """입력: state(topic, draft) / 출력: {final_report}.

    동작: 최종 포맷팅.
    """
    prompt = f"주제: {state['topic']}\n\n초안:\n{state['draft']}\n\n출판용으로 포맷팅하라."
    report = await llm.chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=2000)
    return {"final_report": report}


# ---------- 라우팅 ----------

def route_after_review(state: ReportState) -> str:
    """입력: state(review_score, review_round) / 출력: 다음 노드명.

    동작: 점수 < 0.7이고 2회 미만이면 write로(수정), 아니면 publish로.
    """
    if state["review_score"] < 0.7 and state["review_round"] < 2:
        return "write"
    return "publish"


def route_to_research(state: ReportState) -> list[Send]:
    """입력: state(subtasks) / 출력: Send 리스트.

    동작: 각 subtask에 대해 researcher 노드로 Send — LangGraph의
    map-reduce 패턴. 노드가 병렬로 실행되고 notes는 자동 병합된다.
    """
    return [Send("research", {"topic": state["topic"], "subtask": st}) for st in state["subtasks"]]


# ---------- 그래프 빌드 ----------

def build_graph(llm: LLMClient):
    """입력: LLMClient / 출력: 컴파일된 그래프.

    동작: StateGraph에 노드/엣지/조건부 분기/병렬 Send를 등록한다.
    """
    graph = StateGraph(ReportState)

    # 노드 래퍼: llm을 클로저로 주입
    graph.add_node("plan", lambda s: plan_node(s, llm))
    graph.add_node("research", lambda s: research_node(s, llm, s.get("subtask", "")))
    graph.add_node("write", lambda s: write_node(s, llm))
    graph.add_node("review", lambda s: review_node(s, llm))
    graph.add_node("publish", lambda s: publish_node(s, llm))

    graph.add_edge(START, "plan")
    # 병렬 팬아웃: plan → 각 subtask마다 research 노드
    graph.add_conditional_edges("plan", route_to_research, ["research"])
    # research 완료 후 write로 (모든 병렬 결과가 모이면 자동 진행)
    graph.add_edge("research", "write")
    graph.add_edge("write", "review")
    graph.add_conditional_edges("review", route_after_review, {"write": "write", "publish": "publish"})
    graph.add_edge("publish", END)

    return graph.compile()


async def run(topic: str) -> None:
    """입력: 주제 / 출력: 콘솔 출력.

    동작: LangGraph 버전으로 파이프라인 실행.
    """
    settings = get_settings()
    budget = BudgetTracker(settings.max_cost_usd, settings.max_total_tokens, settings.agent_model)
    llm = LLMClient(settings, budget)
    graph = build_graph(llm)

    result = await graph.ainvoke({"topic": topic, "review_round": 0})
    print(f"\n{'='*60}")
    print(f"주제: {topic}")
    print(f"계획: {result.get('plan', [])}")
    print(f"리뷰 점수: {result.get('review_score', 0):.2f} ({result.get('review_round', 0)}회)")
    print(f"예산: {budget.summary()}")
    print(f"\n[최종 리포트]\n{result.get('final_report', '')}")
    print(f"{'='*60}")
    await llm.aclose()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('사용법: python -m frameworks.langgraph_impl "주제"')
        sys.exit(1)
    asyncio.run(run(" ".join(sys.argv[1:])))
