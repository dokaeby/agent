"""LangGraph 상태 스키마 — 그래프 전체가 공유하는 상태.

모든 노드는 이 TypedDict의 일부 필드만 읽고/쓴다.
langgraph는 반환된 dict의 키만 state에 merge한다.
"""

from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from langgraph.graph.message import add_messages


class RetrievedChunk(TypedDict):
    """검색된 청크 하나 — retrieve 노드가 만드는 항목.

    입력: Retriever.retrieve() 결과
    출력: state["retrieved"]의 원소
    동작: document(청크 원문), metadata(출처/제목), 점수 필드들.
    """

    id: str
    document: str
    metadata: dict
    rrf: float
    rerank_score: NotRequired[float]


class AgentState(TypedDict):
    """RAG 에이전트 그래프의 전체 상태.

    입력: 사용자 질문 + thread_id(세션)
    출력: 최종 answer + 메타데이터(토큰, 사용된 청크 등)
    동작:
    - messages: 대화 이력 — add_messages 리듀서가 append/갱신을 관리
    - retrieved: 검색된 청크 — retrieve 노드가 덮어씀
    - graded: 관련성 통과 여부 — grade 노드가 기록
    - iteration: retrieve 재시도 카운터 (최대 2회)
    - approval: HITL 승인 결과 — interrupt() resume 시 주입
    """

    messages: Annotated[list, add_messages]
    question: str
    rewritten_query: str
    retrieved: list[RetrievedChunk]
    graded_relevant: bool
    iteration: int
    context_summary: str  # 단기 메모리 요약
    user_facts: dict[str, str]  # 장기 메모리 스냅샷
    answer: str
    approval: NotRequired[dict]  # {"approved": bool, "feedback": str}
    tokens_used: int
    sources: list[str]  # 답변에 인용된 문서 출처
