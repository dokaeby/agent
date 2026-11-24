"""세션 러너 — 그래프 + 체크포인터 + 트레이서를 묶는 실행 계층.

역할:
- SqliteSaver 체크포인터로 thread_id별 대화 상태를 영속화
  → 프로세스를 재시작해도 같은 thread_id로 이어서 대화 가능
- Tracer로 노드별 실행 시간을 JSONL에 기록
- HITL interrupt 발생 시 승인 요청을 반환하고, resume()으로 재개
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .bm25 import BM25Index
from .config import Settings, get_settings
from .graph import build_graph
from .llm import LLMClient
from .memory import LongTermMemory
from .retrieval import Retriever
from .tracing import Tracer
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


class AgentSession:
    """컴파일된 그래프 + 체크포인터 + 트레이서를 묶은 세션 러너.

    입력: Settings(없으면 싱글턴)
    출력: ask()/resume() 메서드
    동작:
    - __init__에서 VectorStore/BM25/Retriever/LongTermMemory/LLM을 조립하고
      SqliteSaver로 그래프를 컴파일한다.
    - ask(): 새 메시지를 그래프에 넣고 실행. interrupt가 발생하면
      approval_request를 반환한다(HITL 대기).
    - resume(): interrupt된 thread를 승인/거절로 재개한다.
    """

    def __init__(self, settings: Settings | None = None, *, use_hitl: bool = True) -> None:
        """입력: settings, use_hitl / 출력: 인스턴스.

        동작: 모든 의존성을 생성하고 checkpointer로 그래프를 컴파일한다.
        """
        self.settings = settings or get_settings()
        self.llm = LLMClient(self.settings)
        self.vs = VectorStore(self.settings.chroma_dir)
        self.bm25 = BM25Index.load(self.settings.bm25_index_path)
        self.retriever = Retriever(
            self.vs, self.bm25, self.llm,
            retrieve_top_k=self.settings.retrieve_top_k,
            rerank_candidates=self.settings.rerank_candidates,
            final_top_k=self.settings.final_top_k,
        )
        self.longterm = LongTermMemory(self.settings.longterm_db)
        self.tracer = Tracer(self.settings.traces_dir)

        # 체크포인터: sqlite 파일에 thread_id별 상태를 저장
        self._conn = sqlite3.connect(
            str(self.settings.checkpoint_db), check_same_thread=False
        )
        self._checkpointer = SqliteSaver(self._conn)
        # checkpointer를 주입해 컴파일 — thread_id별 상태가 sqlite에 영속화된다
        self._app = build_graph(
            self.llm, self.retriever, self.longterm,
            use_hitl=use_hitl, checkpointer=self._checkpointer,
        )

    async def ask(self, message: str, thread_id: str = "default") -> dict[str, Any]:
        """입력: 사용자 메시지, 세션 thread_id / 출력: 결과 dict.

        출력 형식:
        - 정상 완료: {"status":"done","answer":...,"sources":...,"tokens_used":...}
        - HITL 대기: {"status":"approval_required","draft_answer":...,"sources":...}

        동작:
        1. tracer.start_run으로 run_id 발급
        2. graph.ainvoke로 실행 — interrupt 발생 시 approval_request 반환
        3. 완료 시 tracer.end_run 기록
        """
        run_id = self.tracer.start_run(message, thread_id)
        state = {
            "messages": [{"role": "user", "content": message}],
            "question": message,
            "iteration": 0,
            "tokens_used": 0,
        }
        config = {"configurable": {"thread_id": thread_id}}

        try:
            final = await self._app.ainvoke(state, config=config)

            # interrupt 발생 여부 확인
            # ainvoke는 interrupt 시 예외를 던지지 않고 __interrupt__ 키를 반환
            interrupts = final.get("__interrupt__")
            if interrupts:
                req = interrupts[0].value if interrupts else {}
                self.tracer.end_run(run_id, "", 0, status="approval_required")
                return {
                    "status": "approval_required",
                    "thread_id": thread_id,
                    "draft_answer": req.get("draft_answer", ""),
                    "sources": req.get("sources", []),
                    "run_id": run_id,
                }

            answer = final.get("answer", "")
            tokens = final.get("tokens_used", 0)
            self.tracer.end_run(run_id, answer, tokens, status="done")
            return {
                "status": "done",
                "answer": answer,
                "sources": final.get("sources", []),
                "tokens_used": tokens,
                "run_id": run_id,
            }
        except Exception as exc:
            self.tracer.end_run(run_id, "", 0, status="error", error=str(exc))
            raise

    async def resume(
        self, thread_id: str, approved: bool, feedback: str = ""
    ) -> dict[str, Any]:
        """입력: thread_id, 승인 여부, 피드백(거절 시) / 출력: ask()와 동일 형식.

        동작: Command(resume={...})로 interrupt를 재개한다.
        approved=True면 END, False면 rewrite로 돌아가 피드백 반영.
        """
        run_id = self.tracer.start_run(f"[resume] approved={approved}", thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        try:
            final = await self._app.ainvoke(
                Command(resume={"approved": approved, "feedback": feedback}),
                config=config,
            )
            interrupts = final.get("__interrupt__")
            if interrupts:
                req = interrupts[0].value if interrupts else {}
                return {
                    "status": "approval_required",
                    "thread_id": thread_id,
                    "draft_answer": req.get("draft_answer", ""),
                    "sources": req.get("sources", []),
                    "run_id": run_id,
                }
            answer = final.get("answer", "")
            self.tracer.end_run(run_id, answer, final.get("tokens_used", 0), status="done")
            return {
                "status": "done",
                "answer": answer,
                "sources": final.get("sources", []),
                "tokens_used": final.get("tokens_used", 0),
                "run_id": run_id,
            }
        except Exception as exc:
            self.tracer.end_run(run_id, "", 0, status="error", error=str(exc))
            raise

    async def aclose(self) -> None:
        """입력: 없음 / 출력: 없음. LLM/DB 연결을 정리한다."""
        await self.llm.aclose()
        self.longterm.close()
        self._conn.close()
