"""RAG 에이전트 CLI.

사용법:
  python -m rag_agent.cli ingest            # data/docs 인제스천
  python -m rag_agent.cli ask "질문"        # 1회 질문 (HITL 끔)
  python -m rag_agent.cli chat              # 대화형 (HITL 켬)
  python -m rag_agent.cli eval              # 평가 데이터셋 실행
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from .bm25 import BM25Index
from .config import get_settings
from .ingest import ingest_directory
from .llm import LLMClient
from .runner import AgentSession
from .vectorstore import VectorStore


async def _ingest() -> None:
    """입력: 없음 / 출력: 콘솔 로그.

    동작: data/docs/의 문서를 청킹→임베딩→VectorStore+BM25에 저장한다.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        sys.exit("OPENAI_API_KEY가 없습니다. .env.example을 참고해 .env를 만드세요.")
    llm = LLMClient(settings)
    vs = VectorStore(settings.chroma_dir)
    bm25 = BM25Index()
    docs_dir = Path("data/docs")
    stats = await ingest_directory(docs_dir, vs, bm25, llm, settings)
    print(f"ingest 완료: {stats}")
    await llm.aclose()


async def _ask(question: str, *, interactive: bool = False) -> None:
    """입력: 질문 문자열, 대화형 여부 / 출력: 콘솔 출력.

    동작: AgentSession으로 질문을 실행한다.
    - HITL 대기 시 draft answer를 보여주고 승인/거절을 입력받는다.
    - 거절 시 피드백을 받아 resume으로 재시도한다.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        sys.exit("OPENAI_API_KEY가 없습니다.")
    session = AgentSession(settings, use_hitl=interactive)
    thread_id = "cli-default"
    try:
        result = await session.ask(question, thread_id=thread_id)
        while result["status"] == "approval_required":
            print(f"\n[초안 답변]\n{result['draft_answer']}")
            print(f"출처: {', '.join(result['sources'])}")
            ans = input("\n승인? [y=승인 / n=거절+피드백 / q=종료]: ").strip().lower()
            if ans == "q":
                return
            if ans == "y":
                result = await session.resume(thread_id, approved=True)
            else:
                feedback = input("피드백: ").strip()
                result = await session.resume(thread_id, approved=False, feedback=feedback)
        if result["status"] == "done":
            print(f"\n[답변]\n{result['answer']}")
            print(f"\n출처: {', '.join(result['sources'])}")
            print(f"토큰 추정: {result['tokens_used']}")
    finally:
        await session.aclose()


async def _chat() -> None:
    """입력: 없음 / 출력: 콘솔 REPL.

    동작: 대화형 루프. exit/quit으로 종료. HITL 승인 단계 포함.
    """
    print("RAG 에이전트 REPL (종료: exit/quit)")
    settings = get_settings()
    if not settings.llm_enabled:
        sys.exit("OPENAI_API_KEY가 없습니다.")
    session = AgentSession(settings, use_hitl=True)
    history: list[str] = []
    try:
        while True:
            try:
                msg = input("\nuser> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not msg or msg.lower() in {"exit", "quit"}:
                break
            history.append(msg)
            await _ask_in_session(session, msg, history)
    finally:
        await session.aclose()


async def _ask_in_session(session: AgentSession, message: str, history: list[str]) -> None:
    """입력: session, 메시지, 히스토리 / 출력: 콘솔 출력.

    동작: _ask와 동일하지만 기존 session을 재사용한다.
    """
    thread_id = "cli-chat"
    result = await session.ask(message, thread_id=thread_id)
    while result["status"] == "approval_required":
        print(f"\n[초안]\n{result['draft_answer']}")
        ans = input("승인? [y/n/q]: ").strip().lower()
        if ans == "q":
            return
        if ans == "y":
            result = await session.resume(thread_id, approved=True)
        else:
            feedback = input("피드백: ").strip()
            result = await session.resume(thread_id, approved=False, feedback=feedback)
    if result["status"] == "done":
        print(f"\nassistant: {result['answer']}")
        history.append(result["answer"])


async def _eval() -> None:
    """입력: 없음 / 출력: 콘솔 평가 리포트.

    동작: evals/dataset.jsonl로 평가를 실행하고 결과를 저장한다.
    """
    import json
    import time

    from .eval.run import run_eval

    dataset = Path("evals/dataset.jsonl")
    if not dataset.exists():
        sys.exit(f"데이터셋이 없습니다: {dataset}")
    report = await run_eval(dataset)
    print("\n=== 평가 결과 ===")
    print(
        f"항목: {report['items']} | 재현율: {report['avg_retrieval_recall']:.1%} "
        f"| 점수: {report['avg_answer_score']:.3f}"
    )
    out = f"evals/report_{int(time.time())}.json"
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {out}")


def main() -> None:
    """입력: sys.argv / 출력: 없음.

    동작: 첫 인자로 서브커맨드 분기.
    """
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "ingest":
        asyncio.run(_ingest())
    elif cmd == "ask":
        if len(sys.argv) < 3:
            sys.exit("사용법: python -m rag_agent.cli ask \"질문\"")
        asyncio.run(_ask(" ".join(sys.argv[2:])))
    elif cmd == "chat":
        asyncio.run(_chat())
    elif cmd == "eval":
        asyncio.run(_eval())
    else:
        print(f"알 수 없는 명령: {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
