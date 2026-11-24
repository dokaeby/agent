"""평가 실행기 — 데이터셋 기반 RAG 품질 측정.

지표:
1. retrieval_recall: 기대 문서(source)가 검색 결과에 포함됐는가 (0/1)
2. answer_score: LLM-as-judge가 답변 품질을 0-1로 채점
3. faithfulness: 답변이 검색 문서에 근거하는가 (간이 판정)

사용:
  python -m rag_agent.eval.run --dataset evals/dataset.jsonl
  python -m rag_agent.eval.run --dataset evals/dataset.jsonl --baseline evals/baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from ..bm25 import BM25Index
from ..config import get_settings
from ..graph import build_graph
from ..llm import LLMClient
from ..memory import LongTermMemory
from ..retrieval import Retriever
from ..vectorstore import VectorStore

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """당신은 RAG 시스템의 답변 품질 평가자다.

질문: {question}
기대 답변(참고): {expected}
실제 답변: {answer}
검색된 문서 출처: {sources}

실제 답변을 0.0~1.0으로 채점하라.
- 1.0: 기대 답변의 핵심 내용을 정확히 포함
- 0.5: 부분적으로만 관련/정확
- 0.0: 무관하거나 틀림, 또는 '모른다'고만 답함

JSON으로만 답하라: {{"score": 0.8, "reason": "한 줄 이유"}}"""


async def _judge(
    llm: LLMClient, question: str, expected: str, answer: str, sources: list[str]
) -> dict[str, Any]:
    """입력: LLM, 질문/기대답변/실제답변/출처 / 출력: {"score": float, "reason": str}.

    동작: LLM-as-judge로 답변 품질 채점. 파싱 실패 시 score=0.
    """
    prompt = JUDGE_PROMPT.format(
        question=question, expected=expected, answer=answer, sources=", ".join(sources)
    )
    try:
        data = await llm.chat_json([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=200)
        return {"score": float(data.get("score", 0.0)), "reason": data.get("reason", "")}
    except Exception as exc:  # noqa: BLE001
        logger.warning("judge 파싱 실패: %s", exc)
        return {"score": 0.0, "reason": f"judge error: {exc}"}


async def run_eval(
    dataset_path: str | Path,
    *,
    limit: int | None = None,
    use_hitl: bool = False,
) -> dict[str, Any]:
    """입력: 데이터셋 JSONL 경로, 제한/Hitl 여부 / 출력: 평가 리포트 dict.

    동작:
    1. JSONL을 읽어 각 항목을 그래프에 입력한다.
    2. retrieval_recall(기대 source 포함 여부)과 answer_score(judge)를 계산.
    3. 평균 점수와 항목별 상세를 반환한다.
    """
    settings = get_settings()
    llm = LLMClient(settings)
    vs = VectorStore(settings.chroma_dir)
    bm25 = BM25Index.load(settings.bm25_index_path)
    retriever = Retriever(vs, bm25, llm, retrieve_top_k=settings.retrieve_top_k,
                          rerank_candidates=settings.rerank_candidates, final_top_k=settings.final_top_k)
    longterm = LongTermMemory(settings.longterm_db)
    # eval은 매번 새 세션 — checkpointer 없이 실행해 상태 오염을 막는다
    compiled = build_graph(llm, retriever, longterm, use_hitl=use_hitl)

    raw_lines = Path(dataset_path).read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in raw_lines if line.strip()]
    if limit:
        items = items[:limit]

    results = []
    t0 = time.time()
    for i, item in enumerate(items):
        qid = item.get("id", str(i))
        question = item["question"]
        expected = item.get("expected_answer", "")
        expected_source = item.get("expected_source", "")

        state = {
            "messages": [{"role": "user", "content": question}],
            "question": question,
            "iteration": 0,
            "tokens_used": 0,
        }
        try:
            thread_id = f"eval-{qid}-{int(time.time())}"
            final = await compiled.ainvoke(state, config={"configurable": {"thread_id": thread_id}})
            answer = final.get("answer", "")
            sources = final.get("sources", [])
            retrieved = final.get("retrieved", [])

            # retrieval recall: 기대 source가 검색 결과에 있는가
            retrieved_sources = {c.get("metadata", {}).get("source", "") for c in retrieved}
            recall = 1.0 if (not expected_source or expected_source in retrieved_sources) else 0.0

            judge = await _judge(llm, question, expected, answer, sources)
            results.append({
                "id": qid, "question": question,
                "answer": answer[:300], "sources": sources,
                "retrieval_recall": recall, "answer_score": judge["score"],
                "judge_reason": judge["reason"],
            })
            logger.info("[%d/%d] %s — recall=%.0f score=%.2f", i+1, len(items), qid, recall, judge["score"])
        except Exception as exc:  # noqa: BLE001
            results.append({"id": qid, "question": question, "error": str(exc),
                            "retrieval_recall": 0.0, "answer_score": 0.0})
            logger.error("[%d/%d] %s 실패: %s", i+1, len(items), qid, exc)

    n = len(results)
    avg_recall = sum(r["retrieval_recall"] for r in results) / max(n, 1)
    avg_score = sum(r["answer_score"] for r in results) / max(n, 1)
    report = {
        "dataset": str(dataset_path), "items": n,
        "avg_retrieval_recall": round(avg_recall, 3),
        "avg_answer_score": round(avg_score, 3),
        "elapsed_sec": round(time.time() - t0, 1),
        "results": results,
    }
    await llm.aclose()
    return report


def compare_reports(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """입력: 이전 리포트, 현재 리포트 / 출력: 비교 결과.

    동작: 평균 recall/score의 차이와 항목별 점수 변화를 계산한다.
    프롬프트 변경 전후의 회귀 여부를 판단하는 데 사용.
    """
    def _by_id(report):
        return {r["id"]: r for r in report.get("results", [])}

    b, c = _by_id(baseline), _by_id(current)
    diffs = []
    for qid in sorted(set(b) & set(c)):
        delta = c[qid].get("answer_score", 0) - b[qid].get("answer_score", 0)
        if abs(delta) > 0.01:
            diffs.append({"id": qid, "baseline": b[qid].get("answer_score"),
                          "current": c[qid].get("answer_score"), "delta": round(delta, 3)})
    return {
        "avg_recall_delta": round(current["avg_retrieval_recall"] - baseline["avg_retrieval_recall"], 3),
        "avg_score_delta": round(current["avg_answer_score"] - baseline["avg_answer_score"], 3),
        "changed_items": diffs,
        "verdict": "improved" if current["avg_answer_score"] > baseline["avg_answer_score"] else
                   "regressed" if current["avg_answer_score"] < baseline["avg_answer_score"] else "same",
    }


async def _main() -> None:
    """입력: CLI 인자 / 출력: 콘솔 리포트.

    동작: --dataset으로 평가 실행, --baseline과 함께면 비교 리포트 출력.
    """
    parser = argparse.ArgumentParser(description="RAG 평가 실행기")
    parser.add_argument("--dataset", required=True, help="JSONL 데이터셋 경로")
    parser.add_argument("--limit", type=int, default=None, help="처음 N개만 평가")
    parser.add_argument("--baseline", help="비교할 이전 리포트 JSON")
    parser.add_argument("--output", help="결과 저장 경로 (JSON)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = await run_eval(args.dataset, limit=args.limit)

    print("\n=== 평가 결과 ===")
    print(f"항목 수:        {report['items']}")
    print(f"검색 재현율:    {report['avg_retrieval_recall']:.1%}")
    print(f"답변 평균 점수: {report['avg_answer_score']:.3f}")
    print(f"소요 시간:      {report['elapsed_sec']}s")

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        comp = compare_reports(baseline, report)
        print("\n=== baseline 대비 ===")
        print(f"재현율 변화: {comp['avg_recall_delta']:+.3f}")
        print(f"점수 변화:   {comp['avg_score_delta']:+.3f}  →  {comp['verdict']}")
        for d in comp["changed_items"][:10]:
            print(f"  {d['id']}: {d['baseline']} → {d['current']} ({d['delta']:+.2f})")

    out = args.output or f"evals/report_{int(time.time())}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n리포트 저장: {out}")


if __name__ == "__main__":
    asyncio.run(_main())
