"""하이브리드 검색 + 리랭킹 — dense·BM25를 RRF로 융합하고 LLM이 재정렬.

흐름:
  1. 쿼리 임베딩 → VectorStore top-K (dense)
  2. 쿼리 → BM25Index top-K (sparse/키워드)
  3. RRF(Reciprocal Rank Fusion)로 두 순위 목록을 융합 → 후보 top-N
  4. LLM 리랭커가 후보를 관련성 점수로 재정렬 → 최종 top-M
"""

from __future__ import annotations

import logging
from typing import Any

from .bm25 import BM25Index
from .llm import LLMClient
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF 상수 — 표준값 60 사용


class Retriever:
    """하이브리드 리트리버 — dense+BM25+RRF+LLM 리랭크.

    입력: VectorStore, BM25Index, LLMClient, 설정값들
    출력: retrieve(query) → 관련 청크 리스트
    동작: 각 단계 실패 시 가능한 범위에서 graceful degradation.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        bm25: BM25Index,
        llm: LLMClient,
        *,
        retrieve_top_k: int = 20,
        rerank_candidates: int = 10,
        final_top_k: int = 5,
        use_llm_rerank: bool = True,
    ) -> None:
        """입력: 저장소/인덱스/LLM + 검색 파라미터.
        출력: 인스턴스.

        동작: use_llm_rerank=False면 RRF 결과를 그대로 반환(비용/지연 절약).
        """
        self._vs = vectorstore
        self._bm25 = bm25
        self._llm = llm
        self.retrieve_top_k = retrieve_top_k
        self.rerank_candidates = rerank_candidates
        self.final_top_k = final_top_k
        self.use_llm_rerank = use_llm_rerank

    async def _dense_search(self, query: str) -> list[dict[str, Any]]:
        """입력: 쿼리 / 출력: [{id,document,metadata,score,dense_rank}].

        동작: 쿼리 임베딩 후 코사인 검색. score는 distance(낮을수록 유사).
        """
        vectors = await self._llm.embed([query])
        hits = self._vs.query(vectors[0], self.retrieve_top_k)
        for rank, h in enumerate(hits):
            h["dense_rank"] = rank
        return hits

    def _bm25_search(self, query: str) -> list[dict[str, Any]]:
        """입력: 쿼리 / 출력: [{id,text,metadata,score,bm25_rank}].

        동작: 키워드 검색. score는 BM25 원점수(높을수록 관련).
        """
        hits = self._bm25.search(query, self.retrieve_top_k)
        for rank, h in enumerate(hits):
            h["bm25_rank"] = rank
        return hits

    @staticmethod
    def _rrf_fuse(dense: list[dict], sparse: list[dict], top_n: int) -> list[dict]:
        """입력: 두 검색 결과 리스트, 반환 개수 / 출력: 융합된 후보 리스트.

        동작: RRF = Σ 1/(K + rank). dense는 document, sparse는 text 키를
        document로 통일하고, 양쪽 점수를 합산해 정렬한다.
        """
        fused: dict[str, dict] = {}
        for h in dense:
            cid = h["id"]
            entry = {"id": cid, "document": h["document"], "metadata": h["metadata"], "rrf": 0.0}
            fused.setdefault(cid, entry)
            fused[cid]["rrf"] += 1.0 / (RRF_K + h["dense_rank"])
            fused[cid]["dense_score"] = h["score"]
        for h in sparse:
            cid = h["id"]
            entry = {"id": cid, "document": h["text"], "metadata": h["metadata"], "rrf": 0.0}
            fused.setdefault(cid, entry)
            fused[cid]["rrf"] += 1.0 / (RRF_K + h["bm25_rank"])
            fused[cid]["bm25_score"] = h["score"]
        ranked = sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)[:top_n]
        return ranked

    async def _llm_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """입력: 쿼리, RRF 후보 리스트 / 출력: 재정렬된 리스트.

        동작: 후보 본문을 LLM에 보여주고 각 항목의 관련성을 0-10점으로
        매기게 한다. 점수 순으로 정렬해 상위 final_top_k만 반환.
        LLM 호출 실패 시 RRF 순서를 그대로 반환(fail-safe).
        """
        if not candidates:
            return []
        lines = []
        for i, c in enumerate(candidates):
            text = c["document"][:400].replace("\n", " ")
            lines.append(f"[{i}] {text}")
        prompt = (
            "다음은 질문에 대해 검색된 문서 조각들이다. 각 조각이 질문에 "
            "얼마나 관련 있는지 0-10점으로 평가하라.\n\n"
            f"질문: {query}\n\n" + "\n".join(lines) +
            "\n\nJSON으로만 답하라: {\"scores\": [{\"index\": 0, \"score\": 8}, ...]}"
        )
        try:
            data = await self._llm.chat_json(
                [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=500
            )
            score_map = {int(s["index"]): float(s["score"]) for s in data.get("scores", [])}
            for i, c in enumerate(candidates):
                c["rerank_score"] = score_map.get(i, 0.0)
            return sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        except Exception as exc:  # noqa: BLE001 - 리랭크 실패는 RRF 순서로 폴백
            logger.warning("LLM 리랭크 실패, RRF 순서 사용: %s", exc)
            return candidates

    async def retrieve(self, query: str) -> list[dict[str, Any]]:
        """입력: 자연어 쿼리 / 출력: 최종 관련 청크 리스트(final_top_k개).

        동작: dense+BM25 병렬 검색 → RRF 융합 top rerank_candidates →
        LLM 리랭크(활성 시) → top final_top_k 반환.
        각 결과에 {id, document, metadata, rrf, rerank_score?}가 담긴다.
        """
        dense = await self._dense_search(query)
        sparse = self._bm25_search(query)
        fused = self._rrf_fuse(dense, sparse, self.rerank_candidates)
        if not fused:
            return []
        if self.use_llm_rerank:
            fused = await self._llm_rerank(query, fused)
        return fused[: self.final_top_k]
