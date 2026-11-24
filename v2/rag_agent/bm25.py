"""BM25 키워드 검색 인덱스 — 하이브리드 검색의 sparse 쪽.

rank_bm25로 청크 텍스트의 키워드 유사도를 계산한다.
임베딩이 놓치는 정확한 용어 매칭(고유명사, 버전번호 등)을 보완한다.
인덱스는 pickle로 디스크에 캐시해 재시작 시 재구축 비용을 피한다.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25Okapi 래퍼 — 청크 텍스트의 토큰화 인덱스.

    입력: 청크 dict 리스트({id, text, metadata})
    출력: build()/search()/save()/load()
    동작: 인메모리 인덱스 + pickle 영속화.
    """

    def __init__(self) -> None:
        """입력: 없음 / 출력: 빈 인덱스.

        동작: _bm25=None, _docs=[], _ids=[] — build() 전까지 검색 불가.
        """
        self._bm25: BM25Okapi | None = None
        self._docs: list[str] = []
        self._metas: list[dict[str, Any]] = []
        self._ids: list[str] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """입력: 텍스트 / 출력: 소문자 알파뉴메릭 토큰 리스트.

        동작: 간단한 공백+구두점 분리. 한국어는 어절 단위로 토큰화된다
        (형태소 분석은 하지 않음 — 학습용 단순화).
        """
        return [t for t in text.lower().replace("\n", " ").split() if t.strip(".,!?;:'\"()[]{}")]

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """입력: [{"id","text","metadata"}...] / 출력: 없음.

        동작: 각 청크를 토큰화해 BM25Okapi를 만든다. 빈 청크는 스킵.
        """
        self._ids, self._docs, self._metas = [], [], []
        tokenized = []
        for c in chunks:
            tokens = self._tokenize(c["text"])
            if not tokens:
                continue
            tokenized.append(tokens)
            self._ids.append(c["id"])
            self._docs.append(c["text"])
            self._metas.append(c.get("metadata", {}))
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """입력: 쿼리 문자열, 반환 개수 / 출력: [{id, text, metadata, score}].

        동작: 쿼리를 토큰화해 BM25 점수로 정렬. score는 BM25 원점수
        (높을수록 관련). 인덱스가 없으면 빈 리스트.
        """
        if self._bm25 is None:
            return []
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"id": self._ids[i], "text": self._docs[i], "metadata": self._metas[i], "score": float(scores[i])}
            for i in ranked
            if scores[i] > 0
        ]

    def save(self, path: str | Path) -> None:
        """입력: 저장 경로 / 출력: 없음.

        동작: _bm25 + 문서/메타/id를 pickle로 저장. BM25Okapi는
        pickle 가능하다.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"bm25": self._bm25, "docs": self._docs, "metas": self._metas, "ids": self._ids}, f
            )

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        """입력: 저장 경로 / 출력: 복원된 BM25Index.

        동작: 파일이 없으면 빈 인덱스를 반환(빌드 전 상태).
        """
        path = Path(path)
        idx = cls()
        if not path.exists():
            return idx
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx._bm25 = data["bm25"]
        idx._docs = data["docs"]
        idx._metas = data["metas"]
        idx._ids = data["ids"]
        return idx
