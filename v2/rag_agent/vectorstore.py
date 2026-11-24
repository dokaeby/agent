"""벡터 저장소 — ChromaDB persistent 클라이언트 래퍼.

로컬 파일(chroma_dir) 기반이라 서버 없이 동작한다.
pgvector/Qdrant로 바꾸려면 이 클래스의 add()/query() 인터페이스만 유지하면 된다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings

COLLECTION = "rag_chunks"


class VectorStore:
    """ChromaDB 컬렉션 래퍼 — 임베딩 벡터의 저장/검색.

    입력: persist_dir(ChromaDB 데이터 디렉터리)
    출력: add()/query()/count() 메서드
    동작: collection은 get_or_create — ingest를 여러 번 실행해도 누적된다.
    """

    def __init__(self, persist_dir: str | Path) -> None:
        """입력: persist_dir / 출력: 인스턴스.

        동작: 디렉터리를 만들고 PersistentClient + 컬렉션을 준비한다.
        anonymized_telemetry를 꺼서 외부 전송을 막는다.
        """
        self._dir = Path(persist_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._client = PersistentClient(
            path=str(self._dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},  # 코사인 유사도
        )

    def add(self, ids: list[str], documents: list[str], embeddings: list[list[float]],
            metadatas: list[dict[str, Any]]) -> None:
        """입력: id/원문/벡터/메타데이터 4개 병렬 리스트(같은 길이).
        출력: 없음.

        동작: upsert — 같은 id면 덮어쓴다. 재-ingest 시 중복 없이 갱신 가능.
        """
        # ChromaDB는 빈 metadata dict를 거부하므로 기본값을 넣는다
        metas = [m if m else {"source": "unknown"} for m in metadatas]
        self._collection.upsert(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metas
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """입력: 쿼리 벡터, 반환 개수 / 출력: [{id, document, metadata, score}].

        동작: 코사인 유사도 검색. score는 Chroma가 반환하는
        distance(유사도) — 1.0에 가까울수록 유사(코사인 distance = 1 - similarity).
        """
        resp = self._collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        results = []
        for i, doc_id in enumerate(resp["ids"][0]):
            results.append(
                {
                    "id": doc_id,
                    "document": resp["documents"][0][i],
                    "metadata": resp["metadatas"][0][i] or {},
                    "score": float(resp["distances"][0][i]),
                }
            )
        return results

    def count(self) -> int:
        """입력: 없음 / 출력: 컬렉션 내 청크 수."""
        return self._collection.count()

    def delete_all(self) -> None:
        """입력: 없음 / 출력: 없음.

        동작: 컬렉션을 삭제하고 재생성 — ingest 초기화용.
        """
        self._client.delete_collection(COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )
