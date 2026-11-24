"""문서 인제스천 — 파일 로드 → 청킹 → 임베딩 → 벡터DB+BM25 저장.

흐름:
  data/docs/*.md 로드 → chunk_text()로 분할 → LLMClient.embed()로 벡터화
  → VectorStore.add() + BM25Index.build() + .save()
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from .bm25 import BM25Index
from .chunking import chunk_text
from .config import Settings, get_settings
from .llm import LLMClient
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


def _chunk_id(source: str, index: int, text: str) -> str:
    """입력: 출처 파일명, 청크 인덱스, 텍스트 / 출력: 결정적 ID.

    동작: source+index+텍스트 해시로 고유 ID 생성 — 같은 문서를
    재-ingest해도 같은 ID가 나와 upsert로 안전하게 갱신된다.
    """
    h = hashlib.sha256(f"{source}:{index}:{text[:100]}".encode()).hexdigest()[:16]
    return f"{Path(source).stem}_{index}_{h}"


async def ingest_directory(
    docs_dir: str | Path,
    vectorstore: VectorStore,
    bm25: BM25Index,
    llm: LLMClient,
    settings: Settings | None = None,
) -> dict[str, int]:
    """입력: 문서 디렉터리, 저장소들, LLM, 설정 / 출력: 통계 dict.

    동작:
    1. docs_dir의 .md/.txt 파일을 모두 읽는다.
    2. 각 파일을 chunk_text로 분할한다.
    3. 모든 청크를 embed()로 벡터화한다(배치).
    4. VectorStore에 upsert + BM25Index 재빌드 + pickle 저장.
    반환: {"files": n, "chunks": n, "vectors": n}
    """
    settings = settings or get_settings()
    docs_dir = Path(docs_dir)
    files = sorted(docs_dir.glob("*.md")) + sorted(docs_dir.glob("*.txt"))
    if not files:
        logger.warning("문서가 없습니다: %s", docs_dir)
        return {"files": 0, "chunks": 0, "vectors": 0}

    all_chunks: list[dict[str, Any]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "id": _chunk_id(path.name, i, chunk),
                    "text": chunk,
                    "metadata": {"source": path.name, "chunk_index": i},
                }
            )
        logger.info("%s → %d 청크", path.name, len(chunks))

    if not all_chunks:
        return {"files": len(files), "chunks": 0, "vectors": 0}

    # 임베딩 배치 처리
    texts = [c["text"] for c in all_chunks]
    vectors = await llm.embed(texts)

    vectorstore.add(
        ids=[c["id"] for c in all_chunks],
        documents=texts,
        embeddings=vectors,
        metadatas=[c["metadata"] for c in all_chunks],
    )

    # BM25 인덱스 재빌드 (전체 청크 기준)
    bm25.build(all_chunks)
    bm25.save(settings.bm25_index_path)

    stats = {"files": len(files), "chunks": len(all_chunks), "vectors": len(vectors)}
    logger.info("ingest 완료: %s", stats)
    return stats
