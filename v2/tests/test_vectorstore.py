"""VectorStore(ChromaDB) 테스트 — 로컬 파일 기반, API 키 불필요."""

import pytest

chromadb = pytest.importorskip("chromadb")

from rag_agent.vectorstore import VectorStore  # noqa: E402


@pytest.fixture
def vs(tmp_path):
    """입력: pytest tmp_path / 출력: VectorStore.

    동작: 임시 디렉터리에 ChromaDB를 만든다 — 실제 chroma_db를 건드리지 않음.
    """
    return VectorStore(tmp_path / "chroma")


def test_add_and_query(vs):
    """입력: 2개 청크의 더미 벡터 / 출력: 없음(단언).

    동작: add 후 query로 가장 유사한 벡터가 검색되는지 확인.
    """
    ids = ["a", "b"]
    docs = ["연차는 11일", "비밀번호 12자"]
    # 더미 벡터: a는 [1,0], b는 [0,1]
    embs = [[1.0, 0.0], [0.0, 1.0]]
    metas = [{"source": "vacation.md"}, {"source": "security.md"}]
    vs.add(ids, docs, embs, metas)
    assert vs.count() == 2

    hits = vs.query([0.9, 0.1], top_k=2)
    assert hits[0]["id"] == "a"
    assert hits[0]["metadata"]["source"] == "vacation.md"


def test_upsert_same_id(vs):
    """입력: 같은 id로 2번 add / 출력: 없음(단언).

    동작: upsert라 중복 없이 갱신되는지 확인.
    """
    vs.add(["x"], ["old"], [[1.0, 0.0]], [{"source": "a.md"}])
    vs.add(["x"], ["new"], [[1.0, 0.0]], [{"source": "b.md"}])
    assert vs.count() == 1
    hits = vs.query([1.0, 0.0], top_k=1)
    assert hits[0]["document"] == "new"


def test_delete_all(vs):
    """입력: 데이터가 있는 vs / 출력: 없음(단언).

    동작: delete_all 후 count=0.
    """
    vs.add(["x"], ["doc"], [[1.0, 0.0]], [{}])
    vs.delete_all()
    assert vs.count() == 0
