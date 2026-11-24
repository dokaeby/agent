"""BM25 인덱스 테스트 — API 키 불필요."""


from rag_agent.bm25 import BM25Index


def _chunks():
    """입력: 없음 / 출력: 테스트용 청크 3개.

    동작: 휴가/보안/제품 주제의 청크를 만든다.
    """
    return [
        {"id": "c1", "text": "연차 유급휴가는 입사 1년 후 11일입니다", "metadata": {"source": "vacation.md"}},
        {"id": "c2", "text": "비밀번호는 12자 이상이어야 합니다", "metadata": {"source": "security.md"}},
        {"id": "c3", "text": "SmartMonitor Pro는 월 29달러입니다", "metadata": {"source": "product.md"}},
    ]


def test_build_and_search():
    """입력: 3개 청크 / 출력: 없음(단언).

    동작: '연차' 검색 시 c1이 최상위에 오는지 확인.
    """
    idx = BM25Index()
    idx.build(_chunks())
    hits = idx.search("연차 휴가", 3)
    assert hits[0]["id"] == "c1"


def test_search_empty_index():
    """입력: 빌드 안 된 인덱스 / 출력: 빈 리스트."""
    idx = BM25Index()
    assert idx.search("anything", 5) == []


def test_save_load(tmp_path):
    """입력: 임시 경로 / 출력: 없음(단언).

    동작: pickle 저장 후 load로 복원해 동일 검색 결과가 나오는지 확인.
    """
    idx = BM25Index()
    idx.build(_chunks())
    path = tmp_path / "bm25.pkl"
    idx.save(path)
    loaded = BM25Index.load(path)
    # 한국어는 어절+조사가 한 토큰 — 실제 토큰 그대로 검색
    hits = loaded.search("비밀번호는 12자", 3)
    assert hits[0]["id"] == "c2"


def test_load_missing_file(tmp_path):
    """입력: 없는 경로 / 출력: 빈 인덱스(예외 없음)."""
    idx = BM25Index.load(tmp_path / "nope.pkl")
    assert idx.search("x", 1) == []
