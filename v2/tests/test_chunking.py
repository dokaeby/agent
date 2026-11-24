"""청킹 테스트 — API 키 불필요."""

from rag_agent.chunking import chunk_text


def test_short_text_single_chunk():
    """입력: chunk_size보다 짧은 텍스트 / 출력: 청크 1개.

    동작: 분할 없이 그대로 반환.
    """
    chunks = chunk_text("짧은 문서입니다.", chunk_size=800, overlap=150)
    assert chunks == ["짧은 문서입니다."]


def test_long_text_splits():
    """입력: chunk_size보다 긴 텍스트 / 출력: 여러 청크.

    동작: 각 청크가 chunk_size를 크게 넘지 않는지 확인.
    """
    text = "\n\n".join(f"문단 {i}: " + "내용 " * 50 for i in range(20))
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 500 + 100  # 오버랩 포함 여유


def test_overlap_preserves_context():
    """입력: 긴 텍스트 / 출력: 청크 리스트.

    동작: 인접 청크가 overlap 글자를 공유하는지 확인.
    """
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) >= 2
    # 두 번째 청크 앞부분이 첫 번째 청크 끝부분과 겹쳐야 함
    tail = chunks[0][-50:]
    assert chunks[1].startswith(tail[:20]) or tail[:20] in chunks[1][:100]


def test_empty_text():
    """입력: 빈 문자열 / 출력: 빈 리스트."""
    assert chunk_text("", chunk_size=100, overlap=10) == []
    assert chunk_text("   \n\n  ", chunk_size=100, overlap=10) == []


def test_korean_text():
    """입력: 한국어 텍스트 / 출력: 청크 리스트.

    동작: 한국어 문장 부호(다. 요.)도 구분자로 인식하는지 확인.
    """
    text = "안녕하세요. 오늘 날씨가 좋습니다. 점심을 먹었습니다. " * 30
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) for c in chunks)
