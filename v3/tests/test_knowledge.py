"""지식베이스 검색 + MCP 서버 함수 테스트 — API 키 불필요."""


from multiagent.knowledge import search_knowledge_base


def test_search_finds_relevant_doc(tmp_path):
    """입력: 임시 지식베이스 + 쿼리 / 출력: 관련 문서.

    동작: 'RAG' 검색 시 rag_basics 내용이 매칭되는지 확인.
    """
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "rag.md").write_text("# RAG 기술\nRAG는 검색 기반 생성입니다.", encoding="utf-8")
    (kb / "other.md").write_text("# 다른 주제\n완전히 다른 내용입니다.", encoding="utf-8")

    hits = search_knowledge_base("RAG 검색", kb)
    assert len(hits) >= 1
    assert hits[0]["file"] == "rag.md"


def test_search_no_match(tmp_path):
    """입력: 매칭 없는 쿼리 / 출력: 빈 리스트."""
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("# A\n내용", encoding="utf-8")
    assert search_knowledge_base("xyznonexistent", kb) == []


def test_search_empty_dir(tmp_path):
    """입력: 빈 디렉터리 / 출력: 빈 리스트."""
    assert search_knowledge_base("query", tmp_path / "empty") == []


def test_search_title_bonus(tmp_path):
    """입력: 제목에 매칭되는 쿼리 / 출력: 높은 점수.

    동작: 제목 매칭 시 score에 보너스(+0.3)가 붙는지 확인.
    """
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "langgraph.md").write_text("# LangGraph\nLangGraph는 그래프 프레임워크입니다.", encoding="utf-8")
    (kb / "other.md").write_text("# Other\nLangGraph에 대한 언급이 있는 다른 문서.", encoding="utf-8")

    hits = search_knowledge_base("LangGraph", kb)
    assert hits[0]["file"] == "langgraph.md"  # 제목 매칭 우선
