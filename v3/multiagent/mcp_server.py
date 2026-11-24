"""MCP 서버 — 지식베이스 검색/시간/리포트 저장 도구를 표준 프로토콜로 노출.

실행: python -m multiagent.mcp_server  (stdio transport)

FastMCP가 tool() 데코레이터로 함수를 MCP tool로 자동 등록한다.
클라이언트는 tools/list, tools/call 표준 메서드로 접근한다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .config import get_settings
from .knowledge import search_knowledge_base

mcp = FastMCP("knowledge-base")


def _reports_dir() -> Path:
    """입력: 없음 / 출력: reports 디렉터리 Path.

    동작: 설정에서 경로를 읽어 디렉터리를 보장한다.
    """
    d = get_settings().reports_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


@mcp.tool()
def search_docs(query: str) -> str:
    """지식베이스에서 쿼리와 관련된 문서를 검색한다.

    입력: query(검색어)
    출력: 관련 문서 목록 (제목/파일/발췌/점수)
    동작: data/knowledge/*.md를 키워드 매칭으로 검색.
    """
    settings = get_settings()
    hits = search_knowledge_base(query, settings.knowledge_dir)
    if not hits:
        return "관련 문서를 찾을 수 없습니다."
    lines = []
    for h in hits:
        lines.append(f"[{h['score']}] {h['title']} ({h['file']})\n{h['excerpt']}")
    return "\n\n---\n\n".join(lines)


@mcp.tool()
def get_current_time() -> str:
    """입력: 없음 / 출력: 현재 시각 ISO 문자열.

    동작: 서버 로컬 시간 기준 ISO 8601 형식.
    """
    return datetime.now().isoformat()


@mcp.tool()
def save_report(title: str, content: str) -> str:
    """리포트를 파일로 저장한다 (write 권한 필요).

    입력: title(파일명용 제목), content(마크다운 본문)
    출력: 저장된 파일 경로
    동작: reports/{slug}.md로 저장. slug는 파일명 안전 문자만.
    """
    import re

    slug = re.sub(r"[^\w가-힣-]+", "_", title).strip("_")[:50] or "report"
    path = _reports_dir() / f"{slug}.md"
    path.write_text(f"# {title}\n\n{content}", encoding="utf-8")
    return f"saved: {path}"


@mcp.tool()
def list_reports() -> str:
    """입력: 없음 / 출력: 저장된 리포트 목록.

    동작: reports/ 디렉터리의 .md 파일 목록을 반환.
    """
    files = sorted(_reports_dir().glob("*.md"))
    if not files:
        return "저장된 리포트가 없습니다."
    return "\n".join(f"- {f.name} ({f.stat().st_size} bytes)" for f in files)


if __name__ == "__main__":
    mcp.run(transport="stdio")
