"""지식베이스 — MCP 서버와 로컬 검색이 공유하는 데이터 계층.

data/knowledge/*.md를 읽어 키워드 매칭으로 검색한다.
임베딩 없이 간단한 TF 점수로 동작 — MCP 서버 내부 구현체.
"""

from __future__ import annotations

import re
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    """입력: 텍스트 / 출력: 소문자 토큰 리스트.

    동작: 공백+구두점 분리. 2글자 미만 토큰 제거.
    """
    return [t for t in re.split(r"[\s.,!?;:'\"()\[\]{}]+", text.lower()) if len(t) >= 2]


def search_knowledge_base(query: str, knowledge_dir: str | Path, top_k: int = 3) -> list[dict]:
    """입력: 쿼리, 지식베이스 디렉터리, 반환 개수 / 출력: [{title, excerpt, score}].

    동작: 각 .md 파일을 읽어 쿼리 토큰과의 매칭 점수를 계산한다.
    점수 = 매칭 토큰 수 / 쿼리 토큰 수 + 구문 일치 보너스.
    임베딩 없는 단순 검색이지만 MCP 도구로는 충분하다.
    """
    kb_dir = Path(knowledge_dir)
    if not kb_dir.exists():
        return []
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return []

    results = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.split("\n", 1)[0].lstrip("# ").strip() or path.stem
        doc_tokens = set(_tokenize(text))
        matched = q_tokens & doc_tokens
        if not matched:
            continue
        score = len(matched) / len(q_tokens)
        # 제목에 매칭되면 보너스
        title_tokens = set(_tokenize(title))
        if q_tokens & title_tokens:
            score += 0.3
        # 발췌: 첫 매칭 토큰 주변 200자
        excerpt = _excerpt_around(text, matched, size=200)
        results.append({"title": title, "file": path.name, "excerpt": excerpt, "score": round(score, 3)})

    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]


def _excerpt_around(text: str, tokens: set[str], size: int = 200) -> str:
    """입력: 텍스트, 매칭 토큰 집합, 길이 / 출력: 발췌 문자열.

    동작: 첫 매칭 토큰 위치 앞뒤로 size만큼 자른다. 없으면 앞부분.
    """
    lower = text.lower()
    pos = -1
    for tok in tokens:
        p = lower.find(tok)
        if p != -1 and (pos == -1 or p < pos):
            pos = p
    if pos == -1:
        return text[:size]
    start = max(0, pos - size // 2)
    return text[start : start + size].strip()
