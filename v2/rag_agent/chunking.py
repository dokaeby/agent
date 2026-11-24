"""텍스트 청킹 — 재귀적 분할기.

전략: 문단(\\n\\n) → 줄(\\n) → 문장(.!?) → 강제 문자 절단 순으로
chunk_size 이하가 될 때까지 재귀 분할한다. overlap으로 경계 문맥을 보존.
"""

from __future__ import annotations

import re

# 큰 단위 → 작은 단위 순서의 구분자 계층
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "다. ", "요. ", ", ", " ", ""]
_SENTENCE_END = re.compile(r"(?<=[.!?다요])\s+")


def _split_by_sep(text: str, sep: str) -> list[str]:
    """입력: text, 구분자 / 출력: 분할된 조각 리스트(공백 정리, 빈 조각 제거).

    동작: sep으로 split 후 strip하고 빈 문자열을 버린다.
    """
    if not sep:
        return [text]
    return [p.strip() for p in text.split(sep) if p.strip()]


def _merge_small(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """입력: 조각 리스트, 크기/오버랩 / 출력: 병합된 청크 리스트.

    동작: 조각을 순서대로 이어 붙이되 chunk_size를 넘으면 새 청크 시작.
    새 청크 시작 시 이전 청크 끝 overlap 글자를 앞에 붙여 문맥을 잇는다.
    """
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            # 오버랩: 이전 청크의 끝 부분을 다음 청크 앞에 붙인다
            current = (current[-overlap:] + "\n" + piece)[: chunk_size + overlap]
        else:
            # 조각 자체가 chunk_size보다 큼 → 강제 문자 절단
            current = piece[:chunk_size]
    if current.strip():
        chunks.append(current)
    return chunks


def _recursive_split(text: str, chunk_size: int, overlap: int, level: int = 0) -> list[str]:
    """입력: text, chunk_size, overlap, 재귀 깊이 / 출력: 청크 리스트.

    동작: _SEPARATORS[level]로 나눠 각 조각이 chunk_size 이하면
    _merge_small로 병합, 크면 다음 더 작은 구분자로 재귀.
    모든 구분자를 써도 크면 문자 단위로 자른다.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    if level >= len(_SEPARATORS) - 1:
        # 최후 수단: 문자 경계 절단 + 오버랩
        return [
            text[i : i + chunk_size]
            for i in range(0, len(text), chunk_size - overlap)
        ]

    sep = _SEPARATORS[level]
    pieces = _split_by_sep(text, sep)
    # 더 작은 구분자가 의미 없을 정도로 조각이 작으면 여기서 병합
    if all(len(p) <= chunk_size for p in pieces):
        return _merge_small(pieces, chunk_size, overlap)
    chunks: list[str] = []
    for piece in pieces:
        chunks.extend(_recursive_split(piece, chunk_size, overlap, level + 1))
    return chunks


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """입력: 원문 텍스트, 청크 크기/오버랩 / 출력: 청크 문자열 리스트.

    동작: 공백 정규화 후 재귀 분할. 각 청크는 chunk_size를 크게 넘지 않고,
    앞뒤 청크와 overlap 글자를 공유해 검색 시 문맥이 끊기지 않게 한다.
    """
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    return _recursive_split(text, chunk_size, overlap)
