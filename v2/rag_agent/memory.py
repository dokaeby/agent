"""메모리 계층 — 단기(대화 요약)와 장기(사용자 정보)를 분리.

- 단기: 최근 대화가 길어지면 LLM으로 요약해 state에 저장.
  체크포인터 덕분에 세션(thread_id)을 다시 열어도 요약이 유지된다.
- 장기: 사용자가 말한 사실(이름, 선호 등)을 sqlite에 영구 저장.
  세션이 바뀌어도 같은 사용자 프로필을 재사용한다.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from .llm import LLMClient

# 단기 메모리: 이 메시지 수를 넘으면 요약 대상으로 본다
SUMMARIZE_THRESHOLD = 8


class LongTermMemory:
    """사용자 사실 영구 저장소 — sqlite3 기반.

    입력: db_path(sqlite 파일 경로)
    출력: remember()/recall_all()/forget()
    동작: user_facts 테이블(key-value+created_at). 프로세스 재시작 후에도 유지.
    """

    def __init__(self, db_path: str | Path) -> None:
        """입력: db_path / 출력: 인스턴스.

        동작: 디렉터리 생성 + 테이블이 없으면 생성.
        """
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS user_facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self._conn.commit()

    def remember(self, key: str, value: str) -> None:
        """입력: 사실의 key/value / 출력: 없음.

        동작: INSERT OR REPLACE — 같은 key는 최신 값으로 갱신.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO user_facts (key, value, created_at) VALUES (?,?,?)",
            (key, value, datetime.now().isoformat()),
        )
        self._conn.commit()

    def recall_all(self) -> dict[str, str]:
        """입력: 없음 / 출력: {key: value} dict.

        동작: 저장된 모든 사실을 반환. 프롬프트에 '사용자 정보'로 주입용.
        """
        rows = self._conn.execute("SELECT key, value FROM user_facts").fetchall()
        return {k: v for k, v in rows}

    def forget(self, key: str) -> bool:
        """입력: 삭제할 key / 출력: 삭제됐으면 True.

        동작: rowcount>0이면 True. 없으면 False.
        """
        cur = self._conn.execute("DELETE FROM user_facts WHERE key=?", (key,))
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        """입력: 없음 / 출력: 없음. sqlite 연결을 닫는다."""
        self._conn.close()


async def summarize_conversation(llm: LLMClient, messages: list[dict[str, str]]) -> str:
    """입력: LLMClient, 대화 메시지 리스트 / 출력: 요약 문자열.

    동작: 최근 대화를 3-5문장 요약으로 압축한다.
    사용자의 질문 의도, 제공된 정보, 미해결 사항을 중심으로 요약하도록 지시.
    요약은 다음 턴의 시스템 컨텍스트로 주입되어 장기 의존 없이 맥락을 유지한다.
    """
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages[-SUMMARIZE_THRESHOLD:])
    prompt = (
        "다음 대화를 3-5문장으로 요약하라. 사용자의 의도, 핵심 정보, "
        "아직 해결되지 않은 질문을 포함하라.\n\n" + convo
    )
    return await llm.chat(
        [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=300
    )


def extract_facts_prompt(user_message: str, assistant_reply: str) -> str:
    """입력: 사용자 메시지, 어시스턴트 답변 / 출력: 프롬프트 문자열.

    동작: 대화에서 '기억할 가치가 있는 사용자 사실'을 JSON으로 추출하는
    프롬프트를 만든다. 호출자가 chat_json으로 파싱해 LongTermMemory에 저장.
    """
    return (
        "다음 대화에서 사용자의 개인 정보/선호/사실 중 기억할 가치가 있는 것을 "
        "JSON 배열로 추출하라. 없으면 빈 배열.\n"
        '형식: [{"key": "user_name", "value": "김철수"}]\n\n'
        f"사용자: {user_message}\n어시스턴트: {assistant_reply}"
    )
