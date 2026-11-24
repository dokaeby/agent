"""관측(Observability) — JSONL 트레이서 + LangSmith 연동 안내.

두 계층:
1. 로컬 Tracer: 각 노드의 시작/끝/소요시간/토큰을 JSONL로 기록.
   서버 없이 ./traces/에 파일로 남아 회귀 분석에 바로 쓸 수 있다.
2. LangSmith: LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY가 설정되면
   langgraph가 자동으로 LangSmith에 트레이스를 보낸다(코드 불필요).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Tracer:
    """실행 트레이스 JSONL 기록기.

    입력: traces_dir(출력 디렉터리)
    출력: start_run()/log_node()/end_run()
    동작: run_id별로 노드 이벤트를 traces/{date}.jsonl에 append한다.
    """

    def __init__(self, traces_dir: str | Path) -> None:
        """입력: traces_dir / 출력: 인스턴스.

        동작: 디렉터리 생성. 파일은 날짜별로 나뉜다(하루 1파일).
        """
        self._dir = Path(traces_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        """입력: 없음 / 출력: 오늘 날짜의 JSONL 경로."""
        return self._dir / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"

    def start_run(self, question: str, thread_id: str) -> str:
        """입력: 사용자 질문, 세션 thread_id / 출력: run_id(UUID).

        동작: run_started 이벤트를 기록하고 run_id를 반환한다.
        이후 log_node/end_run이 이 run_id로 연결된다.
        """
        run_id = uuid.uuid4().hex[:12]
        self._write(
            {
                "event": "run_started",
                "run_id": run_id,
                "thread_id": thread_id,
                "question": question[:200],
                "ts": datetime.now(UTC).isoformat(),
            }
        )
        return run_id

    def log_node(self, run_id: str, node: str, duration_ms: float, **extra: Any) -> None:
        """입력: run_id, 노드명, 소요시간(ms), 추가 필드 / 출력: 없음.

        동작: node_finished 이벤트 기록. extra로 토큰/결과 수 등을 첨부 가능.
        """
        self._write(
            {
                "event": "node_finished",
                "run_id": run_id,
                "node": node,
                "duration_ms": round(duration_ms, 1),
                "ts": datetime.now(UTC).isoformat(),
                **extra,
            }
        )

    def end_run(self, run_id: str, answer: str, tokens: int, **extra: Any) -> None:
        """입력: run_id, 최종 답변, 토큰 수 / 출력: 없음.

        동작: run_finished 이벤트 기록. 답변 앞 300자와 메타데이터를 남긴다.
        """
        self._write(
            {
                "event": "run_finished",
                "run_id": run_id,
                "answer_preview": answer[:300],
                "tokens_used": tokens,
                "ts": datetime.now(UTC).isoformat(),
                **extra,
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        """입력: 기록할 dict / 출력: 없음.

        동작: JSONL 한 줄 append. 쓰기 실패는 로깅만 하고 삼킨다
        (관측 실패가 에이전트 실행을 막으면 안 된다).
        """
        try:
            with open(self._path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("트레이스 쓰기 실패: %s", exc)


class NodeTimer:
    """노드 실행 시간 측정 컨텍스트 매니저.

    입력: tracer, run_id, 노드명
    출력: with 블록 안에서 소요시간 측정
    동작: __enter__에서 시작 시각, __exit__에서 tracer.log_node 호출.
    """

    def __init__(self, tracer: Tracer, run_id: str, node: str) -> None:
        self._tracer = tracer
        self._run_id = run_id
        self._node = node
        self._start = 0.0

    def __enter__(self) -> NodeTimer:
        """입력: 없음 / 출력: 자기 자신.

        동작: time.monotonic()으로 시작 시각 기록.
        """
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        """입력: 예외 정보(무시) / 출력: 없음.

        동작: 경과 시간을 ms로 변환해 tracer.log_node에 기록한다.
        """
        elapsed_ms = (time.monotonic() - self._start) * 1000
        self._tracer.log_node(self._run_id, self._node, elapsed_ms)
