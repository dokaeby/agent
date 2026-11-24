"""멀티에이전트 CLI.

사용법:
  python -m multiagent.cli run "리서치 주제"
  python -m multiagent.cli serve   # FastAPI 서버
"""

from __future__ import annotations

import asyncio
import sys

from .config import get_settings
from .orchestrator import Orchestrator


async def _run(topic: str) -> None:
    """입력: 주제 / 출력: 콘솔 출력.

    동작: Orchestrator로 파이프라인 실행 후 결과를 출력한다.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        sys.exit("OPENAI_API_KEY가 없습니다. .env.example을 참고해 .env를 만드세요.")

    orch = Orchestrator(settings)
    result = await orch.run(topic)

    print(f"\n{'='*60}")
    print(f"주제: {result.topic}")
    print(f"소요 시간: {result.elapsed_sec}s")
    print(f"리뷰 점수: {result.review_score:.2f} ({result.review_rounds}회)")
    print(f"예산: {result.budget}")
    print("\n[계획]")
    for i, p in enumerate(result.plan, 1):
        print(f"  {i}. {p}")
    print(f"\n[최종 리포트]\n{result.final_report}")
    print(f"\n{'='*60}")


def main() -> None:
    """입력: sys.argv / 출력: 없음."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "run":
        if len(sys.argv) < 3:
            sys.exit("사용법: python -m multiagent.cli run \"주제\"")
        asyncio.run(_run(" ".join(sys.argv[2:])))
    elif cmd == "serve":
        import uvicorn
        uvicorn.run("multiagent.service:app", host="0.0.0.0", port=8000, reload=False)
    else:
        print(f"알 수 없는 명령: {cmd}\n{__doc__}")


if __name__ == "__main__":
    main()
