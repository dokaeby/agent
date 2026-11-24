"""간단한 REPL 채팅 CLI.

사용법:
    python -m agent.cli            # 대화형 루프
    python -m agent.cli "질문"     # 1회 질문
"""

from __future__ import annotations

import asyncio
import sys

from .agent import Agent
from .config import get_settings


async def _chat(agent: Agent, message: str, history: list[dict]) -> list[dict]:
    """입력: agent, message(사용자 메시지), history(누적 대화).
    출력: 갱신된 history 리스트(user/assistant 2개 메시지 추가됨).

    동작: agent.run()을 호출해 결과를 사람이 읽는 형식으로 출력한다 —
    반복 횟수/총 토큰/비용, 사용된 도구명, 최종 답변.
    history에 user+assistant 메시지를 append해 다음 턴의 컨텍스트로 쓴다.
    """
    result = await agent.run(message, history=history)
    print(f"\n[{result.iterations}턴 | {result.prompt_tokens + result.completion_tokens}토큰 | ${result.cost_usd:.5f}] ")
    if result.tool_names_used:
        print(f"  tools: {', '.join(result.tool_names_used)}")
    print(f"assistant: {result.content}\n")
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result.content})
    return history


async def main() -> None:
    """입력: sys.argv(없으면 REPL, 있으면 1회 질문).
    출력: 없음(콘솔 출력).

    동작:
    1. llm_enabled 확인 — 키 없으면 안내 메시지와 함께 종료.
    2. 인자 있으면 " ".join(argv[1:])을 1회 질문으로 처리하고 종료.
    3. 인자 없으면 REPL 루프: input()으로 메시지를 받아 _chat 호출.
       exit/quit/EOF(Ctrl+D)/Ctrl+C로 종료. 개별 질문 실패는
       에러 출력 후 루프를 계속한다.
    4. finally에서 agent.aclose()로 HTTP 세션을 정리한다.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        sys.exit("OPENAI_API_KEY가 없습니다. .env.example을 참고해 .env를 만드세요.")

    agent = Agent(settings=settings)
    history: list[dict] = []
    try:
        if len(sys.argv) > 1:
            await _chat(agent, " ".join(sys.argv[1:]), history)
            return
        print("에이전트 REPL (종료: exit/quit)")
        while True:
            try:
                message = input("\nuser> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not message or message.lower() in {"exit", "quit"}:
                break
            try:
                await _chat(agent, message, history)
            except Exception as exc:  # noqa: BLE001 - REPL은 계속 진행
                print(f"error: {exc}")
    finally:
        await agent.aclose()


if __name__ == "__main__":
    """python -m agent.cli로 직접 실행될 때만 asyncio.run(main()).

    입력: 없음 / 출력: 없음.
    동작: __main__ 가드 — import 시에는 실행되지 않는다.
    """
    asyncio.run(main())
