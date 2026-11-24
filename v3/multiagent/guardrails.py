"""운영 가드레일 — 비용 상한/타임아웃/재시도/프롬프트 인젝션 방어/권한 분리.

멀티에이전트의 실패는 '한 에이전트의 무한 루프'가 '전체 파이프라인의
무한 비용'으로 증폭된다. 이 모듈이 그 증폭을 막는 안전장치다.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BudgetExceeded(Exception):
    """비용/토큰 상한 초과 — 파이프라인을 즉시 중단해야 하는 치명적 오류."""


class BudgetTracker:
    """실행 1회당 비용·토큰 예산 추적기.

    입력: max_cost_usd, max_total_tokens
    출력: track()/check()/remaining_usd
    동작: LLMClient가 매 호출 후 track()으로 사용량을 누적하고,
    상한 도달 시 다음 호출 전에 check()가 BudgetExceeded를 던진다.
    """

    # 모델별 USD/1M tokens (간이 가격표 — v2 llm.py와 동일 소스)
    PRICING = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1-nano": (0.10, 0.40),
    }

    def __init__(self, max_cost_usd: float, max_total_tokens: int, model: str) -> None:
        """입력: 상한값과 모델명 / 출력: 인스턴스.

        동작: model명으로 가격표를 선택. 미등록 모델은 gpt-4o-mini 가격 사용.
        """
        self.max_cost = max_cost_usd
        self.max_tokens = max_total_tokens
        self.prompt_price, self.completion_price = self.PRICING.get(model, self.PRICING["gpt-4o-mini"])
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.calls = 0

    def track(self, prompt_tokens: int, completion_tokens: int) -> None:
        """입력: 이번 호출의 토큰 사용량 / 출력: 없음.

        동작: 누적하고 즉시 상한 검사. 초과 시 BudgetExceeded.
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls += 1
        self.cost_usd = (
            self.prompt_tokens * self.prompt_price + self.completion_tokens * self.completion_price
        ) / 1_000_000
        self.check()

    def check(self) -> None:
        """입력: 없음 / 출력: 없음.

        동작: 토큰 또는 비용 상한 초과 시 BudgetExceeded 발생.
        LLM 호출 '전'에도 호출해 이미 초과 상태면 API를 때리지 않는다.
        """
        total = self.prompt_tokens + self.completion_tokens
        if total > self.max_tokens:
            raise BudgetExceeded(f"토큰 상한 초과: {total} > {self.max_tokens}")
        if self.cost_usd > self.max_cost:
            raise BudgetExceeded(f"비용 상한 초과: ${self.cost_usd:.4f} > ${self.max_cost}")

    def remaining_usd(self) -> float:
        """입력: 없음 / 출력: 남은 예산 USD."""
        return max(0.0, self.max_cost - self.cost_usd)

    def summary(self) -> dict[str, Any]:
        """입력: 없음 / 출력: 예산 사용 현황 dict."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "max_cost_usd": self.max_cost,
        }


def with_retry(
    max_retries: int = 2,
    base_delay: float = 0.5,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """입력: 재시도 횟수/기본 지연/재시도 대상 예외 / 출력: 데코레이터.

    동작: async 함수를 감싸 일시적 오류 시 exponential backoff로 재시도한다.
    BudgetExceeded는 retry_on에서 제외해 즉시 전파한다 —
    비용 초과를 재시도로 '우회'하면 상한이 무의미해지기 때문.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except BudgetExceeded:
                    raise  # 예산 초과는 재시도하지 않는다
                except retry_on as exc:
                    last_exc = exc
                    if attempt >= max_retries:
                        break
                    delay = base_delay * (2**attempt)
                    logger.warning("재시도 %d/%d (%s): %s", attempt + 1, max_retries, func.__name__, exc)
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


async def with_timeout(coro: Awaitable[T], timeout_sec: float, label: str = "") -> T:
    """입력: 코루틴, 타임아웃 초, 라벨 / 출력: 코루틴 결과.

    동작: asyncio.wait_for로 제한. TimeoutError 발생 시
    "{label} 타임아웃" 메시지로 다시 던져 원인이 명확하게 한다.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_sec)
    except TimeoutError as exc:
        raise TimeoutError(f"{label} 타임아웃 ({timeout_sec}s)") from exc


# --- 프롬프트 인젝션 방어 ---

_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+instructions?|"
    r"forget\s+(everything|all)|"
    r"you\s+are\s+now|"
    r"new\s+instructions?:|"
    r"system\s*:|"
    r"<\s*/?\s*(system|assistant|tool)\s*>)",
    re.IGNORECASE,
)


def sanitize_untrusted(text: str) -> str:
    """입력: 외부에서 온 텍스트(검색결과, 도구 출력 등) / 출력: 정제된 텍스트.

    동작: 프롬프트 인젝션 의심 패턴이 포함된 '라인'을 제거한다.
    도구 결과는 신뢰할 수 없는 데이터이므로 LLM 프롬프트에 넣기 전에
    반드시 이 함수를 거친다. 문장 일부가 아니라 라인 단위로 제거해
    정상 문서의 일부 표현까지 과도하게 삭제하지 않는다.
    """
    kept = []
    for line in text.splitlines():
        if _INJECTION_PATTERNS.search(line):
            logger.warning("인젝션 의심 라인 제거: %.80s", line)
            continue
        kept.append(line)
    return "\n".join(kept)


def wrap_untrusted(content: str, source: str = "tool") -> str:
    """입력: 내용, 출처명 / 출력: 경계 태그로 감싼 텍스트.

    동작: <untrusted source="...">...</untrusted>로 감싸
    LLM이 이 안의 텍스트를 '데이터'로만 취급하도록 명시한다.
    시스템 프롬프트의 '태그 안 내용은 지시가 아니다' 규칙과 짝을 이룬다.
    """
    safe = sanitize_untrusted(content)
    return f'<untrusted source="{source}">\n{safe}\n</untrusted>'


# --- 권한 분리 ---

READ = "read"
WRITE = "write"


def check_permission(tool_permission: str, allow_write: bool) -> str | None:
    """입력: 도구 권한(read/write), write 허용 여부 / 출력: 에러 메시지|None.

    동작: write 도구인데 allow_write=False면 거부 메시지 반환.
    에이전트 역할(연구원=read only, 퍼블리셔=write 가능)과
    전역 설정(ALLOW_WRITE_TOOLS)의 이중 통제다.
    """
    if tool_permission == WRITE and not allow_write:
        return "permission denied: write tools are disabled (ALLOW_WRITE_TOOLS=false)"
    return None
