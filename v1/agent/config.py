"""환경변수 기반 설정. pydantic-settings로 .env를 자동 로드한다.

동작:
- Settings 인스턴스 생성 시 .env 파일과 실제 환경변수를 읽어 필드를 채운다.
- frozen=True라 생성 후 값 변경이 불가능하다(설정 불변성).
- openai_api_key는 SecretStr이라 repr/log에 마스킹되어 노출되지 않는다.
- get_settings()는 lru_cache로 프로세스당 1회만 파싱한다.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """앱 전체 설정. 환경변수명 = 필드명 대문자 (예: OPENAI_API_KEY).

    입력:  .env 파일 + 프로세스 환경변수
    출력:  검증된 불변 설정 객체. 범위 밖 값은 Field 제약(ge/le)에서 ValidationError.
    동작:  필드 기본값으로도 생성 가능 — 키 없이 테스트/서버 기동이 된다.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    openai_api_key: SecretStr = Field(default=SecretStr(""), max_length=255)
    """OpenAI API 키. SecretStr이라 str(settings)에도 '**********'로 마스킹."""

    openai_base_url: str = Field(default="https://api.openai.com/v1", max_length=300)
    """OpenAI 호환 API 엔드포인트. OpenRouter/로컬 LLM 등 프록시도 여기로 지정."""

    agent_model: str = Field(default="gpt-4o-mini", max_length=100)
    """채팅 completions에 사용할 모델명."""

    agent_max_iterations: int = Field(default=8, ge=1, le=50)
    """ReAct 루프 최대 반복 횟수. 초과 시 tools=None으로 최종 답변을 강제한다."""

    agent_max_tokens: int = Field(default=2048, ge=1, le=128_000)
    """completion 최대 토큰 수."""

    agent_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    """샘플링 온도. 낮을수록 결정적."""

    @property
    def llm_enabled(self) -> bool:
        """입력: 없음 / 출력: API 키가 비어있지 않으면 True.

        동작: 키 미설정 상태에서도 서버/테스트가 기동되도록 하는 게이트.
        /healthz 응답과 cli 사전검사에 사용한다.
        """
        return bool(self.openai_api_key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    """입력: 없음(.env/환경변수 자동 로드) / 출력: 캐시된 Settings 싱글턴.

    동작: lru_cache로 최초 1회만 Settings()를 생성한다. 테스트에서
    환경변수를 바꿔 검증하려면 get_settings.cache_clear()가 필요하다.
    """
    return Settings()
