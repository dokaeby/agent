"""환경변수 기반 설정."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """앱 전체 설정. 환경변수명 = 필드명 대문자.

    입력: .env + 환경변수 / 출력: 불변 설정 객체
    동작: 운영 가드레일(비용/토큰/타임아웃/재시도/권한) 값도 여기서 관리.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    openai_api_key: SecretStr = Field(default=SecretStr(""), max_length=255)
    openai_base_url: str = Field(default="https://api.openai.com/v1", max_length=300)

    agent_model: str = Field(default="gpt-4o-mini", max_length=100)
    agent_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    agent_max_tokens: int = Field(default=2048, ge=1, le=128_000)

    # 운영 가드레일
    max_cost_usd: float = Field(default=0.50, gt=0)
    max_total_tokens: int = Field(default=200_000, gt=0)
    agent_timeout_sec: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_review_rounds: int = Field(default=2, ge=0, le=5)
    review_pass_score: float = Field(default=0.7, ge=0.0, le=1.0)
    allow_write_tools: bool = Field(default=False)

    knowledge_dir: Path = Path("./data/knowledge")
    reports_dir: Path = Path("./reports")

    @property
    def llm_enabled(self) -> bool:
        """입력: 없음 / 출력: API 키 유무."""
        return bool(self.openai_api_key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    """입력: 없음 / 출력: 캐시된 Settings 싱글턴."""
    return Settings()
