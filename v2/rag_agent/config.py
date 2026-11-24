"""환경변수 기반 설정."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """앱 전체 설정. 환경변수명 = 필드명 대문자.

    입력:  .env 파일 + 프로세스 환경변수
    출력:  검증된 불변 설정 객체
    동작:  경로 필드는 프로젝트 루트(v2/) 기준 상대경로를 Path로 변환한다.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    openai_api_key: SecretStr = Field(default=SecretStr(""), max_length=255)
    openai_base_url: str = Field(default="https://api.openai.com/v1", max_length=300)

    agent_model: str = Field(default="gpt-4o-mini", max_length=100)
    embed_model: str = Field(default="text-embedding-3-small", max_length=100)
    agent_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    agent_max_tokens: int = Field(default=2048, ge=1, le=128_000)

    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    retrieve_top_k: int = Field(default=20, ge=1, le=100)
    final_top_k: int = Field(default=5, ge=1, le=50)
    rerank_candidates: int = Field(default=10, ge=1, le=50)

    chroma_dir: Path = Path("./chroma_db")
    bm25_index_path: Path = Path("./bm25_index.pkl")
    checkpoint_db: Path = Path("./checkpoints.db")
    longterm_db: Path = Path("./longterm_memory.db")
    traces_dir: Path = Path("./traces")

    @property
    def llm_enabled(self) -> bool:
        """입력: 없음 / 출력: API 키가 있으면 True.

        동작: 키 없이도 chunking/bm25 같은 로컬 기능은 테스트 가능.
        """
        return bool(self.openai_api_key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    """입력: 없음 / 출력: 캐시된 Settings 싱글턴.

    동작: 최초 1회만 생성. 테스트에서 환경을 바꾸려면 cache_clear().
    """
    return Settings()
