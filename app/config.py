import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "memory_service"

    POSTGRES_HOST: str = "memory_db"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "memory_user"
    POSTGRES_PASSWORD: str = "memory_pass"
    POSTGRES_DB: str = "memory_db"

    # DigitalOcean Spaces Configuration (S3-compatible)
    MINIO_ENDPOINT: str = os.getenv("MEMORY_STORAGE_ENDPOINT", os.getenv("STORAGE_ENDPOINT", "sfo3.digitaloceanspaces.com"))
    MINIO_ROOT_USER: str = os.getenv("MEMORY_STORAGE_ACCESS_KEY", os.getenv("DO_SPACES_ACCESS_KEY", ""))
    MINIO_ROOT_PASSWORD: str = os.getenv("MEMORY_STORAGE_SECRET_KEY", os.getenv("DO_SPACES_SECRET_KEY", ""))
    MINIO_BUCKET: str = os.getenv("MEMORY_STORAGE_BUCKET", "genesis2026-memory")
    MINIO_SECURE: bool = os.getenv("MEMORY_STORAGE_SECURE", "true").lower() == "true"
    MINIO_REGION: str = os.getenv("MEMORY_STORAGE_REGION", "sfo3")

    # Embeddings
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536
    
    # Nomic Embed (local embeddings - free, no API key needed)
    USE_NOMIC_EMBED: bool = True  # Use Nomic Embed as primary
    NOMIC_MATRYOSHKA_DIM: int = 512  # 768, 512, 256, 128, or 64

    # Chunking
    CHUNK_SIZE: int = 500  # tokens
    CHUNK_OVERLAP: int = 50  # tokens

    # LLM Service
    LLM_SERVICE_URL: str = "http://llm_service:8000"

    class Config:
        env_file = ".env"
        env_prefix = "MEMORY_"
        case_sensitive = False


settings = Settings()

DATABASE_URL = os.getenv(
    "MEMORY_DATABASE_URL",
    f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
).replace("postgresql://", "postgresql+asyncpg://").replace("?sslmode=", "?ssl=")
