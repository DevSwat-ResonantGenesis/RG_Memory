import os

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

from .config import DATABASE_URL


Base = declarative_base()

pool_size = int(os.getenv("MEMORY_DB_POOL_SIZE", "10"))
max_overflow = int(os.getenv("MEMORY_DB_MAX_OVERFLOW", "5"))
pool_timeout = int(os.getenv("MEMORY_DB_POOL_TIMEOUT", "30"))
pool_recycle = int(os.getenv("MEMORY_DB_POOL_RECYCLE", "1800"))
pool_class = (os.getenv("MEMORY_DB_POOL_CLASS", "queue").strip().lower() or "queue")

_engine_kwargs = {
    "echo": False,
    "pool_pre_ping": True,
}

if pool_class in ("null", "none"):
    _engine_kwargs["poolclass"] = NullPool
else:
    _engine_kwargs.update(
        {
            "pool_size": pool_size,
            "max_overflow": max_overflow,
            "pool_timeout": pool_timeout,
            "pool_recycle": pool_recycle,
        }
    )

engine = create_async_engine(DATABASE_URL, **_engine_kwargs)

async_session = sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_session():
    async with async_session() as session:
        yield session
