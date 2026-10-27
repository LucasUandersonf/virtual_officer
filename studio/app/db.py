"""Async engine + session factory. PostgreSQL (asyncpg) on Railway, SQLite locally and in tests."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.models import Base, OfficeState


class Database:
    def __init__(self, url: str) -> None:
        if url.startswith("sqlite"):
            path = url.split("///", 1)[-1]
            if path and path != ":memory:":
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        kwargs: dict[str, Any] = {} if url.startswith("sqlite") else {"pool_size": 5, "max_overflow": 5, "pool_pre_ping": True}
        self.engine: AsyncEngine = create_async_engine(url, **kwargs)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init(self) -> None:
        # create_all is idempotent; schema changes beyond new tables need a migration (see README).
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessions() as s:
            try:
                yield s
                await s.commit()
            except BaseException:
                await s.rollback()
                raise

    async def get_state(self, key: str) -> dict[str, Any]:
        async with self.session() as s:
            row = await s.get(OfficeState, key)
            return dict(row.value) if row else {}

    async def set_state(self, key: str, value: dict[str, Any]) -> None:
        async with self.session() as s:
            row = await s.get(OfficeState, key)
            if row:
                row.value = value
            else:
                s.add(OfficeState(key=key, value=value))

    async def close(self) -> None:
        await self.engine.dispose()


__all__ = ["Database", "select"]
