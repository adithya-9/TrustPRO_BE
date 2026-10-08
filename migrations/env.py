"""Alembic environment.

The database URL comes from the application settings (DATABASE_URL), never from alembic.ini.
The version table is kept in the epsoft schema under a TrustPRO-specific name, so other
Trustmate products can run their own migrations in the same schema without colliding.
"""
from __future__ import annotations

import sys
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402

settings = get_settings()
VERSION_TABLE = "trustpro_alembic_version"


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        literal_binds=True,
        version_table=VERSION_TABLE,
        version_table_schema=settings.db_schema,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(settings.database_url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            version_table=VERSION_TABLE,
            version_table_schema=settings.db_schema,
            transaction_per_migration=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
