"""
db_init.py

This module defines InitDatabase, a utility class to initialize database schemas and tables
using SQLAlchemy (async) with dynamic schema injection. It is designed for use with HashiCorp Vault
for credential management and supports runtime-safe schema setup.

Typically used during development or deployment setup via the `--init-db` flag in run.py.
"""

from colorama import Fore, Style
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, configure_mappers
from sqlalchemy import text
import asyncio

from app.utils.security.vault_mgr import VaultManager
from app.utils.database.db import PostgresConnEngine  # Must provide SQLAlchemy engine object
from app.models import Base  # SQLAlchemy Base containing all model metadata


class InitDatabase:
    """
    Initializes the database connection and dynamically sets the schema for all SQLAlchemy models.

    Args:
        database (str): The name of the target database.
        schema (str): The schema within the database where tables will be created.
    """
    def __init__(self, database: str, schema: str):
        print(f"""
        Initializing database connection and setting schema for models...
        {Fore.YELLOW}Database: {database}{Style.RESET_ALL}
        {Fore.YELLOW}Schema: {schema}{Style.RESET_ALL}
        """)

        self.vault_manager = VaultManager()  # Fetch secrets securely from Vault
        self.database = database
        self.schema = schema

        # Get base SQLAlchemy engine (sync) with configured credentials and schema
        self.engine = PostgresConnEngine(
            config=self.vault_manager.read_secret(mount_point='db', path='postgresql/postgres'),
            database=self.database,
            schema=self.schema
        ).get_async_engine()

        # Wrap in an asyncpg-compatible async engine for async DB operations
        self.async_engine = create_async_engine(
            self.engine.url.render_as_string(hide_password=False).replace("postgresql://", "postgresql+asyncpg://"),
            echo=True,
        )

        self.async_session_factory = sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Modify SQLAlchemy table metadata to apply target schema
        self._set_model_schema()

    def _set_model_schema(self):
        """
        Dynamically assigns the schema name to all SQLAlchemy tables before table creation.
        Only updates tables that don't already have an explicit schema set.
        """
        configure_mappers()  # Ensures all models are discovered
        for mapper in Base.registry.mappers:
            table = mapper.local_table
            if table.schema is None:
                table.schema = self.schema

    async def create_tables(self):
        """
        Creates all tables under the specified schema. Ensures the schema exists first.
        """
        await self._ensure_schema_exists()

        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        print(f"Tables created in database `{self.database}` under schema `{self.schema}`!")

    async def _ensure_schema_exists(self):
        """
        Verifies that the schema exists in the database, creating it if necessary.
        """
        async with self.async_engine.begin() as conn:
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema};"))

        print(f"Schema `{self.schema}` ensured in database `{self.database}`.")
