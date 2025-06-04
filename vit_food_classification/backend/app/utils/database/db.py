"""
db.py

Provides an asynchronous PostgreSQL connection engine and session factory using SQLAlchemy with asyncpg.

Supports:
- Vault-injected credentials
- Schema-specific connections via search_path
- Safe instantiation of async sessions for use in FastAPI or background tasks
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import urllib.parse


class PostgresConnEngine:
    """
    Initializes an async SQLAlchemy engine and session factory for PostgreSQL.

    Args:
        config (dict): Vault-injected dictionary containing connection credentials.
            Expected keys:
                - POSTGRES_USER
                - POSTGRES_PASSWORD
                - POSTGRES_HOST
                - POSTGRES_PORT
        database (str): Name of the database to connect to.
        schema (str): Target schema to apply via search_path.
    """
    def __init__(self, config: dict, database: str, schema: str):
        self.__config = config
        self.database = database
        self.schema = schema

        self.__async_engine = self.__create_async_engine()

        self.__async_session_factory = sessionmaker(
            bind=self.__async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    def __create_async_engine(self):
        """Constructs and returns the async SQLAlchemy engine with search_path set to the target schema."""
        user = self.__config.get("POSTGRES_USER")
        password = self.__config.get("POSTGRES_PASSWORD")
        host = self.__config.get("POSTGRES_HOST")
        port = self.__config.get("POSTGRES_PORT")

        encoded_schema = urllib.parse.quote(self.schema)

        db_uri = f'postgresql+asyncpg://{user}:{password}@{host}:{port}/{self.database}'
        connect_args = {"options": f"-csearch_path={encoded_schema}"}

        return create_async_engine(
            db_uri,
            echo=True,
            connect_args=connect_args
        )

    def get_async_engine(self):
        """
        Returns:
            SQLAlchemy async engine instance for raw connection access.
        """
        return self.__async_engine

    def get_async_session(self):
        """
        Returns:
            AsyncSession: A fresh session for DB interaction.
        """
        return self.__async_session_factory()
