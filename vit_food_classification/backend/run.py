"""
run.py

CLI entrypoint for launching the FastAPI application.

Supports:
- Running the app using Uvicorn with customizable host/port/debug settings
- Initializing the database schema via an async method
- Configurable schema/database through command-line arguments or environment variables

Typical usage:
    python run.py --host 127.0.0.1 --port 8080 --debug
    python run.py --init-db --database mydb --schema myschema
"""
import os
import argparse
import uvicorn
import asyncio  # Required for awaiting async DB setup
from colorama import Fore, Style

from app.main import create_app
from app.utils.database.db_init import InitDatabase


if __name__ == "__main__":
    # Argument parsing for CLI options
    parser = argparse.ArgumentParser(description="Run the FastAPI application")
    parser.add_argument("--host", default="0.0.0.0", help="Host address for the server")
    parser.add_argument("--port", type=int, default=8000, help="Port number for the server")
    parser.add_argument("--debug", action="store_true", help="Enable FastAPI debug mode")
    parser.add_argument("--init-db", action="store_true", help="Initialize the database tables")
    parser.add_argument("--database", help="Specify the database name to initialize")
    parser.add_argument("--schema", help="Specify the database schema to initialize")
    args = parser.parse_args()

    # Override database and schema from CLI args or fall back to environment defaults
    database = args.database if args.database else os.getenv('FASTAPI_DATABASE', 'postgres')
    schema = args.schema if args.schema else os.getenv('FASTAPI_SCHEMA', 'public')

    # Run async DB initialization if requested
    if args.init_db:
        print(f"Initializing database tables for '{database}.{schema}'...")

        db_initializer = InitDatabase(database=database, schema=schema)
        asyncio.run(db_initializer.create_tables())  # Await async method in event loop

        print(f"{Fore.GREEN}Database tables initialized successfully.{Style.RESET_ALL}")

    # Launch the FastAPI app using Uvicorn (factory pattern)
    app_module = "app.main:create_app"
    uvicorn.run(app_module, host=args.host, port=args.port, reload=args.debug, factory=True)
