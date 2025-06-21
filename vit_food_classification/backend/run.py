"""
run.py

CLI entrypoint for launching the FastAPI application.

Supports:
- Running the FastAPI app using **Uvicorn** with customizable host, port, and debug settings.
- Initializing the **database schema** and **tables** via an async method.
- Configurable **schema/database** through command-line arguments or environment variables.

Typical usage:
    python run.py --host 127.0.0.1 --port 8080 --debug
    python run.py --init-db --database mydb --schema myschema

Arguments:
    --host          : Host address for the server (default: "0.0.0.0")
    --port          : Port number for the server (default: 8000)
    --debug         : Enable FastAPI debug mode (default: False)
    --init-db       : Initialize the database schema and tables
    --database      : Specify the database name to initialize (default: environment variable FASTAPI_DATABASE)
    --schema        : Specify the schema to initialize (default: environment variable FASTAPI_SCHEMA)

How it works:
- The script sets up the FastAPI server by running **Uvicorn** with the provided or default **host** and **port**.
- If **--init-db** is specified, it initializes the database schema using the provided **database** and **schema** or falls back to environment variables.
- The **InitDatabase** class is used to asynchronously create tables in the specified database schema.
- The FastAPI app is launched using the **factory pattern**, which allows you to configure and run the app from `app.main:create_app`.

Example:
    To run the app in debug mode on a custom host and port:
        python run.py --host 127.0.0.1 --port 8080 --debug

    To initialize a specific database schema:
        python run.py --init-db --database mydb --schema myschema
"""

import os
import argparse
import uvicorn
import asyncio  # Required for awaiting async DB setup
from colorama import Fore, Style

from app.main import create_app

if __name__ == "__main__":
    # Argument parsing for CLI options
    parser = argparse.ArgumentParser(description="Run the FastAPI application")
    parser.add_argument("--host", default="0.0.0.0", help="Host address for the server")
    parser.add_argument("--port", type=int, default=8000, help="Port number for the server")
    parser.add_argument("--debug", action="store_true", help="Enable FastAPI debug mode")
    args = parser.parse_args()

    # Launch the FastAPI app using Uvicorn (factory pattern)
    app_module = "app.main:create_app"
    uvicorn.run(app_module, host=args.host, port=args.port, reload=args.debug, factory=True)
