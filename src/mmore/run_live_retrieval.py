import argparse

import uvicorn
from fastapi import FastAPI

from .run_index_api import make_router as index_router
from .run_retriever import make_router as retriever_router


def run(config_file: str, host: str, port: int):
    app = FastAPI(title="Live Indexing & Retrieval API")

    app.include_router(index_router(config_file))
    app.include_router(retriever_router(config_file))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, help="Path to the retriever configuration file."
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host on which the API should be run."
    )
    parser.add_argument(
        "--port", default=8000, help="Port on which the API should be run."
    )
    args = parser.parse_args()

    run(args.config_file, args.host, args.port)
