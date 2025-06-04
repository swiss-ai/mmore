import argparse
import uvicorn
from fastapi import FastAPI

from .run_index_api import make_router as index_router
from .run_retriever import make_router as retriever_router


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
    
    app = FastAPI(title="Live Indexing & Retrieval API")
    
    app.include_router(index_router(args.config_file))
    app.include_router(retriever_router(args.config_file))
    
    uvicorn.run(app, host=args.host, port=args.port)