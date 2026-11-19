"""Aggregator entrypoint:  python -m swarm.server.main   (configured via SWARM_* env vars)."""

from __future__ import annotations

import logging
import os

import uvicorn

from swarm.config import ExperimentConfig
from swarm.server.app import create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = ExperimentConfig.from_env()
    app = create_app(
        cfg,
        public_ctx_path=os.environ.get("SWARM_PUBLIC_CTX"),
        out_dir=os.environ.get("SWARM_OUT_DIR", "runs/latest"),
        enroll_secret=os.environ.get("SWARM_ENROLL_SECRET", "dev-enroll-secret-change-me"),
        exit_when_done=os.environ.get("SWARM_EXIT_WHEN_DONE", "0") == "1",
    )
    server = uvicorn.Server(uvicorn.Config(app, host=os.environ.get("SWARM_HOST", "0.0.0.0"),
                                           port=int(os.environ.get("SWARM_PORT", "8080")),
                                           log_level="warning"))
    app.state.uvicorn_server = server
    server.run()


if __name__ == "__main__":
    main()
