"""FastAPI surface of the aggregator."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from swarm import wire
from swarm.config import ExperimentConfig
from swarm.crypto import ckks
from swarm.server.coordinator import Coordinator, Reject

log = logging.getLogger("swarm.server")


class RegisterBody(BaseModel):
    node_id: str
    device_class: str = "phone"
    n_train: int
    n_val: int
    enroll_secret: str


def create_app(cfg: ExperimentConfig, *, public_ctx_path: str | None, out_dir: str,
               enroll_secret: str, exit_when_done: bool = False,
               linger_s: float = 8.0) -> FastAPI:
    pub = None
    if cfg.mode == "fhe":
        if not public_ctx_path:
            raise SystemExit("fhe mode requires SWARM_PUBLIC_CTX")
        pub = ckks.load_context(Path(public_ctx_path).read_bytes(), require_secret=False)
    coord = Coordinator(cfg, pub, Path(out_dir), enroll_secret)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async def loop():
            while True:
                try:
                    await coord.tick()
                except Exception:
                    log.exception("coordinator tick failed")
                    raise
                if coord.phase == "finished" and exit_when_done and coord.finished_at is not None:
                    await asyncio.sleep(linger_s)
                    srv = getattr(app.state, "uvicorn_server", None)
                    if srv is not None:
                        srv.should_exit = True
                    return
                await asyncio.sleep(0.2)

        task = asyncio.create_task(loop())
        yield
        task.cancel()

    app = FastAPI(title="SwarmShield aggregator", lifespan=lifespan)
    app.state.coordinator = coord

    @app.exception_handler(Reject)
    async def _reject(_: Request, exc: Reject):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)

    def auth(request: Request):
        h = request.headers.get("authorization", "")
        token = h[7:] if h.lower().startswith("bearer ") else None
        return coord.authenticate(request.headers.get("x-node-id"), token)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "phase": coord.phase}

    @app.post("/register")
    async def register(body: RegisterBody):
        info = coord.register(body.node_id, body.device_class, body.n_train, body.n_val,
                              body.enroll_secret)
        return {"token": info.token, "config": cfg.to_dict()}

    @app.get("/status")
    async def status(request: Request):
        node = auth(request)
        return coord.status(node.node_id)

    @app.get("/global")
    async def get_global(request: Request):
        auth(request)
        version, chunks = coord.current_global()
        if chunks is None:
            return Response(status_code=204, headers={"X-Model-Version": "0"})
        body = wire.pack(chunks)
        coord.bytes_out_total += len(body)
        return Response(body, media_type="application/octet-stream",
                        headers={"X-Model-Version": str(version)})

    @app.post("/update")
    async def update(request: Request):
        node = auth(request)
        body = await request.body()
        try:
            round_id = int(request.headers["x-round"])
            attempt = int(request.headers["x-attempt"])
            parts = wire.unpack(body)
            meta = json.loads(parts[0])
        except (KeyError, ValueError, IndexError) as exc:
            raise Reject(422, f"malformed update: {exc}") from exc
        await coord.submit_update(node, round_id, attempt,
                                  request.headers.get("x-signature", ""), body, meta, parts[1:])
        return {"accepted": True}

    @app.post("/eval")
    async def submit_eval(request: Request):
        node = auth(request)
        try:
            data = await request.json()
            round_id, attempt = int(data["round"]), int(data["attempt"])
        except (KeyError, ValueError) as exc:
            raise Reject(422, "malformed eval") from exc
        coord.submit_eval(node, round_id, attempt, data.get("eval", {}))
        return {"accepted": True}

    @app.get("/history")
    async def history():
        return coord.snapshot()

    return app
