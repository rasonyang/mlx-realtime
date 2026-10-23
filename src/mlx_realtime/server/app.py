"""The Starlette application: ``/v1/realtime`` (WebSocket), ``/healthz``, ``/readyz``, ``/metrics``
and, when ``observability.debug_endpoints`` is on, ``/debug/turns``."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
from collections.abc import AsyncGenerator

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from mlx_realtime.protocol.encode import EventEncoder
from mlx_realtime.server.connection import Connection
from mlx_realtime.server.gateway import Gateway
from mlx_realtime.server.metrics import CONTENT_TYPE
from mlx_realtime.session.errors import ErrorInfo

logger = logging.getLogger(__name__)

_TRY_AGAIN_LATER = 1013


def create_app(gateway: Gateway) -> Starlette:
    async def realtime(websocket: WebSocket) -> None:
        if not gateway.authorized(websocket.headers.get("authorization")):
            gateway.metrics.session_refused("invalid_api_key")
            body = {"error": {"type": "invalid_request_error", "code": "invalid_api_key"}}
            await websocket.send_denial_response(JSONResponse(body, status_code=401))
            return
        await websocket.accept()
        if not gateway.readiness.ever_ready:
            gateway.metrics.session_refused("not_ready")
            await _refuse(websocket, "not_ready", "The service is starting; try again shortly.")
            return
        if gateway.admission.full:
            gateway.metrics.session_refused("session_limit_reached")
            await _refuse(websocket, "session_limit_reached", "Too many sessions; try again later.")
            return
        connection = Connection(websocket, gateway, model=websocket.query_params.get("model"))
        gateway.admission.enter(connection)
        gateway.metrics.session_started()
        try:
            await connection.run()
        finally:
            gateway.admission.leave(connection)

    async def healthz(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def readyz(_: Request) -> Response:
        ready, backends = await gateway.readiness.report()
        return JSONResponse(
            {"ready": ready, "backends": backends}, status_code=200 if ready else 503
        )

    async def metrics(_: Request) -> Response:
        backends = (await gateway.readiness.report())[1]
        gateway.metrics.set_readiness(backends)
        return Response(gateway.metrics.exposition(), media_type=CONTENT_TYPE)

    async def debug_turns(request: Request) -> Response:
        """Each turn's marks and stage durations for one session (the benchmark, D8); without
        ``session_id``, the sessions kept."""
        records = gateway.metrics.debug
        session_id = request.query_params.get("session_id")
        if session_id is None:
            return JSONResponse({"sessions": records.sessions()})
        turns = records.session(session_id)
        if turns is None:
            return JSONResponse({"error": f"no turns kept for session '{session_id}'"}, 404)
        return JSONResponse({"session_id": session_id, "turns": turns})

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncGenerator[None]:
        warming = asyncio.create_task(gateway.readiness.warm_up())
        try:
            yield
        finally:
            warming.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warming
            await gateway.aclose()

    routes = [
        WebSocketRoute("/v1/realtime", realtime),
        Route("/healthz", healthz),
        Route("/readyz", readyz),
        Route("/metrics", metrics),
    ]
    if gateway.config.observability.debug_endpoints:
        routes.append(Route("/debug/turns", debug_turns))
    return Starlette(routes=routes, lifespan=lifespan)


async def _refuse(websocket: WebSocket, code: str, message: str) -> None:
    """Admission refusal: an ``error`` event, then close 1013 (try again later)."""
    event = EventEncoder().error(ErrorInfo("server_error", code, message))
    await websocket.send_json(event)
    await websocket.close(_TRY_AGAIN_LATER, code)


class GatewayServer(uvicorn.Server):
    """uvicorn, closing sessions with 1001 before it shuts connections down (ARCHITECTURE §8)."""

    def __init__(self, gateway: Gateway) -> None:
        config = gateway.config
        super().__init__(
            uvicorn.Config(
                create_app(gateway),
                host=config.gateway.host,
                port=config.gateway.port,
                ws="websockets-sansio",
                ws_max_size=config.limits.client_max_message_bytes,
                log_config=None,
                lifespan="on",
            )
        )
        self._gateway = gateway

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        drain_s = self._gateway.config.limits.client_write_timeout_ms / 1000 + 1
        await self._gateway.admission.stop_all(1001, "server shutting down", drain_s)
        await super().shutdown(sockets)
