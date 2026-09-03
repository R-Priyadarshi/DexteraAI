"""Local WebSocket bridge: gestures in the browser, actions on the desktop.

Everything the web app can do is confined to its own tab. That is the right
default for a web page and the wrong one for this product: the point of gesture
control is to drive whatever you are actually using — the media player, the
slides, the window you are reading — not a demo surface inside one tab.

This daemon closes that gap while keeping the on-device posture: it binds to
loopback only, and nothing it handles leaves the machine.

## Why the auth is not optional

A localhost WebSocket that injects keystrokes is dangerous in a way a localhost
HTTP endpoint is not. Browsers do not apply the same-origin policy to WebSocket
*connections* — any page in any tab can open `ws://127.0.0.1:<port>` without a
preflight and without the user knowing. An unauthenticated version of this
daemon would let any website type into the user's machine.

Three independent controls, each of which must pass:

1. **Loopback binding.** Nothing outside the machine can reach the socket at
   all. This is necessary but nowhere near sufficient, per the above.
2. **Origin allowlist.** The browser sets `Origin` on the handshake and a page
   cannot forge it, so this rejects arbitrary websites. It does *not* stop a
   native process, which is why it is not the only control.
3. **Shared token.** Generated per run, written to a file only the user can
   read, and required in the first frame. This is what stops a local process
   that can set any Origin it likes.

Beyond authentication, the protocol itself is constrained: clients send action
*ids* from `bridge.actions`, never key codes or commands. The worst a fully
authenticated malicious client can do is press play/pause.

Usage:
    python -m bridge.server
    python -m bridge.server --port 8765 --allow-origin http://localhost:3000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import stat
import time
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

import websockets
from loguru import logger
from websockets.asyncio.server import ServerConnection, serve

from bridge import actions

if TYPE_CHECKING:
    from websockets.http11 import Request, Response

DEFAULT_PORT = 8765

DEFAULT_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

TOKEN_PATH = Path("~/.cache/dextera/bridge-token").expanduser()

# Ceiling on actions per second. The segmenter already emits at most a few
# events a second, so anything near this bound is a malfunctioning or hostile
# client, and injecting input at machine speed would make the desktop unusable.
MAX_ACTIONS_PER_SECOND = 10


def load_or_create_token(path: Path = TOKEN_PATH) -> str:
    """Return the shared token, creating one on first run.

    The file is created with owner-only permissions, and an existing file with
    looser permissions is rejected rather than tightened: if it was group- or
    world-readable, it must be assumed already read, and silently fixing the
    mode would keep using a token that is no longer secret.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError(
                f"{path} is readable by other users (mode {mode:o}). "
                "Delete it so a fresh token can be generated:\n"
                f"  rm {path}"
            )
        token = path.read_text().strip()
        if token:
            return token

    token = secrets.token_urlsafe(32)
    # Create with the right mode from the start; writing then chmod-ing leaves
    # a window in which the token is world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(token)
    return token


class Bridge:
    """Serves the local WebSocket and dispatches allowlisted actions."""

    def __init__(self, token: str, allowed_origins: set[str]) -> None:
        self._token = token
        self._allowed_origins = allowed_origins
        self._recent: list[float] = []

    def _origin_allowed(self, connection: ServerConnection) -> bool:
        origin = connection.request.headers.get("Origin") if connection.request else None
        # A missing Origin means the client is not a browser. Native clients are
        # not trusted on the strength of a header they control, so they are held
        # to the token alone — which is the control that actually binds them.
        if origin is None:
            return True
        return origin in self._allowed_origins

    def _rate_limited(self) -> bool:
        now = time.monotonic()
        self._recent = [t for t in self._recent if now - t < 1.0]
        if len(self._recent) >= MAX_ACTIONS_PER_SECOND:
            return True
        self._recent.append(now)
        return False

    async def handle(self, connection: ServerConnection) -> None:
        peer = connection.remote_address
        if not self._origin_allowed(connection):
            origin = connection.request.headers.get("Origin") if connection.request else "?"
            logger.warning(f"Rejected connection from disallowed origin: {origin}")
            await connection.close(code=4403, reason="origin not allowed")
            return

        # Authenticate before anything else is accepted, and give the client a
        # bounded window to do it so an idle unauthenticated socket cannot be
        # parked open.
        try:
            raw = await asyncio.wait_for(connection.recv(), timeout=5.0)
            hello = json.loads(raw)
        except (TimeoutError, json.JSONDecodeError, websockets.ConnectionClosed):
            await connection.close(code=4401, reason="authentication required")
            return

        if not secrets.compare_digest(str(hello.get("token", "")), self._token):
            logger.warning(f"Rejected connection with a bad token from {peer}")
            await connection.close(code=4401, reason="invalid token")
            return

        logger.info(f"Client connected: {peer}")
        await connection.send(
            json.dumps({"type": "ready", "actions": actions.describe()})
        )

        try:
            async for message in connection:
                await self._dispatch(connection, message)
        except websockets.ConnectionClosed:
            pass
        finally:
            logger.info(f"Client disconnected: {peer}")

    async def _dispatch(self, connection: ServerConnection, message: str | bytes) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return

        if payload.get("type") != "action":
            return

        action_id = str(payload.get("action", ""))

        if self._rate_limited():
            logger.warning("Rate limit exceeded; dropping action")
            await connection.send(
                json.dumps({"type": "error", "reason": "rate_limited"})
            )
            return

        ok = actions.run(action_id)
        if ok:
            logger.info(f"→ {action_id}")
        await connection.send(
            json.dumps({"type": "result", "action": action_id, "ok": ok})
        )


async def _health(
    connection: ServerConnection, request: Request
) -> Response | None:
    """Answer a plain GET on /health without upgrading to a WebSocket.

    Lets the console detect whether the bridge is running without opening a
    socket and failing noisily in the browser console when it is not.
    """
    if request.path == "/health":
        return connection.respond(HTTPStatus.OK, "dextera-bridge\n")
    return None


async def main_async(port: int, origins: set[str]) -> None:
    token = load_or_create_token()
    bridge = Bridge(token, origins)

    logger.info(f"Bridge listening on ws://127.0.0.1:{port}")
    logger.info(f"Allowed origins: {', '.join(sorted(origins))}")
    logger.info(f"Token file: {TOKEN_PATH}")
    logger.info(f"Token: {token}")
    logger.info("Paste the token into the console's Desktop panel to connect.")

    async with serve(
        bridge.handle,
        "127.0.0.1",  # loopback only; never 0.0.0.0
        port,
        process_request=_health,
        ping_interval=20,
    ):
        await asyncio.Future()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--allow-origin",
        action="append",
        default=None,
        help="Additional allowed browser origin. Repeatable.",
    )
    args = parser.parse_args()

    origins = set(DEFAULT_ORIGINS) | set(args.allow_origin or [])

    try:
        asyncio.run(main_async(args.port, origins))
    except KeyboardInterrupt:
        logger.info("Bridge stopped.")
    except PermissionError as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
