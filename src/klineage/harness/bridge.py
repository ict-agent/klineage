"""Stream provider requests across a sandbox's private network namespace."""

from __future__ import annotations

import argparse
import contextlib
import http.client
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn, UnixStreamServer
from urllib.parse import urlsplit

from klineage.harness.logging import ResponseUsage, now, record
from klineage.harness.protocol import WireAPI

RELAY_PORT = 18081
BODY_LIMIT = 32 * 1024 * 1024
TRANSFER_BYTES = 16 * 1024
CONNECT_TIMEOUT = 10.0
STREAM_TIMEOUT = 180.0
POLL_SECONDS = 0.1
STOP_SECONDS = 2.0
PROVIDER_PATHS = {
    WireAPI.RESPONSES: frozenset(("/responses", "/responses/compact")),
    WireAPI.MESSAGES: frozenset(
        route + query
        for route in ("/v1/messages", "/v1/messages/count_tokens")
        for query in ("", "?beta=true")
    ),
}
REQUEST_HEADERS = frozenset((
    "anthropic-version", "anthropic-beta", "anthropic-workspace-id",
    "content-type", "accept", "openai-beta", "openai-organization", "openai-project",
    "session_id", "conversation_id", "x-codex-turn-state", "x-codex-turn-metadata",
    "x-codex-beta-features", "originator", "version", "user-agent",
))
RESPONSE_HEADERS = frozenset((
    "content-type", "retry-after", "x-request-id", "openai-processing-ms",
    "x-codex-turn-state", "request-id",
))
HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
PROTECTED_HEADERS = frozenset((
    "host", "content-length", "transfer-encoding", "connection", "keep-alive",
    "proxy-connection", "proxy-authenticate", "proxy-authorization", "te",
    "trailer", "upgrade", "expect",
))


def disconnect(connection: socket.socket) -> None:
    with contextlib.suppress(OSError):
        connection.shutdown(socket.SHUT_RDWR)


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: Path):
        super().__init__("localhost", timeout=STREAM_TIMEOUT)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


class Connections:
    """Close both sides before joining request threads during shutdown."""

    daemon_threads = False
    block_on_close = True

    def configure(self, connect, prefix="", headers=None, *, directory=None,
                  wire_api=WireAPI.RESPONSES):
        self.paths = PROVIDER_PATHS[WireAPI(wire_api)]
        self.connect = connect
        self.prefix = prefix
        self.headers = dict(headers or {})
        self.directory = directory
        self.lock = threading.Lock()
        self.connections = set()
        self.closing = False

    def register(self, connection):
        with self.lock:
            self.connections.add(connection)
            if self.closing:
                disconnect(connection)

    def discard(self, connection):
        with self.lock:
            self.connections.discard(connection)

    def process_request_thread(self, request, address):
        self.register(request)
        try:
            super().process_request_thread(request, address)
        finally:
            self.discard(request)

    def close_streams(self):
        with self.lock:
            self.closing = True
            for connection in self.connections:
                disconnect(connection)

    def handle_error(self, request, address):
        # Never expose provider URLs, request contents or credentials in logs.
        pass


class UnixServer(Connections, ThreadingMixIn, UnixStreamServer):
    pass


class RelayServer(Connections, ThreadingMixIn, HTTPServer):
    pass


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(STREAM_TIMEOUT)

    def log_message(self, *args):
        pass

    def reject(self, status):
        self.response_status = status
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def body_length(self):
        target = self.requestline.split()[1]
        if target not in self.server.paths or self.command != "POST":
            raise ValueError("unsupported provider route")
        names = set()
        for name, value in self.headers.items():
            if name.lower() in names or any(ord(char) < 32 or ord(char) == 127 for char in value):
                raise ValueError("ambiguous HTTP headers")
            names.add(name.lower())
        if "transfer-encoding" in names or "expect" in names:
            raise ValueError("unsupported request framing")
        value = self.headers.get("Content-Length", "")
        if not value.isascii() or not value.isdecimal():
            raise ValueError("Content-Length is required")
        length = int(value)
        if length > BODY_LIMIT:
            raise OverflowError("request exceeds limit")
        return length

    def watch_client(self, upstream, stopped):
        while not stopped.wait(POLL_SECONDS):
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if readable and not self.connection.recv(1, socket.MSG_PEEK):
                    disconnect(upstream)
                    return
            except OSError:
                disconnect(upstream)
                return

    def forward(self):
        started_at = now()
        started = time.monotonic()
        self.response_status = HTTPStatus.BAD_GATEWAY
        self.stats = {}
        self.usage = None
        try:
            self.forward_request()
        finally:
            if self.server.directory is not None:
                if self.usage is not None:
                    self.stats.update(self.usage.finish())
                record(
                    "api_response", directory=self.server.directory,
                    at=started_at, finished_at=now(),
                    duration_ms=(time.monotonic() - started) * 1000,
                    status=self.response_status, **self.stats,
                )

    def forward_request(self):
        self.close_connection = True
        try:
            length = self.body_length()
        except ValueError:
            self.reject(HTTPStatus.BAD_REQUEST)
            return
        except OverflowError:
            self.reject(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return

        upstream = None
        peer = None
        watcher = None
        stopped = threading.Event()
        started = False
        try:
            upstream = self.server.connect()
            upstream.connect()
            peer = upstream.sock
            peer.settimeout(STREAM_TIMEOUT)
            self.server.register(peer)
            upstream.putrequest("POST", self.server.prefix + self.path, skip_accept_encoding=True)
            for name, value in self.headers.items():
                if name.lower() in REQUEST_HEADERS and name.lower() not in self.server.headers:
                    upstream.putheader(name, value)
            for name, value in self.server.headers.items():
                upstream.putheader(name, value)
            upstream.putheader("Content-Length", str(length))
            upstream.putheader("Connection", "close")
            upstream.endheaders()

            while length:
                data = self.rfile.read(min(length, TRANSFER_BYTES))
                if not data:
                    raise ConnectionError("incomplete request")
                upstream.send(data)
                length -= len(data)

            # A disconnected client cancels even an idle upstream SSE stream.
            watcher = threading.Thread(target=self.watch_client, args=(peer, stopped), daemon=True)
            watcher.start()
            with upstream.getresponse() as response:
                self.response_status = response.status
                if self.server.directory is not None and response.status < HTTPStatus.BAD_REQUEST:
                    self.usage = ResponseUsage(response.headers.get("Content-Type", ""))
                self.send_response(response.status)
                for name, value in response.headers.items():
                    if name.lower() in RESPONSE_HEADERS:
                        self.send_header(name, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                started = True
                while data := response.read1(TRANSFER_BYTES):
                    if self.usage is not None:
                        self.usage.feed(data)
                    self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (OSError, http.client.HTTPException) as error:
            self.stats["error"] = type(error).__name__
            if not started:
                with contextlib.suppress(OSError):
                    self.reject(HTTPStatus.BAD_GATEWAY)
        finally:
            stopped.set()
            if peer is not None:
                disconnect(peer)
                self.server.discard(peer)
            if upstream is not None:
                upstream.close()
            if watcher is not None:
                watcher.join(STOP_SECONDS)

    do_POST = forward
    do_GET = forward
    do_DELETE = forward
    do_PUT = forward
    do_CONNECT = forward

    def handle_expect_100(self):
        self.reject(HTTPStatus.BAD_REQUEST)
        return False


class Service:
    def start(self):
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": POLL_SECONDS}, daemon=True,
        )
        self.thread.start()
        return self

    def close(self):
        if self.server is None:
            return
        self.server.shutdown()
        self.server.close_streams()
        self.server.server_close()
        self.thread.join(STOP_SECONDS)
        self.server = None

    def __exit__(self, *exc):
        self.close()


class ProviderBridge(Service):
    """Host-only credentials and a fixed upstream behind one private socket.

    Host headers override sandbox headers. A non-None API key overrides their
    Authorization value; None leaves authentication to the supplied headers.
    An explicit directory receives one stats.jsonl record per request.
    """

    def __init__(
        self, base_url: str, api_key: str | None, socket_path: Path,
        *, headers: Mapping[str, str] | None = None, directory: Path | None = None,
        wire_api: WireAPI = WireAPI.RESPONSES,
    ):
        self.wire_api = WireAPI(wire_api)
        target = urlsplit(base_url)
        if (target.scheme not in ("http", "https") or not target.hostname
                or target.username or target.password or target.query or target.fragment
                or any(char in base_url for char in "\r\n\\")
                or "%" in target.path or ".." in target.path.split("/")):
            raise ValueError("provider must have a fixed HTTP(S) base URL")
        if api_key is not None and (
            not isinstance(api_key, str) or not api_key
            or any(ord(char) < 33 or ord(char) > 126 for char in api_key)
        ):
            raise ValueError("provider requires a nonempty API key")
        if headers is not None and not isinstance(headers, Mapping):
            raise ValueError("provider headers must be a mapping")
        self.headers = {}
        for name, value in (headers or {}).items():
            if not isinstance(name, str) or not HEADER_NAME.fullmatch(name):
                raise ValueError("invalid provider header name")
            if not isinstance(value, str) or any(
                ord(char) < 32 or ord(char) == 127 or ord(char) > 255 for char in value
            ):
                raise ValueError("invalid provider header value")
            name = name.lower()
            if name in PROTECTED_HEADERS:
                raise ValueError("provider headers cannot control HTTP framing")
            if name in self.headers:
                raise ValueError("duplicate provider header")
            self.headers[name] = value
        if api_key is not None:
            self.headers["authorization"] = "Bearer " + api_key
        self.target = target
        self.socket_path = Path(socket_path).absolute()
        self.directory = directory
        self.server = None

    @property
    def url(self):
        return f"http://127.0.0.1:{RELAY_PORT}"

    def command(self, argv, *, python=sys.executable):
        return [str(python), "-m", "klineage.harness.bridge", "--socket",
                str(self.socket_path), "--port", str(RELAY_PORT),
                "--wire-api", self.wire_api.value, "--", *map(str, argv)]

    def connect(self):
        cls = http.client.HTTPSConnection if self.target.scheme == "https" else http.client.HTTPConnection
        return cls(self.target.hostname, self.target.port, timeout=CONNECT_TIMEOUT)

    def __enter__(self):
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.server = UnixServer(str(self.socket_path), ProxyHandler)
        self.socket_path.chmod(0o600)
        prefix = self.target.path.rstrip("/")
        if self.wire_api == WireAPI.MESSAGES:
            prefix = prefix.removesuffix("/v1")
        self.server.configure(
            self.connect, prefix, self.headers, directory=self.directory,
            wire_api=self.wire_api,
        )
        return self.start()

    def close(self):
        if self.server is None:
            return
        super().close()
        self.socket_path.unlink(missing_ok=True)


class LoopbackRelay(Service):
    """Sandbox-side HTTP endpoint; its only destination is the mounted socket."""

    def __init__(self, socket_path: Path, *, port=RELAY_PORT, wire_api=WireAPI.RESPONSES):
        self.socket_path = Path(socket_path)
        self.wire_api = WireAPI(wire_api)
        self.port = port
        self.server = None

    def __enter__(self):
        self.server = RelayServer(("127.0.0.1", self.port), ProxyHandler)
        self.port = self.server.server_port
        self.server.configure(lambda: UnixConnection(self.socket_path), wire_api=self.wire_api)
        return self.start()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--port", default=RELAY_PORT, type=int)
    parser.add_argument("--wire-api", choices=list(WireAPI), default=WireAPI.RESPONSES)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("an agent command is required")

    with LoopbackRelay(args.socket, port=args.port, wire_api=args.wire_api):
        process = subprocess.Popen(command)

        def terminate(signum, frame):
            process.send_signal(signum)
            raise SystemExit(128 + signum)

        previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(STOP_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
