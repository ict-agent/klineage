"""Provider isolation and streamed forwarding over a Unix socket."""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from klineage.harness.bridge import (
    BODY_LIMIT,
    TRANSFER_BYTES,
    LoopbackRelay,
    ProviderBridge,
    UnixConnection,
)
from klineage.harness.logging import LOG_ENV


class Provider(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        first = self.rfile.read(min(size, TRANSFER_BYTES))
        self.server.received.set()
        body = first + self.rfile.read(size - len(first))
        self.server.requests.append((self.path, self.headers, body))
        self.send_response(self.server.status)
        self.send_header("Content-Type", self.server.content_type)
        self.send_header("Connection", "close")
        self.end_headers()
        if self.server.chunks is not None:
            for data in self.server.chunks:
                self.wfile.write(data)
                self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(b"data: first\n\n")
        self.wfile.flush()
        self.server.release.wait(3)
        try:
            self.wfile.write(b"data: last\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
        self.upstream.daemon_threads = True
        self.upstream.requests = []
        self.upstream.received = threading.Event()
        self.upstream.release = threading.Event()
        self.upstream.status = HTTPStatus.OK
        self.upstream.content_type = "text/event-stream"
        self.upstream.chunks = None
        self.thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.thread.start()
        self.stack.callback(self.stop_provider)
        self.base_url = f"http://127.0.0.1:{self.upstream.server_port}/api/v1"
        self.path = self.root / "provider.sock"
        self.directory = self.root / "run"
        self.bridge = self.stack.enter_context(
            ProviderBridge(self.base_url, "host-secret", self.path, directory=self.directory)
        )

    def stats(self):
        deadline = time.monotonic() + 2
        while self.bridge.server and self.bridge.server.connections and time.monotonic() < deadline:
            time.sleep(0.01)
        path = self.directory / "stats.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_anthropic_routes(self):
        from klineage.harness.bridge import WireAPI

        self.bridge.close()
        self.bridge = self.stack.enter_context(ProviderBridge(
            self.base_url, None, self.path, headers={"x-api-key": "host-secret"},
            wire_api=WireAPI.MESSAGES,
        ))
        self.upstream.chunks = [b'{"type":"message","content":[]}']
        with LoopbackRelay(self.path, port=0, wire_api=WireAPI.MESSAGES) as relay:
            for route in ("/v1/messages", "/v1/messages?beta=true", "/v1/messages/count_tokens"):
                connection = http.client.HTTPConnection("127.0.0.1", relay.port, timeout=2)
                self.addCleanup(connection.close)
                connection.request("POST", route, b"{}", {
                    "x-api-key": "sandbox-key", "Authorization": "Bearer sandbox-token",
                    "anthropic-version": "2023-06-01", "anthropic-beta": "test-beta",
                })
                response = connection.getresponse()
                self.assertEqual(response.status, HTTPStatus.OK)
                response.read()
                path, headers, _ = self.upstream.requests[-1]
                self.assertEqual(path, "/api" + route)
                self.assertEqual(headers["x-api-key"], "host-secret")
                self.assertNotIn("Authorization", headers)
                self.assertEqual(headers["anthropic-version"], "2023-06-01")
                self.assertEqual(headers["anthropic-beta"], "test-beta")
            connection = self.connection()
            connection.request("POST", "/responses", b"{}")
            self.assertEqual(connection.getresponse().status, HTTPStatus.BAD_REQUEST)

    def usage_response(self):
        return {
            "id": "response-1", "model": "test-model", "object": "response",
            "usage": {
                "input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens_details": {"reasoning_tokens": 1},
            },
        }

    def test_records_split_sse_usage(self):
        payload = json.dumps({"type": "response.completed", "response": self.usage_response()}).encode()
        for ending in (b"\n", b"\r\n"):
            frame = b"event: response.completed" + ending + b"data: " + payload + ending * 2
            self.upstream.chunks = [frame[index:index + 7] for index in range(0, len(frame), 7)]
            connection = self.connection()
            connection.request("POST", "/responses", b'{"private":"request-body"}')
            self.assertEqual(connection.getresponse().read(), frame)
        entries = self.stats()
        self.assertEqual(len(entries), 2)
        for entry in entries:
            self.assertEqual(entry["event"], "api_response")
            self.assertEqual(entry["status"], HTTPStatus.OK)
            self.assertEqual(entry["total_tokens"], 12)
            self.assertEqual(entry["cached_tokens"], 4)
            self.assertEqual(entry["reasoning_tokens"], 1)
            self.assertGreaterEqual(entry["duration_ms"], 0)
            self.assertLessEqual(entry["at"], entry["finished_at"])
        text = (self.directory / "stats.jsonl").read_text()
        self.assertNotIn("host-secret", text)
        self.assertNotIn("request-body", text)

    def test_json_usage_once(self):
        body = json.dumps(self.usage_response()).encode()
        self.upstream.content_type = "application/json"
        self.upstream.chunks = [body[:17], body[17:]]
        with patch.dict(os.environ, {LOG_ENV: str(self.directory)}):
            with LoopbackRelay(self.path, port=0) as relay:
                connection = http.client.HTTPConnection("127.0.0.1", relay.port, timeout=2)
                self.addCleanup(connection.close)
                connection.request("POST", "/responses", b"{}")
                self.assertEqual(connection.getresponse().read(), body)
        [entry] = self.stats()
        self.assertEqual(entry["response_id"], "response-1")
        self.assertEqual(entry["model"], "test-model")
        self.assertEqual(entry["input_tokens"], 10)

    def test_records_http_failure(self):
        self.upstream.status = HTTPStatus.TOO_MANY_REQUESTS
        self.upstream.content_type = "application/json"
        self.upstream.chunks = [b'{"error":"provider-secret"}']
        connection = self.connection()
        connection.request("POST", "/responses", b"{}")
        response = connection.getresponse()
        self.assertEqual(response.status, HTTPStatus.TOO_MANY_REQUESTS)
        self.assertEqual(response.read(), self.upstream.chunks[0])
        [entry] = self.stats()
        self.assertEqual(entry["status"], HTTPStatus.TOO_MANY_REQUESTS)
        self.assertNotIn("provider-secret", json.dumps(entry))

    def test_connection_failure(self):
        with patch.object(self.bridge.server, "connect", side_effect=ConnectionRefusedError("secret-host")):
            connection = self.connection()
            connection.request("POST", "/responses", b"{}")
            response = connection.getresponse()
            self.assertEqual(response.status, HTTPStatus.BAD_GATEWAY)
            self.assertEqual(response.read(), b"")
            [entry] = self.stats()
        self.assertEqual(entry["status"], HTTPStatus.BAD_GATEWAY)
        self.assertEqual(entry["error"], "ConnectionRefusedError")
        self.assertNotIn("secret-host", json.dumps(entry))

    def stop_provider(self):
        self.upstream.release.set()
        self.upstream.shutdown()
        self.upstream.server_close()
        self.thread.join(2)

    def connection(self):
        connection = UnixConnection(self.path)
        self.addCleanup(connection.close)
        return connection

    def test_forwards_and_streams(self):
        connection = self.connection()
        connection.request("POST", "/responses", b"{}", {
            "Authorization": "Bearer sandbox-dummy",
            "Content-Type": "application/json",
            "X-Secret": "discard",
        })
        response = connection.getresponse()
        self.assertEqual(response.status, HTTPStatus.OK)
        self.assertEqual(response.read(len(b"data: first\n\n")), b"data: first\n\n")
        self.assertFalse(self.upstream.release.is_set())
        path, headers, body = self.upstream.requests[0]
        self.assertEqual(path, "/api/v1/responses")
        self.assertEqual(headers["Authorization"], "Bearer host-secret")
        self.assertNotIn("X-Secret", headers)
        self.assertEqual(body, b"{}")
        self.upstream.release.set()
        self.assertEqual(response.read(), b"data: last\n\n")
        [entry] = self.stats()
        self.assertEqual(entry["status"], HTTPStatus.OK)
        self.assertNotIn("total_tokens", entry)

    def test_request_streams(self):
        connection = self.connection()
        connection.putrequest("POST", "/responses")
        connection.putheader("Content-Length", str(TRANSFER_BYTES * 2))
        connection.endheaders()
        connection.send(b"a" * TRANSFER_BYTES)
        self.assertTrue(self.upstream.received.wait(2), "request body was buffered")
        connection.send(b"b" * TRANSFER_BYTES)
        self.upstream.release.set()
        self.assertEqual(connection.getresponse().status, HTTPStatus.OK)

    def test_host_headers_override_agent(self):
        path = self.root / "headers.sock"
        self.stack.enter_context(ProviderBridge(
            self.base_url, "host-secret", path,
            headers={
                "ChatGPT-Account-Id": "host-account",
                "api-key": "header-secret",
                "OpenAI-Project": "host-project",
                "aUtHoRiZaTiOn": "Basic ignored",
            },
        ))
        relay = self.stack.enter_context(LoopbackRelay(path, port=0))
        connection = http.client.HTTPConnection("127.0.0.1", relay.port, timeout=2)
        self.addCleanup(connection.close)
        connection.request("POST", "/responses", b"{}", {
            "Authorization": "Bearer sandbox-dummy",
            "ChatGPT-Account-Id": "sandbox-account",
            "API-Key": "sandbox-key",
            "openai-project": "sandbox-project",
        })
        self.upstream.release.set()
        self.assertEqual(connection.getresponse().status, HTTPStatus.OK)
        headers = self.upstream.requests[-1][1]
        for name, value in (
            ("ChatGPT-Account-Id", "host-account"),
            ("api-key", "header-secret"),
            ("OpenAI-Project", "host-project"),
            ("Authorization", "Bearer host-secret"),
        ):
            self.assertEqual(headers.get_all(name), [value])

    def test_optional_bearer(self):
        cases = (None, {"api-key": "header-secret"}, {"Authorization": "Basic host-secret"})
        for index, supplied in enumerate(cases):
            with self.subTest(headers=supplied):
                path = self.root / f"optional-{index}.sock"
                self.stack.enter_context(ProviderBridge(self.base_url, None, path, headers=supplied))
                connection = UnixConnection(path)
                self.addCleanup(connection.close)
                connection.request("POST", "/responses", b"{}", {
                    "Authorization": "Bearer sandbox-dummy", "api-key": "sandbox-key",
                    "ChatGPT-Account-Id": "sandbox-account",
                })
                self.upstream.release.set()
                self.assertEqual(connection.getresponse().status, HTTPStatus.OK)
                headers = self.upstream.requests[-1][1]
                expected = supplied or {}
                self.assertEqual(headers.get("Authorization"), expected.get("Authorization"))
                self.assertEqual(headers.get("api-key"), expected.get("api-key"))
                self.assertNotIn("ChatGPT-Account-Id", headers)

    def test_invalid_host_headers(self):
        headers = [
            {name: "sensitive"} for name in (
                "Host", "Content-Length", "Transfer-Encoding", "Connection", "Keep-Alive",
                "Proxy-Connection", "Proxy-Authorization", "Proxy-Authenticate", "TE",
                "Trailer", "Upgrade", "Expect", "X-Bad\r\nName", "Bad Name", "",
            )
        ]
        headers.extend({"api-key": value} for value in (
            "sensitive\r\nX-Forged: value", "sensitive\n", "sensitive\x00", "sensitive\u0100", 123,
        ))
        headers.append({"api-key": "sensitive", "API-KEY": "sensitive"})
        for supplied in headers:
            with self.subTest(headers=supplied):
                with self.assertRaises(ValueError) as error:
                    ProviderBridge(self.base_url, None, self.root / "invalid.sock", headers=supplied)
                self.assertNotIn("sensitive", str(error.exception))

    def test_rejects_paths(self):
        for path in ("http://example.com/responses", "//responses", "/../responses",
                     "/responses?url=x", "/responses/%2e%2e", "/models", "/responses/"):
            with self.subTest(path=path):
                connection = self.connection()
                connection.request("POST", path, b"{}")
                self.assertEqual(connection.getresponse().status, HTTPStatus.BAD_REQUEST)
        self.assertEqual(self.upstream.requests, [])

    def test_rejects_framing(self):
        invalid = (
            f"Content-Length: {BODY_LIMIT + 1}\r\n",
            "Content-Length: 2\r\nContent-Length: 2\r\n",
            "Transfer-Encoding: chunked\r\nContent-Length: 2\r\n",
            "Content-Length: -2\r\n",
            "Content-Length: 2\r\nX-Test: x\r\n folded\r\n",
        )
        for headers in invalid:
            with self.subTest(headers=headers):
                connection = self.connection()
                connection.connect()
                connection.send(("POST /responses HTTP/1.1\r\nHost: local\r\n" + headers + "\r\n").encode())
                response = http.client.HTTPResponse(connection.sock)
                response.begin()
                self.assertIn(response.status, (HTTPStatus.BAD_REQUEST, HTTPStatus.REQUEST_ENTITY_TOO_LARGE))
        self.assertEqual(self.upstream.requests, [])

    def test_loopback_relay(self):
        relay = self.stack.enter_context(LoopbackRelay(self.path, port=0))
        connection = http.client.HTTPConnection("127.0.0.1", relay.port, timeout=2)
        self.addCleanup(connection.close)
        connection.request("POST", "/responses/compact", b"{}")
        response = connection.getresponse()
        self.assertEqual(response.read(len(b"data: first\n\n")), b"data: first\n\n")
        self.assertEqual(self.upstream.requests[0][0], "/api/v1/responses/compact")

    def test_shutdown_closes_stream(self):
        connection = self.connection()
        connection.request("POST", "/responses", b"{}")
        response = connection.getresponse()
        response.read(len(b"data: first\n\n"))
        started = time.monotonic()
        self.bridge.close()
        self.assertLess(time.monotonic() - started, 2)
        self.assertFalse(self.path.exists())
        [entry] = self.stats()
        self.assertEqual(entry["event"], "api_response")

    def test_disconnect_cancels(self):
        connection = self.connection()
        connection.request("POST", "/responses", b"{}")
        response = connection.getresponse()
        response.read(len(b"data: first\n\n"))
        response.close()
        connection.close()
        deadline = time.monotonic() + 2
        while self.bridge.server.connections and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(self.bridge.server.connections)
        self.assertFalse(self.upstream.release.is_set())

    def test_launcher_exit_status(self):
        command = self.bridge.command([sys.executable, "-c", "raise SystemExit(7)"])
        command[command.index("--port") + 1] = "0"
        result = subprocess.run(command, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 7, result.stderr.decode())
        self.assertEqual(result.stderr, b"")

    def test_launcher_cancellation(self):
        marker = self.root / "child.pid"
        source = (
            "import os,pathlib,signal,time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
        )
        command = self.bridge.command([sys.executable, "-c", source])
        command[command.index("--port") + 1] = "0"
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 3
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(marker.exists())
            pid = int(marker.read_text())
            process.terminate()
            _, error = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 143, error.decode())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()

    def test_invalid_provider(self):
        for url in ("file:///secret", "https://host/a/../b", "https://user:pass@host",
                    "https://host/api?redirect=x", "https://host/%2e%2e"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ProviderBridge(url, "key", self.root / "invalid.sock")
        for key in ("", "key\r\nX-Evil: x"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ProviderBridge(self.base_url, key, self.root / "invalid.sock")

    def test_private_socket_and_command(self):
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        command = self.bridge.command(["codex", "exec"])
        self.assertNotIn("host-secret", " ".join(command))
        self.assertIn(str(self.path), command)
        self.assertEqual(command[-2:], ["codex", "exec"])
        self.assertEqual(self.bridge.url, "http://127.0.0.1:18081")


if __name__ == "__main__":
    unittest.main()
