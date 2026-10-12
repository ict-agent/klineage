from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from klineage.harness import CodexRunner
from klineage.harness.runner import PROVIDER_KEY_PLACEHOLDER

AGENT_CLI = '''
import json
import os
import socket
import sys
from pathlib import Path
from urllib.request import Request, urlopen

args = sys.argv[1:]
# Task data precedes the runner's time-budget instructions.
task = json.loads(sys.stdin.read().splitlines()[2])
result = {
    "args": args,
    "host_secret": os.environ.get("KLINEAGE_TEST_HOST_KEY"),
    "provider_key": os.environ.get("TEST_PROVIDER_KEY"),
    "home": os.environ["HOME"],
    "devices": list(str(path) for path in Path("/dev").glob("nvidia*")),
}
overrides = dict(args[index + 1].split("=", 1) for index, value in enumerate(args) if value == "-c")
url = overrides["model_providers.klineage.base_url"]
result["endpoint"] = url
request = Request(url + "/responses", data=b'{"model":"test","input":"hi"}', headers={
    "Content-Type": "application/json",
    "Authorization": "Bearer " + os.environ["TEST_PROVIDER_KEY"],
})
with urlopen(request, timeout=3) as response:
    result["response"] = response.read().decode()
output = Path(args[args.index("--output-last-message") + 1])
output.write_text(json.dumps(result))
print(json.dumps({"event": "finished"}), flush=True)
'''


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.requests.append((self.path, self.headers["Authorization"], body, self.headers))
        payload = b'data: {"type":"response.completed","response":{"id":"test-response","model":"test","usage":{"input_tokens":10,"output_tokens":3,"total_tokens":13}}}\n\n'
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class AgentServerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.codex_home = self.root / "codex-home"
        self.codex_home.mkdir()
        self.enterContext(patch.dict(os.environ, {"CODEX_HOME": str(self.codex_home)}))
        self.work = self.root / "work"
        self.cli = self.root / "fake-codex"
        self.cli.write_text(f"#!{sys.executable}\n" + AGENT_CLI)
        self.cli.chmod(0o755)

    def runner(self, **options):
        return CodexRunner(self.work, codex_bin=self.cli, timeout=10, **options)

    def test_provider_new_resume(self):
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
        upstream.requests = []
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {
                "TEST_PROVIDER_KEY": "host-provider-secret", "KLINEAGE_TEST_HOST_KEY": "unrelated-secret",
            }):
                for session in (None, "recorded-session"):
                    options = {"api_key": "explicit-secret"} if session else {}
                    runner = self.runner(
                        api_base_url=f"http://127.0.0.1:{upstream.server_port}/api/v1",
                        api_key_env="TEST_PROVIDER_KEY", resume_id=session,
                        **options,
                    )
                    result = runner(json.dumps({"operation": "provider"}))
                    reply = json.loads(result.final_message)
                    self.assertEqual(reply["provider_key"], PROVIDER_KEY_PLACEHOLDER)
                    self.assertEqual(reply["endpoint"], "http://127.0.0.1:18081")
                    self.assertIn("response.completed", reply["response"])
                    self.assertEqual("resume" in reply["args"], session is not None)
                    self.assertIsNone(reply["host_secret"])
                    self.assertEqual(reply["devices"], [])
                    self.assertNotIn("host-provider-secret", result.final_message)
                    self.assertNotIn("host-provider-secret", result.trace_path.read_text())
                    events = [json.loads(line) for line in
                              (result.trace_path.parent / "stats.jsonl").read_text().splitlines()]
                    self.assertEqual(len(events), 1)
                    self.assertEqual(events[0]["event"], "api_response")
                    self.assertEqual(events[0]["status"], HTTPStatus.OK)
                    self.assertEqual(events[0]["input_tokens"], 10)
                    self.assertEqual(events[0]["output_tokens"], 3)
                    self.assertEqual(events[0]["total_tokens"], 13)
                    self.assertNotIn("host-provider-secret", json.dumps(events))
            self.assertEqual(len(upstream.requests), 2)
            for index, (path, authorization, body, _) in enumerate(upstream.requests):
                self.assertEqual(path, "/api/v1/responses")
                expected = "explicit-secret" if index else "host-provider-secret"
                self.assertEqual(authorization, "Bearer " + expected)
                self.assertEqual(json.loads(body)["model"], "test")
        finally:
            upstream.shutdown()
            upstream.server_close()
            thread.join(2)


if __name__ == "__main__":
    unittest.main()
