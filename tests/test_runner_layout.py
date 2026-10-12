"""Both CLIs preserve the same artifact and per-attempt record layout."""

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

from kernel_fixtures import kernel
from problem_fixtures import problem_spec
from test_agent_server import RecordingRuntime

from klineage.action.action import collect_output
from klineage.artifact import ArtifactKind, source_files
from klineage.backend import get_backend
from klineage.harness import ClaudeCodeRunner, CodexRunner
from klineage.serialization import raw_text
from klineage.server import Client, Service

RECORD_FILES = {
    "prompt.txt", "trace.jsonl", "stats.jsonl", "stderr.log", "final_message.txt",
}
AGENT = '''
import json, os, sys
from pathlib import Path
from urllib.request import Request, urlopen
import klineage

args = sys.argv[1:]
task = json.loads(sys.stdin.read().splitlines()[2])
claude = "--output-format" in args
if claude:
    endpoint = os.environ["ANTHROPIC_BASE_URL"] + "/v1/messages?beta=true"
else:
    settings = dict(args[i+1].split("=", 1) for i, value in enumerate(args) if value == "-c")
    endpoint = settings["model_providers.klineage.base_url"] + "/responses"
with urlopen(Request(endpoint, data=b"{}"), timeout=5) as response:
    json.load(response)

current = Path("current_kernel")
current.mkdir()
metadata = task["metadata"]
metadata["problem"] = {"path":klineage.info()["problem_paths"]["gemm"], **klineage.problem().target()}
(current / "metadata.json").write_text(json.dumps(metadata))
for name, content in task["submission"].items():
    path = current / "submission" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
reply = klineage.evaluate(current)
assert reply.accepted
if claude:
    print(json.dumps({"type":"system", "subtype":"init", "session_id":"f1567b81-8f7e-4444-81d0-8f531ef6d35c"}))
    print(json.dumps({"type":"result", "subtype":"success", "is_error":False, "result":"done"}))
else:
    print(json.dumps({"type":"thread.started", "thread_id":"codex-session"}))
    print(json.dumps({"type":"turn.completed"}))
    Path(args[args.index("--output-last-message")+1]).write_text("done")
'''


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        usage = {"input_tokens":10, "output_tokens":2, "total_tokens":12}
        payload = {"id":"reply", "model":"test-model", "usage":usage}
        if self.path.startswith("/v1/messages"):
            payload["type"] = "message"
        data = json.dumps(payload).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class RunnerLayoutTests(unittest.TestCase):
    def test_artifacts_and_records(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(patch.dict(os.environ, {"CODEX_HOME":str(root / "codex-home"), "KLINEAGE_SERVER":""}))
        program = root / "agent"
        program.write_text(f"#!{sys.executable}\n" + AGENT)
        program.chmod(0o755)
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        backend = get_backend("cuda")
        self.enterContext(patch.object(type(backend), "info", return_value={"available":True, "platform":"sm120"}))
        service = Service(root / "server", backend, {"gemm":problem_spec()}, runtime_factory=RecordingRuntime)
        server = self.enterContext(service.start())
        client = Client(server.endpoint, "gemm")
        candidate = kernel()
        task = json.dumps({
            "metadata":json.loads(raw_text(candidate.find("metadata.json"))),
            "submission":source_files(candidate.find("submission")),
        })
        layouts = []
        for name, runner_type, binary in (
            ("codex", CodexRunner, "codex_bin"),
            ("claude", ClaudeCodeRunner, "claude_bin"),
        ):
            with self.subTest(runner=name):
                work = root / name
                runner = runner_type(
                    work, **{binary:program}, api_key="host-secret",
                    api_base_url=f"http://127.0.0.1:{upstream.server_port}",
                    model="test-model", evaluation=client, timeout=15,
                )
                result = runner(task, run_id="attempt")
                record = result.trace_path.parent
                self.assertEqual(record.relative_to(work), Path("workspace/.klineage/agent-runs/attempt"))
                self.assertEqual({path.name for path in record.iterdir()}, RECORD_FILES)
                stats = [json.loads(line) for line in (record / "stats.jsonl").read_text().splitlines()]
                api = next(row for row in stats if row["event"] == "api_response")
                evaluation = next(row for row in stats if row["event"] == "evaluate")
                self.assertEqual(api["total_tokens"], 12)
                self.assertTrue(evaluation["accepted"])
                trace = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
                self.assertEqual(trace[-1]["type"], "result" if name == "claude" else "turn.completed")
                artifact = collect_output(work, ArtifactKind.GENERATION, evaluation=client)
                records = source_files(artifact.find("records"))
                self.assertEqual(set(records), {"attempt/" + value for value in RECORD_FILES})
                self.assertEqual(records["attempt/trace.jsonl"], result.trace_path.read_text())
                layouts.append({
                    str(path.relative_to(work)) for directory in ("current_kernel", "versions", "problems")
                    for path in (work / directory).rglob("*")
                })
        self.assertEqual(len(layouts), 2)
        self.assertEqual(*layouts)

    def test_legacy_records(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        for name in ("codex-runs", "claude-runs"):
            with self.subTest(directory=name):
                work = root / name
                record = work / "workspace/.klineage" / name / "attempt"
                record.mkdir(parents=True)
                (record / "trace.jsonl").write_text('{"old":true}\n')
                artifact = collect_output(work, ArtifactKind.ACTION)
                self.assertEqual(source_files(artifact.find("records")), {
                    "attempt/trace.jsonl":'{"old":true}\n',
                })
