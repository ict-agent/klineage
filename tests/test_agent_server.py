"""Real sandbox IPC with host evaluation and provider HTTP services."""

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

from kernel_fixtures import edit_kernel, kernel, measured, write_kernel
from problem_fixtures import problem_spec

from klineage.backend.cuda import CudaKernel
from klineage.artifact import load_artifact
from klineage.artifact.kernel import latest_kernel
from klineage.backend import get_backend
from klineage.harness import CodexRunner
from klineage.harness.codex_runner import PROVIDER_KEY_PLACEHOLDER
from klineage.serialization import to_json
from klineage.server import Client, Service

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
if task["operation"] == "service":
    import klineage
    from klineage.server import ServerError
    from klineage.serialization import to_json
    result["endpoint"] = os.environ["KLINEAGE_SERVER"]
    result["evaluation_token"] = os.environ.get("KLINEAGE_SERVER_TOKEN")
    result["info"] = klineage.info()
    result["problem"] = json.loads(to_json(klineage.problem()))
    from klineage.artifact import load_problem
    local_problem = Path("problems/gemm")
    result["problem_link"] = local_problem.is_symlink()
    result["local_problem"] = load_problem(local_problem).name if local_problem.is_dir() else None
    result["problem_writable"] = False
    if local_problem.is_dir():
        try:
            with (local_problem / "definition.json").open("a"):
                pass
            result["problem_writable"] = True
        except OSError:
            pass
    result["problem_mount"] = os.environ.get("KLINEAGE_PROBLEM")
    try:
        klineage.problem("other")
        result["scope_enforced"] = False
    except ServerError:
        result["scope_enforced"] = True
    with socket.socket() as probe:
        probe.settimeout(0.1)
        result["host_tcp_blocked"] = probe.connect_ex(("127.0.0.1", task["port"])) != 0
    reply = klineage.evaluate(Path("current_kernel"))
    result["job_id"] = reply.job_id
    result["accepted"] = reply.accepted
    result["recovered"] = klineage.result(reply).job_id == reply.job_id
    result["docs"] = Path("workspace/AGENTS.md").read_text()
elif task["operation"] == "collection":
    import klineage
    details = klineage.info()
    results = []
    for name in details["problems"]:
        spec = klineage.problem(name)
        path = Path("current_kernel/metadata.json")
        metadata = json.loads(path.read_text())
        metadata["problem"] = {"path": details["problem_paths"][name], **spec.target()}
        path.write_text(json.dumps(metadata))
        results.append(klineage.evaluate(Path("current_kernel")))
    result["problems"] = [reply.problem_id for reply in results]
    result["accepted"] = all(klineage.result(reply).accepted for reply in results)
    result["references"] = [json.loads(reply.kernel.find("metadata.json").content)["problem"]["path"] for reply in results]
    result["problem_mount"] = os.environ.get("KLINEAGE_PROBLEM")
elif task["operation"] == "provider":
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


class RecordingRuntime:
    calls = []

    def __init__(self, *, timeout):
        self.timeout = timeout

    def info(self, backend, **options):
        return backend.info()

    def evaluate(self, candidate, **options):
        self.calls.append((os.getpid(), candidate))
        options["log_dir"].mkdir(parents=True)
        (options["log_dir"] / "worker.log").write_text("test runtime evidence")
        return measured(candidate)


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

    def test_agent_calls_host_service(self):
        backend = get_backend("cuda")
        self.enterContext(patch.object(type(backend), "info", return_value={"available": True, "platform": "sm120"}))
        service = Service(self.root / "server", backend, {
            "gemm": problem_spec(), "other": problem_spec(),
        }, runtime_factory=RecordingRuntime)
        server = self.enterContext(service.start())
        client = Client(server.endpoint, "gemm", timeout=5)
        write_kernel(kernel(), self.work / "current_kernel")
        RecordingRuntime.calls = []
        with patch.dict(os.environ, {"KLINEAGE_TEST_HOST_KEY": "not-in-agent"}):
            result = self.runner(evaluation=client)(json.dumps({
                "operation": "service", "port": int(server.endpoint.rsplit(":", 1)[1]),
            }))
        reply = json.loads(result.final_message)
        self.assertTrue(reply["accepted"])
        self.assertTrue(reply["recovered"])
        self.assertTrue(reply["scope_enforced"])
        self.assertTrue(reply["host_tcp_blocked"])
        self.assertEqual(reply["info"]["problems"], ["gemm"])
        self.assertEqual(reply["problem"], json.loads(to_json(problem_spec())))
        self.assertTrue(reply["problem_link"])
        self.assertEqual(reply["local_problem"], "gemm")
        self.assertFalse(reply["problem_writable"])
        self.assertTrue(reply["endpoint"].startswith("unix:///"))
        self.assertIsNone(reply["evaluation_token"])
        self.assertIsNone(reply["problem_mount"])
        self.assertIsNone(reply["host_secret"])
        self.assertEqual(reply["devices"], [])
        self.assertEqual(reply["home"], "/home/agent")
        self.assertEqual(len(RecordingRuntime.calls), 1)
        self.assertEqual(RecordingRuntime.calls[0][0], os.getpid())
        self.assertNotIn("klineage.harness.evaluate", reply["docs"])
        self.assertNotIn("klineage.harness.eval.evaluate", reply["docs"])
        self.assertNotIn("klineage.tools.profile", reply["docs"])
        self.assertNotIn("load_kernel(build_root", reply["docs"])
        self.assertIn("klineage.evaluate", reply["docs"])
        record = latest_kernel(self.work) / "metadata.json"
        self.assertEqual(json.loads(record.read_text())["evaluation"]["job_id"], reply["job_id"])
        self.assertFalse((self.work / "evaluations").exists())
        events = [json.loads(line) for line in
                  (result.trace_path.parent / "stats.jsonl").read_text().splitlines()]
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["event"], "evaluate")
        self.assertEqual(event["problem_id"], "gemm")
        self.assertEqual(event["job_id"], reply["job_id"])
        self.assertTrue(event["accepted"])
        self.assertTrue(event["compile_passed"])
        self.assertTrue(event["correctness_passed"])
        self.assertTrue(event["profile_passed"])
        self.assertEqual(set(event["workloads"]), {case["uuid"] for case in problem_spec().workloads})
        self.assertLessEqual(event["at"], event["finished_at"])
        self.assertNotIn("sources", event)
        measured = load_artifact(latest_kernel(self.work), artifact_type=CudaKernel)
        client.verify(measured, reply["job_id"])

    def test_agent_external_inputs(self):
        data = self.root / "input.safetensors"
        data.write_bytes(b"fixed input")
        problem = problem_spec()
        cases = list(problem.workloads)
        cases[0]["inputs"]["a"] = {
            "type": "safetensors", "path": str(data), "tensor_key": "a",
        }
        problem = problem.replace(workloads=cases)
        backend = get_backend("cuda")
        self.enterContext(patch.object(type(backend), "info", return_value={"available": True}))
        service = Service(
            self.root / "server", backend, {"gemm": problem}, runtime_factory=RecordingRuntime,
        )
        server = self.enterContext(service.start())
        write_kernel(edit_kernel(kernel(), problem=problem), self.work / "current_kernel")
        result = self.runner(evaluation=Client(server.endpoint, "gemm"))(json.dumps({
            "operation": "service", "port": int(server.endpoint.rsplit(":", 1)[1]),
        }))
        reply = json.loads(result.final_message)
        self.assertEqual(reply["local_problem"], problem.name)
        self.assertTrue(reply["accepted"])
        self.assertEqual(data.read_bytes(), b"fixed input")

    def test_agent_uses_problem_set(self):
        backend = get_backend("cuda")
        self.enterContext(patch.object(type(backend), "info", return_value={"available": True, "platform": "sm120"}))
        service = Service(self.root / "server", backend, {
            "gemm": problem_spec(), "other": problem_spec(),
        }, runtime_factory=RecordingRuntime)
        server = self.enterContext(service.start())
        write_kernel(kernel(), self.work / "current_kernel")
        with patch.dict(os.environ, {"KLINEAGE_PROBLEM": str(self.root / "host-problems")}):
            result = self.runner(evaluation=Client(server.endpoint))(
                json.dumps({"operation": "collection"})
            )
        reply = json.loads(result.final_message)
        self.assertEqual(reply["problems"], ["gemm", "other"])
        self.assertTrue(reply["accepted"])
        self.assertEqual(len(set(reply["references"])), 2)
        self.assertIsNone(reply["problem_mount"])

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

    def test_provider_config_fallback(self):
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
        upstream.requests = []
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            (self.codex_home / "config.toml").write_text(
                'model = "configured-test-model"\n'
                'model_provider = "configured"\n'
                '[model_providers.configured]\n'
                'name = "Configured provider"\n'
                f'base_url = "http://127.0.0.1:{upstream.server_port}/api/v1"\n'
                'env_key = "TEST_PROVIDER_KEY"\n'
                'wire_api = "responses"\n'
                'http_headers = { "X-Provider-Account" = "host-account-secret" }\n'
            )
            with patch.dict(os.environ, {
                "TEST_PROVIDER_KEY": "host-provider-secret",
                "KLINEAGE_TEST_HOST_KEY": "unrelated-secret",
            }):
                result = self.runner()(json.dumps({"operation": "provider"}))
            reply = json.loads(result.final_message)
            self.assertEqual(reply["provider_key"], PROVIDER_KEY_PLACEHOLDER)
            self.assertEqual(reply["endpoint"], "http://127.0.0.1:18081")
            self.assertEqual(reply["args"][reply["args"].index("--model") + 1], "configured-test-model")
            self.assertIsNone(reply["host_secret"])
            self.assertIn("response.completed", reply["response"])
            for secret in ("host-provider-secret", "host-account-secret", "unrelated-secret"):
                self.assertNotIn(secret, result.final_message)
                self.assertNotIn(secret, result.trace_path.read_text())
            self.assertEqual(len(upstream.requests), 1)
            path, authorization, body, headers = upstream.requests[0]
            self.assertEqual(path, "/api/v1/responses")
            self.assertEqual(authorization, "Bearer host-provider-secret")
            self.assertEqual(headers.get_all("X-Provider-Account"), ["host-account-secret"])
            self.assertEqual(json.loads(body)["model"], "test")
        finally:
            upstream.shutdown()
            upstream.server_close()
            thread.join(2)


if __name__ == "__main__":
    unittest.main()
