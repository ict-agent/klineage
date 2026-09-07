from __future__ import annotations

import csv
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from klineage.errors import ActionError
from klineage.harness._docker import _DockerRuntime
from klineage.harness._profiling import _ncu_command, _parse_metrics
from klineage.kernel import Kernel, TargetContext


_LONG_HEADER = (
    "ID", "Kernel Name", "Section Name", "Metric Name", "Metric Unit", "Metric Value",
)
# These columns and values come from NCU 2025.3.1's bundled Sobel report.
_WIDE_HEADER = (
    "ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context",
    "Stream", "Block Size", "Grid Size", "Device", "CC",
    "gpu__time_duration.sum", "launch__registers_per_thread",
)
_WIDE_UNITS = ("",) * 11 + ("ns", "register/thread")
_WIDE_ROW = (
    "0", "3661953", "instructionMix", "127.0.0.1",
    "void Sobel<float>(uchar4 *, uchar4 *, int, int)", "1", "7",
    "(16, 16, 1)", "(64, 64, 1)", "0", "8.6", "31872", "22",
)


def _csv_text(*rows):
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(rows)
    return stream.getvalue()


class ProfileCommandTests(unittest.TestCase):
    def test_target_only_command(self):
        options = SimpleNamespace(set="full", sections=(), kernel_filter=None)

        self.assertEqual(
            _ncu_command(options, Path("/out/raw.csv"), Path("/out/report")),
            (
                "/usr/local/cuda/bin/ncu", "--target-processes", "application-only",
                "--replay-mode", "kernel", "--nvtx", "--nvtx-include",
                "klineage_profile/", "--set", "full", "--page", "raw", "--csv",
                "--print-units", "base", "--log-file", "/out/raw.csv",
                "--export", "/out/report", "--force-overwrite",
                "--kernel-name-base", "demangled",
            ),
        )

    def test_sections_and_kernel_filter(self):
        options = SimpleNamespace(
            set="basic", sections=("LaunchStats", "Occupancy"), kernel_filter="Sobel.*",
        )
        command = _ncu_command(options, Path("/raw.csv"), Path("/report"))

        self.assertEqual(command.count("--section"), 2)
        self.assertIn(("--section", "LaunchStats"), tuple(zip(command, command[1:])))
        self.assertIn(("--section", "Occupancy"), tuple(zip(command, command[1:])))
        self.assertEqual(command[command.index("--kernel-name") + 1], "regex:Sobel.*")

        options.kernel_filter = "regex:Sobel.*"
        command = _ncu_command(options, Path("/raw.csv"), Path("/report"))
        self.assertEqual(command[command.index("--kernel-name") + 1], "regex:Sobel.*")


class ProfileParserTests(unittest.TestCase):
    def test_long_logs_headers(self):
        kernel = 'void kernel<float, "value">\n(int *)'
        row = ("7", kernel, "Launch Statistics", "Registers Per Thread", "register/thread", "32")
        text = (
            "==PROF== Connected to process 42\n" + _csv_text(_LONG_HEADER, row)
            + "==PROF== Profiling, pass 2\n" + _csv_text(_LONG_HEADER, row)
            + "==PROF== Disconnected from process 42\n"
        )

        self.assertEqual(_parse_metrics(text), [{
            "kernel": kernel, "launch_id": "7", "section": "Launch Statistics",
            "metric": "Registers Per Thread", "unit": "register/thread", "value": "32",
        }] * 2)

    def test_installed_raw_csv_format(self):
        text = _csv_text(_WIDE_HEADER, _WIDE_UNITS, _WIDE_ROW)

        self.assertEqual(_parse_metrics(text), [
            {"kernel": _WIDE_ROW[4], "launch_id": "0", "section": "",
             "metric": "gpu__time_duration.sum", "unit": "ns", "value": "31872"},
            {"kernel": _WIDE_ROW[4], "launch_id": "0", "section": "",
             "metric": "launch__registers_per_thread", "unit": "register/thread", "value": "22"},
        ])

    def test_wide_repeated_and_empty(self):
        first = _WIDE_ROW[:-2] + ("0", "")
        second = ("1",) + _WIDE_ROW[1:-2] + ("N/A", "1,234")
        text = _csv_text(_WIDE_HEADER, _WIDE_UNITS, first, _WIDE_HEADER, _WIDE_UNITS, second)

        metrics = _parse_metrics(text)
        self.assertEqual([row["value"] for row in metrics], ["0", "N/A", "1,234"])
        self.assertEqual([row["launch_id"] for row in metrics], ["0", "1", "1"])

    def test_counter_permission_error(self):
        text = _csv_text(_WIDE_HEADER, _WIDE_UNITS, _WIDE_ROW)

        with self.assertRaisesRegex(ValueError, "ERR_NVGPUCTRPERM"):
            _parse_metrics(text + "==ERROR== ERR_NVGPUCTRPERM: Permission denied\n")

    def test_no_profiled_kernels(self):
        with self.assertRaisesRegex(ValueError, "No kernels were profiled"):
            _parse_metrics("==WARNING== No kernels were profiled.\n")

    def test_no_metrics(self):
        for text in ("", "==PROF== Connected\n", _csv_text(_LONG_HEADER),
                     _csv_text(_WIDE_HEADER, _WIDE_UNITS, _WIDE_ROW[:-2] + ("", ""))):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "no metric"):
                _parse_metrics(text)


class ProfileDriverTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        (self.root / "src").mkdir()
        self.problem = self.root / "reference.py"
        self.problem.touch()
        artifact = self.root / "kernel.cu"
        artifact.write_text("__global__ void target() {}")
        self.kernel = Kernel("target", artifact.read_text(), TargetContext("gemm", "cuda", "sm120"),
                             artifact_path=artifact)
        self.mount = Path("/workspace/klineage")
        self.runtime = _DockerRuntime(root=self.root, docker="docker", image="image-id", mount=self.mount)
        self.options = SimpleNamespace(set="detailed", sections=(), kernel_filter=None, timeout_seconds=37)
        self.calls = []

    def run_profile(self, **kwargs):
        return self.runtime.profile(self.kernel, problem=self.problem, build_root=self.root / "build",
                                    log_dir=self.root / "logs", options=self.options, **kwargs)

    def complete(self, command, **kwargs):
        self.calls.append((command, kwargs))
        raw = self.root / Path(command[command.index("--log-file") + 1]).relative_to(self.mount)
        report = self.root / Path(command[command.index("--export") + 1]).relative_to(self.mount)
        raw.write_text(_csv_text(_WIDE_HEADER, _WIDE_UNITS, _WIDE_ROW))
        report.write_bytes(b"NCU report")
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "tool_version": "2025.3.1", "device": {"uuid": "GPU-test", "name": "GPU", "capability": "sm120"},
        }), "")

    def test_isolated_profile_transport(self):
        with patch("subprocess.run", side_effect=self.complete):
            result = self.run_profile()
        command, kwargs = self.calls[0]
        request = json.loads(kwargs["input"])
        self.assertEqual(request["operation"], "profile")
        self.assertNotIn("reference", request)
        self.assertNotIn("validation", request["kernel"])
        self.assertNotIn("profile", request["kernel"])
        self.assertEqual(command[command.index("--entrypoint") + 1], "/usr/local/cuda/bin/ncu")
        self.assertIn("/opt/klineage/venv/bin/python", command)
        self.assertEqual(kwargs["timeout"], 37)
        self.assertIn("--read-only", command)
        self.assertIn("--cap-drop", command)
        self.assertNotIn("--cap-add", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
        writable = [value for value in mounts if not value.endswith(",readonly")]
        self.assertEqual(len(writable), 2)
        self.assertEqual(result["tool"], "ncu")
        self.assertEqual(result["device"]["capability"], "sm120")
        self.assertEqual(len(result["metrics"]), 2)
        self.assertTrue(Path(result["report_path"]).is_file())
        self.assertTrue(Path(result["raw_path"]).is_file())
        self.assertIn("collected_at", result)

    def test_profile_outputs_are_fresh(self):
        with patch("subprocess.run", side_effect=self.complete):
            first = self.run_profile()
            second = self.run_profile()
        self.assertNotEqual(first["report_path"], second["report_path"])
        for result in (first, second):
            directory = Path(result["report_path"]).parent
            self.assertTrue((directory / "profile-request.json").is_file())
            self.assertTrue((directory / "profile-result.json").is_file())

    def test_counter_error_keeps_logs(self):
        def denied(command, **kwargs):
            raw = self.root / Path(command[command.index("--log-file") + 1]).relative_to(self.mount)
            raw.write_text("==ERROR== ERR_NVGPUCTRPERM: Permission denied\n")
            return subprocess.CompletedProcess(command, 1, "", "")

        with patch("subprocess.run", side_effect=denied):
            with self.assertRaisesRegex(ActionError, "ERR_NVGPUCTRPERM"):
                self.run_profile()
        self.assertTrue((self.root / "logs/profile-error.json").is_file())
        self.assertEqual(len(list((self.root / "logs").glob("ncu-*/profile-process.json"))), 1)

    def test_missing_report_fails(self):
        def missing(command, **kwargs):
            completed = self.complete(command, **kwargs)
            report = self.root / Path(command[command.index("--export") + 1]).relative_to(self.mount)
            report.unlink()
            return completed

        with patch("subprocess.run", side_effect=missing):
            with self.assertRaisesRegex(ActionError, "report"):
                self.run_profile()

    def test_empty_report_keeps_reason(self):
        def empty(command, **kwargs):
            completed = self.complete(command, **kwargs)
            raw = self.root / Path(command[command.index("--log-file") + 1]).relative_to(self.mount)
            report = self.root / Path(command[command.index("--export") + 1]).relative_to(self.mount)
            raw.write_text("==WARNING== No kernels were profiled.\n")
            report.unlink()
            return completed

        with patch("subprocess.run", side_effect=empty):
            with self.assertRaisesRegex(ActionError, "No kernels were profiled"):
                self.run_profile()

    def test_invalid_worker_metadata(self):
        def invalid(command, **kwargs):
            completed = self.complete(command, **kwargs)
            completed.stdout = "{}"
            return completed

        with patch("subprocess.run", side_effect=invalid):
            with self.assertRaisesRegex(ActionError, "metadata"):
                self.run_profile()

    def test_profile_build_includes(self):
        include = self.root / "include"
        include.mkdir()
        with patch("subprocess.run", side_effect=self.complete):
            self.run_profile(include_paths=(include,))
        command, kwargs = self.calls[0]
        request = json.loads(kwargs["input"])
        self.assertEqual(request["config"]["include_paths"], [str(self.mount / "include")])
        self.assertIn(f"type=bind,source={include},target={self.mount / 'include'},readonly", command)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is provided by the CUDA image")
class ProfileWorkerTests(unittest.TestCase):
    def test_only_target_is_in_range(self):
        from klineage.harness import _cuda_worker as worker

        events = []
        device = SimpleNamespace(type="cuda", index=0)
        tensor = SimpleNamespace(device=device)
        kernel = Kernel("target", "source", TargetContext("gemm", "cuda", "sm120"))
        module = SimpleNamespace(torch_ref=lambda *_: self.fail("reference was called"))

        @contextmanager
        def capture(name):
            self.assertEqual(name, "klineage_profile")
            events.append("start")
            yield
            events.append("stop")

        def build(*args):
            events.append("build")
            return lambda *inputs: events.append("target")

        with (
            patch.object(worker, "_config", return_value=SimpleNamespace(problem_path=Path("reference.py"), seed=1)),
            patch.object(worker, "_load_problem", return_value=(module, Path("reference.py"))),
            patch.object(worker, "_make_inputs", side_effect=lambda *args: events.append("inputs") or {"x": tensor}),
            patch.object(worker._RawLoader, "__init__", return_value=None),
            patch.object(worker._RawLoader, "load", side_effect=build),
            patch.object(worker, "_reference", side_effect=AssertionError("reference was called")),
            patch.object(worker, "FlashInferCuptiTimer", side_effect=AssertionError("CUPTI was called")),
            patch.object(worker.torch.cuda, "is_available", return_value=True),
            patch.object(worker.torch.cuda, "device", return_value=nullcontext()),
            patch.object(worker.torch.cuda, "synchronize", side_effect=lambda *args: events.append("sync")),
            patch.object(worker.torch.cuda.nvtx, "range", side_effect=capture),
            patch.object(worker.torch.cuda, "get_device_properties", return_value=SimpleNamespace(
                name="GPU", uuid="GPU-test", major=12, minor=0)),
            patch("subprocess.check_output", return_value="NCU 2025.3.1\n"),
        ):
            result = worker._profile_request({"kernel": kernel.to_dict(), "config": {}})

        start = events.index("start")
        self.assertIn("inputs", events[:start])
        self.assertIn("build", events[:start])
        self.assertEqual(events[start - 2:], ["target", "sync", "start", "target", "sync", "stop"])
        self.assertEqual(result["device"], {"name": "GPU", "uuid": "GPU-test", "capability": "sm120"})
        self.assertEqual(result["tool_version"], "NCU 2025.3.1")
