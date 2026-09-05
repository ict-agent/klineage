from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from klineage.harness._docker import _container_config, _DockerError, _DockerRuntime
from klineage.harness.eval import ValidationResult
from klineage.kernel import Kernel, TargetContext


class DockerDriverTests(unittest.TestCase):
    def test_transport_omits_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            problem = root / "reference.py"
            artifact = root / "kernel.cu"
            problem.write_text("", encoding="utf-8")
            artifact.write_text("__global__ void kernel() {}", encoding="utf-8")
            mount = Path("/workspace/klineage")
            runtime = _DockerRuntime(
                root=root,
                docker="docker",
                image="klineage-image",
                mount=mount,
            )
            validation = ValidationResult(True, True, True, 1.0, 2.0)
            kernel = Kernel(
                "kernel",
                artifact.read_text(),
                TargetContext("example", "cuda", "sm120"),
                artifact_path=artifact,
                validation=validation,
            )
            evaluator = runtime.evaluator(
                problem,
                include_paths=(),
                build_root=root / "build",
                log_dir=root / "logs",
            )

            with patch.object(
                runtime, "_invoke", return_value=validation.to_dict()
            ) as invoke:
                result = evaluator.evaluate(kernel, reference=kernel)

            request = invoke.call_args.args[0]
            for key in ("kernel", "reference"):
                payload = request[key]
                self.assertNotIn("validation", payload)
                restored = Kernel.from_dict(payload)
                self.assertEqual(restored.source, kernel.source)
                self.assertEqual(restored.context, kernel.context)
                self.assertEqual(restored.artifact_path, mount / artifact.name)
            self.assertNotIn("verbose_build", request["config"])
            self.assertEqual(result, validation)
            self.assertEqual(kernel.validation, validation)

    def test_prepared_runtime_pins_the_inspected_image(self) -> None:
        root = Path("/workspace/source")
        digest = "sha256:" + "a" * 64
        inspection = json.dumps(
            [
                {
                    "Mounts": [
                        {
                            "Source": str(root),
                            "Destination": "/workspace/klineage",
                        }
                    ],
                    "State": {"Running": True},
                    "Config": {"Image": "klineage:latest"},
                    "Image": digest,
                }
            ]
        )

        mount, image = _container_config(inspection, root)

        self.assertEqual(mount, Path("/workspace/klineage"))
        self.assertEqual(image, digest)

    def test_worker_runs_offline_with_read_only_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            logs = root / "action" / "evaluations"
            (root / "src").mkdir()
            logs.mkdir(parents=True)
            runtime = _DockerRuntime(
                root=root,
                docker="docker",
                image="klineage-image",
                mount=Path("/workspace/klineage"),
            )
            completed = subprocess.CompletedProcess(
                (),
                0,
                stdout=json.dumps({"done": True}),
                stderr="",
            )

            with patch("subprocess.run", return_value=completed) as run:
                runtime._invoke({"operation": "inspect"}, logs, "inspect")

        command = run.call_args.args[0]
        self.assertEqual(command[:2], ("docker", "run"))
        self.assertIn("--network", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("--read-only", command)
        self.assertIn("--cap-drop", command)
        self.assertIn("no-new-privileges", command)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", command)
        self.assertIn("FLASHINFER_WORKSPACE_BASE=/tmp", command)
        self.assertIn("klineage-image", command)
        self.assertNotIn("exec", command)

    def test_worker_remaps_selected_gpu_to_container_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            logs = root / "action" / "evaluations"
            (root / "src").mkdir()
            logs.mkdir(parents=True)
            runtime = _DockerRuntime(
                root=root,
                docker="docker",
                image="klineage-image",
                mount=Path("/workspace/klineage"),
                gpu=2,
            )
            completed = subprocess.CompletedProcess(
                (),
                0,
                stdout=json.dumps({"done": True}),
                stderr="",
            )

            with patch("subprocess.run", return_value=completed) as run:
                runtime._invoke({"operation": "inspect"}, logs, "inspect")

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--gpus") + 1], "device=2")
        self.assertIn("CUDA_VISIBLE_DEVICES=0", command)

    def test_worker_mounts_only_declared_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "src"
            trusted = root / "run" / "input"
            logs = root / "run" / "evaluations"
            source.mkdir()
            trusted.mkdir(parents=True)
            logs.mkdir()
            (root / ".env").write_text("SECRET=value\n", encoding="utf-8")
            runtime = _DockerRuntime(
                root=root,
                docker="docker",
                image="klineage-image",
                mount=Path("/workspace/klineage"),
            )
            completed = subprocess.CompletedProcess(
                (),
                0,
                stdout=json.dumps({"done": True}),
                stderr="",
            )

            with patch("subprocess.run", return_value=completed) as run:
                runtime._invoke(
                    {"operation": "inspect"},
                    logs,
                    "inspect",
                    read_paths=(trusted,),
                )

        command = run.call_args.args[0]
        mounts = [
            command[index + 1]
            for index, item in enumerate(command)
            if item == "--mount"
        ]
        self.assertEqual(
            mounts,
            [
                (
                    f"type=bind,source={source},"
                    "target=/workspace/klineage/src,readonly"
                ),
                (
                    f"type=bind,source={trusted},"
                    "target=/workspace/klineage/run/input,readonly"
                ),
            ],
        )
        self.assertNotIn(f"source={root},target=/workspace/klineage", " ".join(mounts))

    def test_build_cache_is_the_only_writable_project_mount(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            logs = root / "action" / "evaluations"
            build = root / "action" / "build"
            (root / "src").mkdir()
            logs.mkdir(parents=True)
            build.mkdir()
            mount = Path("/workspace/klineage")
            runtime = _DockerRuntime(
                root=root,
                docker="docker",
                image="klineage-image",
                mount=mount,
            )
            completed = subprocess.CompletedProcess(
                (),
                0,
                stdout=json.dumps({"done": True}),
                stderr="",
            )
            request = {
                "operation": "evaluate",
                "config": {"build_root": str(mount / "action/build")},
            }

            with patch("subprocess.run", return_value=completed) as run:
                runtime._invoke(request, logs, "evaluate")

        command = run.call_args.args[0]
        self.assertIn(
            f"TORCH_EXTENSIONS_DIR={mount / 'action/build'}",
            command,
        )
        mounts = [
            command[index + 1]
            for index, item in enumerate(command)
            if item == "--mount"
        ]
        self.assertEqual(len(mounts), 2)
        self.assertIn("readonly", mounts[0])
        self.assertNotIn("readonly", mounts[1])

    def test_timeout_forces_container_cleanup(self) -> None:
        container_id = "a" * 64
        calls: list[tuple[str, ...]] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            logs = root / "evaluations"
            (root / "src").mkdir()
            logs.mkdir()
            runtime = _DockerRuntime(
                root=root,
                docker="docker",
                image="klineage-image",
                mount=Path("/workspace/klineage"),
                timeout=1.0,
            )

            def run(command: tuple[str, ...], **_: object):
                calls.append(command)
                if command[:2] != ("docker", "run"):
                    return subprocess.CompletedProcess(command, 0, "", "")
                cidfile = Path(command[command.index("--cidfile") + 1])
                cidfile.write_text(container_id, encoding="utf-8")
                raise subprocess.TimeoutExpired(command, 1.0)

            with (
                patch("subprocess.run", side_effect=run),
                self.assertRaisesRegex(_DockerError, "timed out"),
            ):
                runtime._invoke({"operation": "inspect"}, logs, "inspect")

            self.assertFalse((logs / ".inspect.cid").exists())

        self.assertIn(("docker", "rm", "--force", container_id), calls)


if __name__ == "__main__":
    unittest.main()
