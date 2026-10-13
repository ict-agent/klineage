from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from kernel_fixtures import make_kernel
from problem_fixtures import problem_spec

from klineage.backend.ascend.kernel import AscendBackend, compile_native
from klineage.harness.sandbox import AgentRuntime, Sandbox, runtime_paths
from klineage.server.worker import WorkerRuntime

RUN_ASCEND = os.environ.get("KLINEAGE_ASCEND_TESTS") == "1"
SMOKE_TIMEOUT = 60
POLL_SECONDS = 0.05


@unittest.skipUnless(RUN_ASCEND, "set KLINEAGE_ASCEND_TESTS=1 on an Ascend device")
class AscendExecutionTests(unittest.TestCase):
    def test_sandbox_device(self):
        backend = AscendBackend()
        runtime = AgentRuntime.local(backend)
        worker = WorkerRuntime(
            python=runtime.python, device_nodes=runtime.device_nodes,
            read_only=runtime.read_only, environment=runtime.environment,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = worker.info(backend, build_root=root / "build", log_dir=root / "logs")
            self.assertTrue(info["available"])
            self.assertEqual(info["device_architecture"], backend.info()["device_architecture"])

    def test_sandbox_matmul(self):
        runtime = AgentRuntime.local(AscendBackend())
        source = """
import torch, torch_npu
SIDE = 16
assert torch.npu.device_count() == 1
value = torch.ones((SIDE, SIDE), device="npu")
torch.testing.assert_close((value @ value).cpu(), torch.full((SIDE, SIDE), float(SIDE)))
"""
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Sandbox(
                directory, devices=runtime.device_nodes,
                read_only=(*runtime_paths(runtime.python), *runtime.read_only),
                environment=runtime.environment,
            )
            result = subprocess.run(
                sandbox.command([str(runtime.python), "-c", source]),
                env=sandbox.environment, capture_output=True, text=True,
                timeout=SMOKE_TIMEOUT, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker_with_live_agent(self):
        backend = AscendBackend()
        runtime = AgentRuntime.local(backend)
        agent = runtime.for_agent(backend)
        worker = WorkerRuntime(
            python=runtime.python, device_nodes=runtime.device_nodes,
            read_only=runtime.read_only, environment=runtime.environment,
            timeout=SMOKE_TIMEOUT,
        )
        problem = problem_spec(language="python", platform="ascend", size=16)
        kernel = make_kernel("smoke", problem, {"kernel.py": "def run(a, b): return a @ b\n"})
        source = """
from pathlib import Path
import torch, torch_npu
Path('ready').write_text(str(torch.npu.is_available()))
input()
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sandbox = Sandbox(
                root, devices=agent.device_nodes,
                read_only=(*runtime_paths(agent.python), *agent.read_only),
                environment=agent.environment,
            )
            with (root / "agent.log").open("w+") as log:
                process = subprocess.Popen(
                    sandbox.command([str(agent.python), "-c", source]),
                    env=sandbox.environment, stdin=subprocess.PIPE,
                    stdout=log, stderr=log, text=True,
                )
                try:
                    deadline = time.monotonic() + SMOKE_TIMEOUT
                    while not (root / "ready").exists() and time.monotonic() < deadline:
                        time.sleep(POLL_SECONDS)
                    self.assertEqual((root / "ready").read_text(), "False")
                    result = worker.evaluate(
                        kernel, include_paths=(), build_root=root / "build",
                        log_dir=root / "logs",
                    )
                    diagnostics = "\n".join(path.read_text() for path in (root / "logs").rglob("*.log"))
                    self.assertTrue(result.compile_passed, diagnostics)
                    self.assertTrue(result.correctness_passed, diagnostics)
                    self.assertGreater(result.latency_ms or 0, 0)
                    self.assertGreater(result.reference_latency_ms or 0, 0)
                finally:
                    process.communicate("\n", timeout=SMOKE_TIMEOUT)

    def test_native_compile(self):
        # Exercise the installed compiler and linker, not a mocked subprocess.
        source = '''
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "kernel_operator.h"

constexpr uint32_t ELEMENTS = 32;

__global__ __aicore__ void fill_device(GM_ADDR output) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, ELEMENTS * sizeof(float));
    auto local = buffer.Get<float>();
    AscendC::Duplicate(local, 1.0f, ELEMENTS);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<float> global;
    global.SetGlobalBuffer((__gm__ float*)output, ELEMENTS);
    AscendC::DataCopy(global, local, ELEMENTS);
}

void fill(torch::Tensor output) {
    c10::DeviceGuard guard(output.device());
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    fill_device<<<1, nullptr, stream>>>((GM_ADDR)output.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("fill", &fill);
}
'''
        import torch

        with tempfile.TemporaryDirectory() as directory:
            module = compile_native({"kernel.cpp": source}, Path(directory))
            value = torch.full((32,), float("nan"), device="npu")
            module.fill(value)
            torch.testing.assert_close(value, torch.ones_like(value))
