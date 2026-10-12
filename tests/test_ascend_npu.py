"""Opt-in Ascend checks; KLINEAGE_ASCEND_DEVICE selects a physical sandbox device."""

import os
import tempfile
import unittest
from pathlib import Path

from klineage.backend.ascend.kernel import AscendBackend, compile_native
from klineage.harness.sandbox import AgentRuntime
from klineage.server.worker import WorkerRuntime


RUN_ASCEND = os.environ.get("KLINEAGE_ASCEND_TESTS") == "1"
TEST_DEVICE = os.environ.get("KLINEAGE_ASCEND_DEVICE")


@unittest.skipUnless(RUN_ASCEND, "set KLINEAGE_ASCEND_TESTS=1 on an Ascend device")
class AscendExecutionTests(unittest.TestCase):
    def test_sandbox_device(self):
        backend = AscendBackend()
        runtime = AgentRuntime.local(backend)
        devices = runtime.device_nodes
        if TEST_DEVICE is not None:
            devices = tuple(
                path for path in devices
                if not path.name.removeprefix("davinci").isdigit()
                or path.name == f"davinci{TEST_DEVICE}"
            )
        worker = WorkerRuntime(
            python=runtime.python, device_nodes=devices,
            read_only=runtime.read_only, environment=runtime.environment,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = worker.info(backend, build_root=root / "build", log_dir=root / "logs")
            self.assertTrue(info["available"])
            self.assertEqual(info["device_architecture"], backend.info()["device_architecture"])

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
