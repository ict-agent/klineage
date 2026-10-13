from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from klineage.backend import get_backend
from klineage.harness import AgentRuntime

SELECTED_DEVICE = 6
LOGICAL_DEVICE = 1
ACL_ERROR = 107001
VISIBLE_ENV = "ASCEND_RT_VISIBLE_DEVICES"
CONTROL_NODES = ("davinci_manager", "devmm_svm", "hisi_hdc")


class AscendDeviceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.backend = get_backend("ascendc")
        self.enterContext(patch.dict(os.environ, {VISIBLE_ENV: "3,6"}, clear=True))
        self.enterContext(patch("klineage.backend.runtime.DEVICE_ROOT", self.root))
        self.enterContext(patch.object(type(self.backend), "info", return_value={"available": True}))
        npu = SimpleNamespace(current_device=lambda: LOGICAL_DEVICE)
        self.enterContext(patch.dict(sys.modules, {"torch_npu": SimpleNamespace(npu=npu)}))
        self.resolve = Mock(side_effect=self.physical_id)
        library = SimpleNamespace(aclrtGetPhyDevIdByLogicDevId=self.resolve)
        self.loader = self.enterContext(patch("ctypes.CDLL", return_value=library))
        for name in ("davinci0", "davinci3", "davinci6", *CONTROL_NODES):
            (self.root / name).symlink_to("/dev/null")

    def physical_id(self, logical, output):
        self.assertEqual(logical, LOGICAL_DEVICE)
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int32))[0] = SELECTED_DEVICE
        return 0

    def test_select_current_device(self):
        runtime = AgentRuntime.local(self.backend)
        expected = {self.root / name for name in ("davinci6", *CONTROL_NODES)}
        self.assertEqual(set(runtime.device_nodes), expected)
        self.assertEqual(runtime.environment[VISIBLE_ENV], "0")
        self.assertEqual(os.environ[VISIBLE_ENV], "3,6")

    def test_select_without_env(self):
        os.environ.pop(VISIBLE_ENV)
        runtime = AgentRuntime.local(self.backend)
        self.assertNotIn(self.root / "davinci0", runtime.device_nodes)
        self.assertIn(self.root / "davinci6", runtime.device_nodes)
        self.assertEqual(runtime.environment[VISIBLE_ENV], "0")

    def test_missing_device(self):
        (self.root / "davinci6").unlink()
        with self.assertRaisesRegex(RuntimeError, "davinci6"):
            AgentRuntime.local(self.backend)

    def test_mapping_failure(self):
        self.resolve.side_effect = None
        self.resolve.return_value = ACL_ERROR
        with self.assertRaisesRegex(RuntimeError, str(ACL_ERROR)):
            AgentRuntime.local(self.backend)

    def test_missing_driver(self):
        self.loader.side_effect = OSError("missing CANN")
        with self.assertRaisesRegex(RuntimeError, "CANN"):
            AgentRuntime.local(self.backend)
