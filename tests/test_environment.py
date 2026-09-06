import unittest
from types import SimpleNamespace
from unittest.mock import patch

from klineage.harness._environment import _runtime_info


class EnvironmentTests(unittest.TestCase):
    def test_records_actual_runtime(self):
        properties = SimpleNamespace(name="H100", uuid="test", major=9, minor=0)
        torch = SimpleNamespace(
            __version__="2.11.0", version=SimpleNamespace(cuda="13.0"),
            cuda=SimpleNamespace(current_device=lambda: 0, get_device_properties=lambda _: properties),
        )
        with patch("klineage.harness._environment.subprocess.check_output", side_effect=[
            "Cuda compilation tools, release 13.0", "580.95.05",
        ]) as command:
            result = _runtime_info(torch, "/usr/local/cuda")
        self.assertEqual(result["gpu"]["uuid"], "test")
        self.assertEqual(result["gpu"]["name"], "H100")
        self.assertEqual(result["nvcc"]["output"], "Cuda compilation tools, release 13.0")
        self.assertEqual(result["driver"]["output"], "580.95.05")
        self.assertEqual(command.call_args_list[0].args[0][0], "/usr/local/cuda/bin/nvcc")
        self.assertIn("--id=GPU-test", command.call_args_list[1].args[0])

    def test_reports_missing_metadata(self):
        torch = SimpleNamespace(
            __version__="test", version=SimpleNamespace(cuda=None),
            cuda=SimpleNamespace(current_device=lambda: 0,
                                 get_device_properties=lambda _: (_ for _ in ()).throw(RuntimeError("no GPU"))),
        )
        with patch("klineage.harness._environment.subprocess.check_output", side_effect=FileNotFoundError("missing")):
            result = _runtime_info(torch, None)
        self.assertIn("error", result["gpu"])
        self.assertIn("error", result["nvcc"])
        self.assertNotIn("uuid", result["gpu"])


if __name__ == "__main__":
    unittest.main()
