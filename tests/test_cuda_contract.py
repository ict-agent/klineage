from __future__ import annotations

import unittest

from problem_fixtures import problem_spec

from klineage import contract
from klineage.kernel import Kernel

_CONFIG = """[solution]
name = "gemm"
definition = "bf16 GEMM"
author = "klineage"

[build]
language = "cuda"
entry_point = "kernel.cu::run"
destination_passing_style = true
"""


def source_bundle(config: str = _CONFIG) -> dict[str, str]:
    return {"config.toml": config, "solution/kernel.cu": "void run() {}"}


def kernel_metadata(sources):
    return Kernel("test", problem_spec(), source_files=sources)


class CudaContractTests(unittest.TestCase):
    def test_cuda_requires_native(self):
        sources = {
            "config.toml": _CONFIG.replace("kernel.cu::run", "binding.py::run"),
            "solution/binding.py": "def run(): pass",
            "solution/kernel.cu": "void run() {}",
        }
        with self.assertRaises(ValueError):
            kernel_metadata(sources)

    def test_reads_cuda_build(self):
        build = kernel_metadata(source_bundle())

        self.assertEqual(build.language, "cuda")
        self.assertEqual(build.entry_point, "kernel.cu::run")
        self.assertEqual(build.source_path, "solution/kernel.cu")
        self.assertEqual(build.symbol, "run")
        self.assertIs(build.output_style, contract.OutputStyle.DESTINATION)
        self.assertEqual(
            kernel_metadata(source_bundle()).source_path,
            "solution/kernel.cu",
        )

    def test_python_entry_and_defaults(self):
        config = _CONFIG.replace("kernel.cu::run", "nested/kernel.py::run").replace(
            'language = "cuda"', 'language = "python"'
        )
        config = config.replace("destination_passing_style = true\n", "")
        sources = {
            "config.toml": config,
            "solution/nested/kernel.py": "def run(): pass",
        }
        build = kernel_metadata(sources)

        self.assertEqual(build.source_path, "solution/nested/kernel.py")
        self.assertEqual(build.language, "python")
        self.assertIs(build.output_style, contract.OutputStyle.RETURN)
        sources["config.toml"] = config.replace(
            'language = "cuda"', 'language = "python"'
        )
        self.assertEqual(kernel_metadata(sources).language, "python")

    def test_cuda_return_style(self):
        for config in (
            _CONFIG.replace(
                "destination_passing_style = true", "destination_passing_style = false"
            ),
            _CONFIG.replace("destination_passing_style = true\n", ""),
        ):
            with self.subTest(config=config):
                build = kernel_metadata(source_bundle(config))
                self.assertIs(build.output_style, contract.OutputStyle.RETURN)

    def test_accepts_additional_fields(self):
        sources = source_bundle(_CONFIG + "\n[benchmark]\nrepeat = 1\n")
        self.assertEqual(kernel_metadata(sources).symbol, "run")

    def test_rejects_invalid_config(self):
        for config in (
            "[solution",
            "",
            _CONFIG.replace("[solution]", "[other]"),
            "solution = []\n" + _CONFIG[_CONFIG.index("[build]") :],
            _CONFIG.replace("[build]", "[other]"),
            _CONFIG.replace('name = "gemm"', "name = 1"),
            _CONFIG.replace('definition = "bf16 GEMM"', 'definition = " "'),
            _CONFIG.replace('author = "klineage"', "author = []"),
            _CONFIG.replace('author = "klineage"\n', ""),
            _CONFIG.replace('language = "cuda"', 'language = "triton"'),
            _CONFIG.replace('language = "cuda"', 'language = "python"'),
            _CONFIG.replace('language = "cuda"', "language = true"),
            _CONFIG.replace('entry_point = "kernel.cu::run"', "entry_point = 1"),
            _CONFIG.replace(
                "destination_passing_style = true", "destination_passing_style = 1"
            ),
            _CONFIG.replace(
                "destination_passing_style = true",
                'destination_passing_style = "false"',
            ),
        ):
            with (
                self.subTest(config=config),
                self.assertRaises((ValueError, TypeError, KeyError)),
            ):
                kernel_metadata(source_bundle(config))

    def test_rejects_bad_entry_paths(self):
        for entry in (
            "../kernel.cu::run",
            "/kernel.cu::run",
            "nested/../kernel.cu::run",
            "C:/kernel.cu::run",
            " kernel.cu::run",
            "kernel.cu::run ",
            "./kernel.cu::run",
            "nested//kernel.cu::run",
            "kernel.cu",
            "kernel.txt::run",
            "kernel.cu::",
            "kernel.cu::nested::run",
            "kernel.cu::bad-name",
            "kernel.cu::run()",
            "kernel.py::class",
        ):
            with (
                self.subTest(entry=entry),
                self.assertRaises((ValueError, TypeError, KeyError)),
            ):
                config = _CONFIG.replace("kernel.cu::run", entry)
                sources = source_bundle(config)
                sources["solution/" + entry.partition("::")[0]] = "entry"
                kernel_metadata(sources)

    def test_requires_entry_source(self):
        with self.assertRaises((ValueError, KeyError)):
            kernel_metadata({"config.toml": _CONFIG})

    def test_requires_nonempty_entry(self):
        for source in ("", " \n", None, 1):
            with (
                self.subTest(source=source),
                self.assertRaises((ValueError, TypeError)),
            ):
                sources = source_bundle()
                sources["solution/kernel.cu"] = source
                kernel_metadata(sources)

    def test_rejects_mixed_layout(self):
        for path in (
            "submission.py",
            "kernel.cu",
            "solution",
            "other/kernel.cu",
            "solution/../outside.cu",
            "solution//extra.cu",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                sources = source_bundle()
                sources[path] = "source"
                kernel_metadata(sources)

    def test_bad_config_cannot_fallback(self):
        sources = {"config.toml": "[invalid", "submission.py": "def load(): pass"}
        with self.assertRaises(ValueError):
            kernel_metadata(sources)

    def test_aux_config_needs_entry(self):
        sources = source_bundle("block_size = 128\n")
        with self.assertRaises(KeyError):
            kernel_metadata(sources)


if __name__ == "__main__":
    unittest.main()
