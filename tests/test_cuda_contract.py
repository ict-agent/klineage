from __future__ import annotations

import unittest

from klineage import contract


_CONFIG = '''[solution]
name = "gemm"
definition = "bf16 GEMM"
author = "klineage"

[build]
language = "cuda"
entry_point = "kernel.cu::run"
destination_passing_style = true
'''


def _sources(config: str = _CONFIG) -> dict[str, str]:
    return {"config.toml": config, "solution/kernel.cu": "void run() {}"}


class CudaContractTests(unittest.TestCase):
    def test_reads_cuda_build(self):
        build = contract.CudaBuild.from_sources(_sources())

        self.assertEqual(build.language, "cuda")
        self.assertEqual(build.entry_point, "kernel.cu::run")
        self.assertEqual(build.source_path, "solution/kernel.cu")
        self.assertEqual(build.symbol, "run")
        self.assertIs(build.output_style, contract.OutputStyle.DESTINATION)
        self.assertEqual(
            contract.source_entry(_sources(), contract.EvaluatorInterface()),
            "solution/kernel.cu",
        )

    def test_python_entry_and_defaults(self):
        config = _CONFIG.replace("kernel.cu::run", "nested/kernel.py::run")
        config = config.replace("destination_passing_style = true\n", "")
        sources = {"config.toml": config, "solution/nested/kernel.py": "def run(): pass"}
        build = contract.CudaBuild.from_sources(sources)

        self.assertEqual(build.source_path, "solution/nested/kernel.py")
        self.assertEqual(build.language, "cuda")
        self.assertIs(build.output_style, contract.OutputStyle.RETURN)
        sources["config.toml"] = config.replace('language = "cuda"', 'language = "python"')
        self.assertEqual(contract.CudaBuild.from_sources(sources).language, "python")

    def test_cuda_return_style(self):
        for config in (
            _CONFIG.replace("destination_passing_style = true", "destination_passing_style = false"),
            _CONFIG.replace("destination_passing_style = true\n", ""),
        ):
            with self.subTest(config=config):
                build = contract.CudaBuild.from_sources(_sources(config))
                self.assertIs(build.output_style, contract.OutputStyle.RETURN)

    def test_accepts_additional_fields(self):
        sources = _sources(_CONFIG + '\n[benchmark]\nrepeat = 1\n')
        self.assertEqual(contract.CudaBuild.from_sources(sources).symbol, "run")

    def test_rejects_invalid_config(self):
        for config in (
            "[solution", "", _CONFIG.replace('[solution]', '[other]'),
            'solution = []\n' + _CONFIG[_CONFIG.index('[build]'):],
            _CONFIG.replace('[build]', '[other]'),
            _CONFIG.replace('name = "gemm"', 'name = 1'),
            _CONFIG.replace('definition = "bf16 GEMM"', 'definition = " "'),
            _CONFIG.replace('author = "klineage"', 'author = []'),
            _CONFIG.replace('author = "klineage"\n', ''),
            _CONFIG.replace('language = "cuda"', 'language = "triton"'),
            _CONFIG.replace('language = "cuda"', 'language = "python"'),
            _CONFIG.replace('language = "cuda"', 'language = true'),
            _CONFIG.replace('entry_point = "kernel.cu::run"', 'entry_point = 1'),
            _CONFIG.replace('destination_passing_style = true', 'destination_passing_style = 1'),
            _CONFIG.replace('destination_passing_style = true', 'destination_passing_style = "false"'),
        ):
            with self.subTest(config=config), self.assertRaises((ValueError, TypeError, KeyError)):
                contract.CudaBuild.from_sources(_sources(config))

    def test_rejects_bad_entry_paths(self):
        for entry in (
            "../kernel.cu::run", "/kernel.cu::run", "nested/../kernel.cu::run",
            "C:/kernel.cu::run", " kernel.cu::run", "kernel.cu::run ",
            "./kernel.cu::run", "nested//kernel.cu::run", "kernel.cu",
            "kernel.txt::run", "kernel.cu::", "kernel.cu::nested::run",
            "kernel.cu::bad-name", "kernel.cu::run()", "kernel.py::class",
        ):
            with self.subTest(entry=entry), self.assertRaises((ValueError, TypeError, KeyError)):
                config = _CONFIG.replace("kernel.cu::run", entry)
                sources = _sources(config)
                sources["solution/" + entry.partition("::")[0]] = "entry"
                contract.CudaBuild.from_sources(sources)

    def test_requires_entry_source(self):
        with self.assertRaises((ValueError, KeyError)):
            contract.CudaBuild.from_sources({"config.toml": _CONFIG})

    def test_requires_config(self):
        for sources in ({}, {"submission.py": "def load(): pass"}):
            with self.subTest(sources=sources), self.assertRaises(KeyError):
                contract.CudaBuild.from_sources(sources)

    def test_requires_nonempty_entry(self):
        for source in ("", " \n", None, 1):
            with self.subTest(source=source), self.assertRaises(ValueError):
                sources = _sources()
                sources["solution/kernel.cu"] = source
                contract.CudaBuild.from_sources(sources)

    def test_rejects_mixed_layout(self):
        for path in (
            "submission.py", "kernel.cu", "solution", "other/kernel.cu",
            "solution/../outside.cu", "solution//extra.cu",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                sources = _sources()
                sources[path] = "source"
                contract.CudaBuild.from_sources(sources)

    def test_rejects_language_types(self):
        for language in ([], {}):
            with self.subTest(language=language), self.assertRaises(TypeError):
                contract.CudaBuild(language, "kernel.cu::run")

    def test_requires_output_enum(self):
        with self.assertRaises(TypeError):
            contract.CudaBuild("cuda", "kernel.cu::run", False)

    def test_bad_config_cannot_fallback(self):
        sources = {"config.toml": "[invalid", "submission.py": "def load(): pass"}
        with self.assertRaises(ValueError):
            contract.source_entry(sources, contract.EvaluatorInterface())

    def test_aux_config_legacy(self):
        for module in ("submission.py", "nested/submission.py"):
            with self.subTest(module=module):
                interface = contract.EvaluatorInterface(module)
                sources = {"config.toml": "block_size = 128\n", module: "def load(): pass"}
                self.assertEqual(contract.source_entry(sources, interface), module)
                self.assertIsNone(contract.bundle_build(sources, interface))
                with self.assertRaises(ValueError):
                    contract.CudaBuild.from_sources(sources)

    def test_aux_config_needs_entry(self):
        sources = _sources("block_size = 128\n")
        with self.assertRaises(KeyError):
            contract.source_entry(sources, contract.EvaluatorInterface())

    def test_ako_entry_matches_module(self):
        config = _CONFIG.replace("kernel.cu::run", "binding.py::run")
        interface = contract.EvaluatorInterface("solution/binding.py")
        sources = {"config.toml": config, interface.module: "def run(): pass"}

        build = contract.bundle_build(sources, interface)
        self.assertEqual(build.entry_point, "binding.py::run")
        self.assertEqual(contract.source_entry(sources, interface), interface.module)

        for missing in ("solution", "build"):
            with self.subTest(missing=missing), self.assertRaises(KeyError):
                sources["config.toml"] = config.replace(f"[{missing}]", "[other]")
                contract.source_entry(sources, interface)

    def test_legacy_entry_unchanged(self):
        interface = contract.EvaluatorInterface("nested/submission.py", "load")
        self.assertEqual(contract.source_entry({}, interface), "nested/submission.py")
        self.assertEqual(interface.to_dict(), {
            "protocol": "python-callable-v1", "module": "nested/submission.py", "loader": "load",
        })
        self.assertEqual(contract.KernelABI().interface, contract.EvaluatorInterface())


if __name__ == "__main__":
    unittest.main()
