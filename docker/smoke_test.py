from __future__ import annotations

import json
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import torch
from cupti import cupti
from flashinfer.testing import bench_gpu_time_with_cupti

import klineage


def version(command: str, *arguments: str) -> str:
    executable = shutil.which(command)
    if executable is None:
        raise RuntimeError(f"missing required executable: {command}")
    completed = subprocess.run(
        (executable, *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.strip().splitlines()[-1]


workspace = Path("/workspace/klineage").resolve(strict=True)
package_source = Path(klineage.__file__).resolve(strict=True)

result = {
    "workspace": str(workspace),
    "workspace_is_current_directory": workspace == Path.cwd().resolve(),
    "klineage_source": str(package_source),
    "python": sys.version.split()[0],
    "python_prefix": sys.prefix,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "flashinfer": metadata.version("flashinfer-python"),
    "cupti": metadata.version("cupti-python"),
    "cupti_module": cupti.__name__,
    "cupti_timer": bench_gpu_time_with_cupti.__name__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "nvcc": version("nvcc", "--version"),
    "codex": version("codex", "--version"),
    "uv": version("uv", "--version"),
    "rg": version("rg", "--version"),
}

print(json.dumps(result, indent=2))

if not result["workspace_is_current_directory"]:
    raise SystemExit("the repository bind mount is not the working directory")
if not package_source.is_relative_to(workspace):
    raise SystemExit("klineage is not imported from the bind-mounted source tree")
if not result["cuda_available"]:
    raise SystemExit("PyTorch cannot access CUDA")
