"""Record the actual CUDA runtime; unavailable probes stay explicit."""

import subprocess
from pathlib import Path
from typing import Any

_PROBE_TIMEOUT_SECONDS = 10
_GPU_UUID_PREFIX = "GPU-"


def _runtime_info(torch: Any, cuda_home: str | None) -> dict[str, Any]:
    result = {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}
    try:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        result["gpu"] = {
            "name": properties.name, "uuid": str(properties.uuid),
            "compute_capability": [properties.major, properties.minor],
        }
    except Exception as exc:
        result["gpu"] = {"error": str(exc)}

    nvcc = str(Path(cuda_home) / "bin/nvcc") if cuda_home else "nvcc"
    result["nvcc"] = _probe((nvcc, "--version"))
    driver = ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
    uuid = result["gpu"].get("uuid")
    if uuid:
        if not uuid.startswith((_GPU_UUID_PREFIX, "MIG-")):
            uuid = _GPU_UUID_PREFIX + uuid
        driver += (f"--id={uuid}",)
    result["driver"] = _probe(driver)
    return result


def _probe(command: tuple[str, ...]) -> dict[str, str]:
    try:
        output = subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, timeout=_PROBE_TIMEOUT_SECONDS,
        ).strip()
        return {"output": output} if output else {"error": "probe returned no output"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}
