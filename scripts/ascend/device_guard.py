"""Reject device contention before starting a model's experiment budget."""
import re
import subprocess


def busy_devices(output: str) -> set[int]:
    header = re.search(r'Process\s+id', output, re.IGNORECASE)
    if not header:
        raise RuntimeError('Cannot parse npu-smi process table; refusing unchecked launch.')
    table = output[header.start():]
    return {int(match[0]) for match in re.findall(r'\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|', table)}


def require_idle(output: str, devices: list[int]) -> None:
    busy = busy_devices(output).intersection(devices)
    if busy:
        raise RuntimeError(f'NPU devices {sorted(busy)} have existing processes. No model sessions started.')


def check_devices(host: str, devices: list[str]) -> None:
    result = subprocess.run(['ssh', host, 'npu-smi info'], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('Device preflight failed; check SSH and npu-smi.')
    require_idle(result.stdout, [int(device) for device in devices])
