"""Map the active CANN device into an isolated runtime."""

import ctypes

ACL_LIBRARY = "libascendcl.so"
ACL_SUCCESS = 0
DEVICE_PREFIX = "davinci"
VISIBLE_ENV = "ASCEND_RT_VISIBLE_DEVICES"
SANDBOX_DEVICE = "0"


def physical_device() -> int:
    """Resolve the current logical device after CANN visibility remapping."""
    from torch_npu import npu

    try:
        library = ctypes.CDLL(ACL_LIBRARY)
        resolve = library.aclrtGetPhyDevIdByLogicDevId
    except (OSError, AttributeError) as error:
        raise RuntimeError("CANN device mapping is unavailable") from error

    resolve.argtypes = (ctypes.c_int32, ctypes.POINTER(ctypes.c_int32))
    resolve.restype = ctypes.c_int
    physical = ctypes.c_int32()
    status = resolve(npu.current_device(), ctypes.byref(physical))
    if status != ACL_SUCCESS:
        raise RuntimeError(f"CANN device mapping failed: {status}")
    return physical.value


def select_device(nodes, environment):
    """Expose one physical NPU and retain its shared control devices."""
    selected = f"{DEVICE_PREFIX}{physical_device()}"
    if not any(path.name == selected for path in nodes):
        raise RuntimeError(f"Ascend device node is unavailable: {selected}")

    # Enumerating unrelated cards can fail with EBUSY in a shared deployment.
    devices = tuple(
        path for path in nodes
        if path.name == selected or not path.name.removeprefix(DEVICE_PREFIX).isdigit()
    )
    return devices, {**environment, VISIBLE_ENV: SANDBOX_DEVICE}
