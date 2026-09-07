"""Capture the current CUDA kernel with Nsight Compute."""

from dataclasses import replace

from klineage.action._sandbox import _Kind, _next, _Sandbox
from klineage.errors import ActionError
from klineage.kernel import Kernel
from klineage.profiling import KernelProfile, ProfileOptions


def profile(kernel: Kernel, *, options: ProfileOptions = ProfileOptions()) -> Kernel:
    """Capture fresh diagnostics; preserve CUPTI validation unchanged."""
    with _next(_Kind.PROFILE, kernel) as sandbox:
        return _profile(sandbox, kernel, options)


def _profile(
    sandbox: _Sandbox, kernel: Kernel, options: ProfileOptions | None = None,
) -> Kernel:
    if options is None and kernel.profile is not None and kernel.profile.matches(kernel):
        return kernel
    options = options or ProfileOptions()
    if not isinstance(options, ProfileOptions):
        raise TypeError("options must be ProfileOptions")
    target = kernel.context.to_dict()
    target.pop("prior_actions")
    result = sandbox._profile(kernel, options)
    evidence = KernelProfile(
        kernel_fingerprint=kernel.fingerprint, target=target, options=options, **result,
    )
    if evidence.device.get("capability") != kernel.context.platform:
        raise ActionError("profile device does not match the kernel target")
    return replace(kernel, profile=evidence)


__all__ = ["profile"]
