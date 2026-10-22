"""Explicit TLE Cube/Vector scopes for the paper workload."""

import torch
import torch_npu
import triton
import triton.language as tl
import triton.experimental.tle as tle
import triton.extension.buffer.language as bl

HEIGHT = 56
WIDTH = 56
CHANNELS = 64
FILTERS = 128
KERNEL = 3
PAD = 1
PITCH = 64
TILE_M = 256
CUBE_WORKERS = 24
COPY_ROWS = 4
POINTER_ALIGNMENT = 16
# External imports may set TRITON_ALL_BLOCKS_PARALLEL; our grids are already explicit.
GRID_OPTIONS = {"enable_auto_blockify": False}
_PLANS = {}


@triton.jit
def _prepare(X, P, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
             PW: tl.constexpr, R: tl.constexpr, KS: tl.constexpr):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    j = tl.arange(0, PW * C)
    with tle.scope(core_mode="vector"):
        if tile < H // R:
            row = tile * R + tl.arange(0, R)
            values = tl.load(X + (batch * H + row[:, None]) * W * C + j[None, :],
                             j[None, :] < W * C, other=0)
            target = P + (batch * (H + 2) + row[:, None] + 1) * PW * C
            tl.store(target + C + j[None, :], values, j[None, :] < (PW - 1) * C)
            c = tl.arange(0, C)
            tl.store(target + c[None, :], 0)
        else:
            tl.store(P + batch * (H + 2) * PW * C + j, 0)
            tl.store(P + (batch * (H + 2) + H + 1) * PW * C + j, 0)
            # The last unused horizontal tile lanes may read past the final row.
            if batch == tl.num_programs(0) - 1:
                guard = tl.arange(0, triton.next_power_of_2((KS - 1) * C))
                tl.store(P + tl.num_programs(0) * (H + 2) * PW * C + guard, 0,
                         guard < (KS - 1) * C)


@triton.jit
def _conv_cube(X, Wt, Y, N: tl.constexpr, H: tl.constexpr, C: tl.constexpr, F: tl.constexpr,
               PW: tl.constexpr, BM: tl.constexpr, KS: tl.constexpr):
    offsets = tl.arange(0, BM)
    c = tl.arange(0, C)
    f = tl.arange(0, F)
    with tle.scope(core_mode="cube"):
        # Explicit grid-stride work avoids compiler-dependent automatic block mapping.
        for tile in range(tl.program_id(0), N * (H * PW // BM), tl.num_programs(0)):
            batch = tile // (H * PW // BM)
            m = tile % (H * PW // BM) * BM + offsets
            acc = tl.zeros((BM, F), tl.float32)
            for position in range(KS * KS):
                address = ((batch * (H + 2) * PW + position // KS * PW
                            + m[:, None] + position % KS) * C + c[None, :])
                a = tl.load(X + address)
                b = tl.load(Wt + (position * C + c[:, None]) * F + f[None, :])
                # The pinned compiler's legacy buffer frontend supports this L1 path.
                a_l1 = bl.to_buffer(a, tle.dsa.ascend.L1)
                acc = tl.dot(bl.to_tensor(a_l1, writable=False), b, acc)
            tl.store(Y + (batch * H * PW + m[:, None]) * F + f[None, :], acc)


@triton.jit
def _crop(P, Y, ROWS: tl.constexpr, W: tl.constexpr, F: tl.constexpr,
          PW: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    j = tl.arange(0, PW * F)
    mask = (row[:, None] < ROWS) & (j[None, :] < W * F)
    with tle.scope(core_mode="vector"):
        values = tl.load(P + row[:, None] * PW * F + j[None, :], mask, other=0)
        tl.store(Y + row[:, None] * W * F + j[None, :], values, mask)


def allocate(x):
    n = x.shape[0]
    padded = torch.empty(n * (HEIGHT + 2) * PITCH * CHANNELS + (KERNEL - 1) * CHANNELS,
                         dtype=x.dtype, device=x.device)
    scratch = torch.empty((n, HEIGHT, PITCH, FILTERS), dtype=x.dtype, device=x.device)
    return padded, scratch


def stages(x, weight, padded, scratch, output):
    """Return launch descriptions; exposed for separate stage measurements."""
    n = x.shape[0]
    return (
        (_prepare, (n, HEIGHT // COPY_ROWS + 1, 1), (x, padded),
         (HEIGHT, WIDTH, CHANNELS, PITCH, COPY_ROWS, KERNEL), GRID_OPTIONS),
        (_conv_cube, (min(CUBE_WORKERS, n * HEIGHT * PITCH // TILE_M), 1, 1), (padded, weight, scratch),
         (n, HEIGHT, CHANNELS, FILTERS, PITCH, TILE_M, KERNEL), {**GRID_OPTIONS, "sync_solver": False}),
        (_crop, (triton.cdiv(n * HEIGHT, COPY_ROWS), 1, 1), (scratch, output),
         (n * HEIGHT, WIDTH, FILTERS, PITCH, COPY_ROWS), GRID_OPTIONS),
    )


def launch(x, weight, output):
    padded, scratch = allocate(x)
    descriptions = stages(x, weight, padded, scratch, output)
    key = (x.device.index, x.shape[0])
    aligned = x.data_ptr() % POINTER_ALIGNMENT == 0 and weight.data_ptr() % POINTER_ALIGNMENT == 0
    plans = _PLANS.get(key) if aligned else None
    stream = torch_npu._C._npu_getCurrentRawStreamNoWait(x.device.index)
    if plans is not None:
        for runner, (_, _, tensors, constants, _) in zip(plans, descriptions):
            runner(*tensors, *constants, stream=stream)
        return output

    plans = []
    for kernel, grid, tensors, constants, options in descriptions:
        compiled = kernel[grid](*tensors, *constants, **options)
        plans.append(compiled[grid])
    if aligned:
        _PLANS[key] = plans
    return output
