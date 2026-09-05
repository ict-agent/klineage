"""Custom SM90 BF16 GEMM; Python handles only compilation and launch."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

import torch

_ROOT = Path(__file__).resolve().parent
_CFG = json.loads((_ROOT / 'config.json').read_text())
_LIB = None
_DIM = 4096


def _library():
    global _LIB
    if _LIB is not None:
        return _LIB
    source = (_ROOT / 'kernel.cu').read_bytes() + (_ROOT / 'mma.cuh').read_bytes()
    digest = hashlib.sha256(source + json.dumps(_CFG, sort_keys=True).encode()).hexdigest()[:20]
    build = _ROOT / '.build' / digest
    build.mkdir(parents=True, exist_ok=True)
    binary = build / 'gemm.so'
    if not binary.exists():
        flags = {'G_OUT_COLS': _CFG['output_cols'], 'G_BM': _CFG['bm'], 'G_BN': _CFG['bn'], 'G_STAGES': _CFG['stages'],
                 'G_MATH_WGS': _CFG['math_wgs'], 'G_GROUP': _CFG['group'],
                 'G_GRID': _CFG['persistent'], 'G_WAIT': _CFG['wait'],
                 'G_FULL_TILES': _CFG['full_tiles'],
                 'G_PROD_WARPS': _CFG.get('producer_warps', 4),
                 'G_PROD_LAST': int(_CFG.get('producer_position', 'first') == 'last'),
                 'G_CACHE_A': _CFG.get('cache_a', 0), 'G_CACHE_B': _CFG.get('cache_b', 0),
                 'G_CACHE_C': _CFG.get('cache_c', 0), 'G_L2_PROMO': _CFG.get('l2_promotion', 0)}
        flags['G_REG_LOAD'] = _CFG['producer_registers']
        flags['G_REG_MATH'] = _CFG['consumer_registers']
        for operand in ['A', 'B', 'C']:
            flags[f'G_FRACTION_{operand}'] = _CFG[f'fraction_{operand.lower()}']
            flags[f'G_FALLBACK_{operand}'] = _CFG[f'fallback_{operand.lower()}']
        command = ['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-lineinfo',
                   '-gencode=arch=compute_90a,code=sm_90a',
                   '-Xcompiler=-fPIC', '-Xptxas=-v', '-shared',
                   str(_ROOT / 'kernel.cu'), '-lcuda', '-o', str(binary)]
        command.extend(f'-D{k}={v}' for k, v in flags.items())
        (build / 'command.json').write_text(json.dumps(command, indent=2))
        with (build / 'build.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError((build / 'build.log').read_text())
    lib = ctypes.CDLL(str(binary))
    lib.launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint64]
    lib.launch.restype = ctypes.c_int
    lib.error_text.argtypes = [ctypes.c_int]
    lib.error_text.restype = ctypes.c_char_p
    _LIB = lib
    return lib


def kernel(*, x, weight, out=None):
    if x.shape != (_DIM, _DIM) or weight.shape != (_DIM, _DIM):
        raise ValueError('This kernel is specialized for 4096x4096 inputs.')
    if out is None:
        out = torch.empty((_DIM, _DIM + _CFG['output_pad']), dtype=x.dtype, device=x.device)[:, :_DIM]
    lib = _library()
    code = lib.launch(x.data_ptr(), weight.data_ptr(), out.data_ptr(),
                      torch.cuda.current_stream(x.device).cuda_stream, out.stride(0))
    if code:
        raise RuntimeError(lib.error_text(code).decode())
    return out
