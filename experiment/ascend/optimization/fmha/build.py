"""Build the fixed-shape FIA kernel and private OPP roots on Ascend 910B1/CANN 9.1."""
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
CANN = Path(os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/cann-9.1.0'))
OPP = CANN/'opp'
NAME = 'FusedInferAttentionScore_8cd36e66e4ceb2d60ef96db29a147347'
KERNEL_DIR = Path('built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_transformer/fused_infer_attention_score')


TILING_KEY = 5000000000000200100
BASELINE_SHA256 = '91644f785e1b618c1aa56f8b083d0523017091d5737fdb22ff13fdf5c21a80d3'


class Role(Enum):
    BASELINE = 'baseline'
    CANDIDATE = 'candidate'


def make_root(dest, role):
    def expand(relative):
        target = dest/relative
        if target.is_symlink():
            target.unlink()
        target.mkdir(exist_ok=True)
        for src in (OPP/relative).iterdir():
            p = target/src.name
            if not p.exists() and not p.is_symlink():
                p.symlink_to(src)
    expand(Path('.'))
    if (dest/'vendors').is_symlink():
        (dest/'vendors').unlink()
    (dest/'vendors').mkdir(exist_ok=True)
    (dest/'vendors/config.ini').write_text('load_priority=\n')
    if role == Role.CANDIDATE:
        relative = Path('.')
        for part in KERNEL_DIR.parts:
            relative /= part
            expand(relative)
        for suffix in ('.o', '.json'):
            target = dest/KERNEL_DIR/(NAME+suffix)
            if target.is_symlink():
                target.unlink()
            shutil.copy2((ROOT/'build')/(NAME+suffix), target)


def source_digest():
    digest = hashlib.sha256()
    for path in sorted((ROOT/'src').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            digest.update(str(path.relative_to(ROOT/'src')).encode() + b'\0')
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    baseline = OPP/KERNEL_DIR/(NAME+'.o')
    if not baseline.is_file():
        raise RuntimeError(f'Required CANN 9.1.0 official kernel is missing: {baseline}')
    baseline_sha256 = hashlib.sha256(baseline.read_bytes()).hexdigest()
    if baseline_sha256 != BASELINE_SHA256:
        raise RuntimeError('Official kernel differs from the validated CANN 9.1.0 baseline.')
    destination = ROOT
    out = destination/'build'
    out.mkdir(parents=True, exist_ok=True)
    cmd = ['asc_opc', str(ROOT/'src/fused_infer_attention_score/FusedInferAttentionScore.py'),
           '--main_func=fused_infer_attention_score', '--input_param='+str(ROOT/'compile_param.json'),
           '--soc_version=Ascend910B1', '--output='+str(out),
           '--impl_mode=high_performance,optional', '--simplified_key_mode=0',
           '--op_mode=dynamic', f'--tiling_key={TILING_KEY}']
    source_sha256 = source_digest()
    subprocess.run(cmd, check=True, cwd=destination)
    if source_digest() != source_sha256:
        raise RuntimeError('Source changed during compilation; refusing to install an untracked binary')
    make_root(ROOT/'opp_baseline', Role.BASELINE)
    make_root(ROOT/'opp_candidate', Role.CANDIDATE)
    metadata = dict(tiling_key=TILING_KEY, shape=[16384,64,128],
                    variant='final', source_sha256=source_sha256,
                    compile_param_sha256=hashlib.sha256((ROOT/'compile_param.json').read_bytes()).hexdigest(),
                    sequence_lengths=[2048]*8, soc='Ascend910B1', cann=str(CANN),
                    baseline_sha256=baseline_sha256,
                    candidate_sha256=hashlib.sha256((out/(NAME+'.o')).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
