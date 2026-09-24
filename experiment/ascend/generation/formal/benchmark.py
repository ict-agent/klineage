"""Evaluate a copy of a formal candidate; never overwrite archived results."""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', choices=['sparse_attention','fused_add_rmsnorm'], required=True)
    parser.add_argument('--setting', choices=['without_memory','with_memory'],default='with_memory')
    parser.add_argument('--device',default='2')
    args = parser.parse_args()
    source = ROOT/args.kernel
    work = Path(tempfile.mkdtemp(prefix='formal-ascend-'))/'work'
    shutil.copytree(source/'problems',work/'problems')
    shutil.copytree(source/args.setting/'submission',work/'submission')
    spec = importlib.util.spec_from_file_location('inputs',REPO/'scripts/ascend/gen_inputs.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    definition = json.loads((work/'problems/definitions'/f'{args.kernel}.json').read_text())
    workload = json.loads((work/'problems/workloads'/f'{args.kernel}.jsonl').read_text().splitlines()[0])['workload']
    names = module.wanted(workload)
    if names:
        from safetensors.torch import save_file
        target = work/'problems/inputs'/f'{args.kernel}.safetensors'
        target.parent.mkdir(exist_ok=True)
        save_file(module.build(definition,workload,names,0),str(target),metadata={module.STRIDES_METADATA:'{}'})
    print('New evaluation workspace:',work,flush=True)
    subprocess.run([sys.executable,str(REPO/'scripts/ascend/eval.py'),'--work',str(work),
        '--device',args.device,'--language','ascendc','--host','910b','--container','pjj-fmha-opt',
        '--arch','Ascend910B3','--remote-root','/mnt/nvme0n1/home/pjj/klineage-generation-harness',
        '--container-root','/mnt/pjj/klineage-generation-harness'],check=True)

if __name__=='__main__': main()
