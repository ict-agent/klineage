"""Reproduce acceptance on the one requested FMHA shape in isolated processes."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def worker(args):
    import torch
    import torch.nn.functional as F
    import torch_npu
    from fmha import run

    torch.set_num_threads(16)
    torch.manual_seed(args.seed)
    cpu = tuple(torch.randn(16384,64,128,dtype=torch.float16) for _ in range(3))
    q,k,v = (x.npu() for x in cpu)
    cuq = torch.tensor([2048*i for i in range(9)],dtype=torch.int32).to(q.device)
    cuk = cuq.clone()
    ends = [2048*i for i in range(1,9)]
    def core():
        return torch_npu.npu_fused_infer_attention_score(
            q,k,v,num_heads=64,num_key_value_heads=64,scale=128**-0.5,
            input_layout='TND',actual_seq_lengths=ends,actual_seq_lengths_kv=ends,
            pre_tokens=2147483647,next_tokens=2147483647,sparse_mode=0)[0]
    out = run(q,k,v,cuq,cuk).cpu()
    assert out.shape == (16384,64,128) and out.dtype == torch.float16 and out.is_contiguous()
    assert torch.isfinite(out).all()
    assert torch.equal(core().cpu(),out)
    results = ROOT/'results/latest'
    baseline_path = results/f'baseline_output_{args.seed}.pt'
    record = dict(role=args.worker,seed=args.seed,correct=False,
                  torch=torch.__version__,torch_npu=torch_npu.__version__,
                  opp=os.environ['ASCEND_OPP_PATH'])
    if args.worker == 'baseline':
        torch.save(out,baseline_path)
    else:
        expected = torch.load(baseline_path,weights_only=True)
        record['bitwise_equal'] = torch.equal(out,expected)
        assert record['bitwise_equal'], 'Candidate differs from official FIA'
        # Re-run against the same full reference to catch stale UB state or
        # cross-invocation ordering errors in the persistent accumulator.
        for _ in range(args.checks - 1):
            assert torch.equal(core().cpu(), expected), 'Repeated invocation differs from official FIA'
        record['full_comparisons'] = args.checks
        record['checked_elements_per_comparison'] = out.numel()
        del expected
    if args.oracle and args.worker == 'baseline':
        max_abs = 0.
        for b in range(8):
            qq,kk,vv = (x[b*2048:(b+1)*2048].float().transpose(0,1).unsqueeze(0) for x in cpu)
            ref = F.scaled_dot_product_attention(qq,kk,vv,dropout_p=0.,is_causal=False,scale=128**-0.5)
            ref = ref.squeeze(0).transpose(0,1)
            got = out[b*2048:(b+1)*2048].float()
            max_abs = max(max_abs,(got-ref).abs().max().item())
            assert torch.allclose(got,ref,atol=1e-3,rtol=1e-3), (b,max_abs)
        record['fp32_full_max_abs'] = max_abs
        record['fp32_checked_elements'] = out.numel()
    record['correct'] = True
    def timed(fn):
        for _ in range(20): fn()
        torch.npu.synchronize()
        times=[]
        for _ in range(args.trials):
            start,end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(args.repeat): fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end)*1000/args.repeat)
        return times
    record['core_trials_us'] = timed(core)
    record['entrypoint_trials_us'] = timed(lambda: run(q,k,v,cuq,cuk))
    record['core_median_us'] = statistics.median(record['core_trials_us'])
    record['entrypoint_median_us'] = statistics.median(record['entrypoint_trials_us'])
    (results/f'{args.worker}_{args.seed}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',default='20260921,123,777')
    ap.add_argument('--oracle',action='store_true',help='Check all outputs against CPU FP32 SDPA for the first seed')
    ap.add_argument('--repeat',type=int,default=50)
    ap.add_argument('--trials',type=int,default=7)
    ap.add_argument('--checks',type=int,default=3)
    ap.add_argument('--worker',choices=['baseline','candidate'])
    ap.add_argument('--seed',type=int)
    args=ap.parse_args()
    if min(args.repeat, args.trials, args.checks) < 1:
        ap.error('repeat, trials and checks must be positive')
    results=ROOT/'results/latest'
    results.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    pairs=[]
    for i,seed in enumerate(map(int,args.seeds.split(','))):
        for role in ('baseline','candidate'):
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',role,'--seed',str(seed),
                 '--repeat',str(args.repeat),'--trials',str(args.trials),'--checks',str(args.checks)]
            if args.oracle and i == 0: cmd.append('--oracle')
            env=dict(os.environ,ASCEND_OPP_PATH=str(ROOT/f'opp_{role}'),ASCEND_CUSTOM_OPP_PATH='')
            env.setdefault('ASCEND_RT_VISIBLE_DEVICES','0')
            with (results/f'{role}_{seed}.log').open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        base=json.loads((results/f'baseline_{seed}.json').read_text())
        opt=json.loads((results/f'candidate_{seed}.json').read_text())
        pair=dict(seed=seed,bitwise_equal=opt['bitwise_equal'],
                  baseline_us=base['core_median_us'],candidate_us=opt['core_median_us'],
                  core_speedup=base['core_median_us']/opt['core_median_us'],
                  baseline_entrypoint_us=base['entrypoint_median_us'],
                  candidate_entrypoint_us=opt['entrypoint_median_us'],
                  entrypoint_speedup=base['entrypoint_median_us']/opt['entrypoint_median_us'])
        pairs.append(pair)
        print(json.dumps(pair),flush=True)
        (results/f'baseline_output_{seed}.pt').unlink()
    summary=dict(warmup=20, repeat=args.repeat, trials=args.trials, checks=args.checks,
                 timer='NPU Event', statistic='median of per-trial mean latency',
                 shape=[16384,64,128],sequence_lengths=[2048]*8,pairs=pairs,
                 core_geomean_speedup=math.prod(x['core_speedup'] for x in pairs)**(1/len(pairs)),
                 entrypoint_geomean_speedup=math.prod(x['entrypoint_speedup'] for x in pairs)**(1/len(pairs)))
    summary['passed']=all(p['bitwise_equal'] and p['core_speedup']>1.05 and p['entrypoint_speedup']>1.05 for p in pairs)
    (results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if not summary['passed']: raise SystemExit('Acceptance failed; see results/summary.json')


if __name__=='__main__':
    main()
