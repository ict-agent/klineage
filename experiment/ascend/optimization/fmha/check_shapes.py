"""Correctness-only shape coverage; no additional performance claims."""
import argparse
from itertools import accumulate
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results/latest/shapes'
SEED = 20260921
TOLERANCE = 1e-3
# Q lengths, KV lengths, heads, dimension. The original benchmark is separate.
CASES = [
    ('resident_b1_h8_s1536', [1536], [1536], 8, 128),
    ('resident_b2_h16_s2048', [2048]*2, [2048]*2, 16, 128),
    ('resident_b3_h8_s3072', [3072]*3, [3072]*3, 8, 128),
    ('resident_b1_h4_s4096', [4096], [4096], 4, 128),
    ('resident_small_tasks', [1536], [1536], 1, 128),
    ('tail', [257]*2, [257]*2, 3, 128),
    ('ragged', [17, 129, 513], [17, 129, 513], 3, 128),
    ('ragged_aligned', [1536, 2048], [1536, 2048], 2, 128),
    ('distinct_boundaries', [31, 65], [47, 49], 2, 64),
    ('different_total_tokens', [1, 33], [17, 65], 1, 64),
    ('batch8_short', [65]*8, [65]*8, 4, 64),
    ('short_kv_loop', [1024], [1024], 4, 128),
] + [(f'dim_{dim}', [1, 17, 65], [1, 17, 65], 1, dim) for dim in range(16, 257, 16)]


def offsets(lengths):
    return [0, *accumulate(lengths)]


def worker(role):
    import torch
    import torch.nn.functional as F
    from fmha import run, select_path

    torch.set_num_threads(16)
    records = []
    for index, (name, q_lengths, k_lengths, heads, dim) in enumerate(CASES):
        torch.manual_seed(SEED + index)
        qo, ko = offsets(q_lengths), offsets(k_lengths)
        cpu = [torch.randn(qo[-1], heads, dim, dtype=torch.float16),
               torch.randn(ko[-1], heads, dim, dtype=torch.float16),
               torch.randn(ko[-1], heads, dim, dtype=torch.float16)]
        q, k, v = (x.npu() for x in cpu)
        cuq, cuk = (torch.tensor(x, dtype=torch.int32).to(q.device) for x in (qo, ko))
        output = run(q, k, v, cuq, cuk).cpu()
        assert output.shape == cpu[0].shape and output.dtype == torch.float16
        assert output.is_contiguous() and torch.isfinite(output).all()
        path = RESULTS/f'{name}.pt'
        if role == 'baseline':
            pieces = []
            for qs, qe, ks, ke in zip(qo, qo[1:], ko, ko[1:]):
                qq = cpu[0][qs:qe].float().transpose(0, 1).unsqueeze(0)
                kk, vv = (x[ks:ke].float().transpose(0, 1).unsqueeze(0) for x in cpu[1:])
                ref = F.scaled_dot_product_attention(qq, kk, vv, dropout_p=0.,
                                                    is_causal=False, scale=dim**-0.5)
                pieces.append(ref.squeeze(0).transpose(0, 1))
            reference = torch.cat(pieces)
            expected = output
            torch.save({'reference': reference, 'output': expected}, path)
        else:
            saved = torch.load(path, weights_only=True)
            reference, expected = saved['reference'], saved['output']
        error = (output.float() - reference).abs().max().item()
        assert torch.allclose(output.float(), reference, atol=TOLERANCE, rtol=TOLERANCE), (name, error)
        assert torch.allclose(output, expected, atol=TOLERANCE, rtol=TOLERANCE), name
        # Exercise CPU offsets and repeated invocation in the same process.
        assert torch.equal(output, run(q, k, v, cuq.cpu(), cuk.cpu()).cpu()), name
        record = dict(name=name, q_lengths=q_lengths, k_lengths=k_lengths, heads=heads,
                      head_dim=dim, path=select_path(dim, [qo, ko]).value,
                      elements=output.numel(), fp32_max_abs=error,
                      official_bitwise_equal=torch.equal(output, expected), correct=True)
        records.append(record)
        print(json.dumps(record), flush=True)
        if role == 'candidate':
            path.unlink()
    # Reject malformed metadata before either device path is launched.
    rejected = []
    for label, invalid in (
        ('nonzero_start', [1, 1, 18, 83]),
        ('empty_sequence', [0, 0, 18, 83]),
        ('wrong_total', [0, 1, 18, 82]),
        ('descending', [0, 18, 1, 83]),
    ):
        try:
            run(q, k, v, torch.tensor(invalid, dtype=torch.int32), cuk)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(f'Accepted {label}')
    result = dict(role=role, cases=records, rejected=rejected, passed=True,
                  atol=TOLERANCE, rtol=TOLERANCE, performance_measured=False)
    (RESULTS/f'{role}.json').write_text(json.dumps(result, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=('baseline', 'candidate'))
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker)
        return
    for role in ('baseline', 'candidate'):
        env = dict(os.environ, ASCEND_OPP_PATH=str(ROOT/f'opp_{role}'), ASCEND_CUSTOM_OPP_PATH='')
        env.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')
        with (RESULTS/f'{role}.log').open('w') as log:
            subprocess.run([sys.executable, __file__, '--worker', role], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((RESULTS/'candidate.json').read_text())
    (RESULTS/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f"Shape correctness: {len(result['cases'])} cases passed; no performance measured.", flush=True)


if __name__ == '__main__':
    main()
