"""Rebuild formal result plots and tables without model/NPU calls."""
import csv
import hashlib
import json
import runpy
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SETTINGS = ('without_memory', 'with_memory')
KERNELS = ('sparse_attention', 'fused_add_rmsnorm')

def read_rows(unit):
    result = json.loads((unit/'result.json').read_text())
    start = datetime.fromisoformat(result['model_started_at'])
    tokens, rows = 0, []
    for line in (unit/'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event.get('event') == 'api_response':
            tokens += event.get('total_tokens') or 0
        if event.get('event') != 'evaluate':
            continue
        valid = all(event.get(k) for k in ('compile_passed', 'correctness_passed', 'profile_passed'))
        latency = event.get('latency_ms') if valid else None
        baseline = event.get('baseline_ms')
        rows.append(dict(version=event['version'], finished_at=event.get('finished_at', event['at']),
            minutes=(datetime.fromisoformat(event.get('finished_at', event['at']))-start).total_seconds()/60,
            tokens=tokens, correct=valid, latency_ms=latency, baseline_ms=baseline,
            speedup=baseline/latency if latency and baseline else None))
    return rows, tokens

def main():
    for kernel in KERNELS:
        if kernel == "sparse_attention":
            runpy.run_path(str(ROOT/kernel/"build_report.py"), run_name="__main__")
            continue
        folder = ROOT/kernel/'plots'
        folder.mkdir(exist_ok=True)
        data = {}
        for setting in SETTINGS:
            rows, tokens = read_rows(ROOT/kernel/setting)
            data[setting] = rows, tokens
            with (folder/f'{setting}.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        for axis in ('minutes', 'tokens'):
            fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={'height_ratios':[3,1]})
            for setting, color, label in zip(SETTINGS, ('#2878b5','#9467bd'), ('A: without pre-injected knowledge','B: with knowledge')):
                rows, tokens = data[setting]
                best, xs, ys = 0, [], []
                factor = 1e6 if axis == 'tokens' else 1
                for row in rows:
                    if row['speedup']:
                        best = max(best, row['speedup']); xs.append(row[axis]/factor); ys.append(best)
                    axes[1].scatter(row[axis]/factor, 0 if setting == SETTINGS[0] else 1,
                        color=color, marker='|' if row['correct'] else 'x')
                if xs:
                    xs.append(tokens/factor if axis=='tokens' else 120); ys.append(best)
                    axes[0].step(xs,ys,where='post',color=color,label=label)
                else:
                    axes[0].plot([],[],color=color,label=label+' (no valid result)')
            axes[0].axhline(1,color='gray',linestyle=':',label='Baseline parity')
            axes[0].set_yscale('log'); axes[0].set_ylabel('Running-best valid speedup')
            axes[0].set_title(kernel+' — formal AscendC runs'); axes[0].legend(fontsize=8)
            axes[1].set_yticks([0,1],['A','B']); axes[1].set_ylim(-.5,1.5)
            axes[1].set_ylabel('Evaluations'); axes[1].set_xlabel('Elapsed minutes' if axis=='minutes' else 'Cumulative tokens (millions, including cached input)')
            for ax in axes: ax.grid(alpha=.2); ax.set_xlim(left=0)
            fig.tight_layout()
            for ext in ('png','pdf'): fig.savefig(folder/f'speedup-vs-{axis}.{ext}',dpi=180)
            plt.close(fig)
    runpy.run_path(str(ROOT/"build_overview.py"), run_name="__main__")
    runpy.run_path(str(ROOT/"overview/build_techniques.py"), run_name="__main__")
    manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json' and '__pycache__' not in p.parts}
    (ROOT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
