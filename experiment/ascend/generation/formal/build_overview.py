"""Build the four-group table and comparison figures from per-operator CSVs."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
KERNELS = ('sparse_attention', 'fused_add_rmsnorm')
SETTINGS = ('without_memory', 'with_memory')

def main():
    out = ROOT/'overview'
    out.mkdir(exist_ok=True)
    selected = json.loads((ROOT/'selection.json').read_text())
    rows = []
    for kernel in KERNELS:
        for setting in SETTINGS:
            s = next(x for x in selected if x['kernel']==kernel and x['setting']==setting)
            unit = ROOT/kernel/setting
            r = json.loads((unit/'result.json').read_text())
            snap = unit/'versions'/r['selected_version']/'submission'
            for p in (unit/'submission').rglob('*'):
                if p.is_file():
                    assert p.read_bytes()==(snap/p.relative_to(unit/'submission')).read_bytes()
            rows.append(dict(kernel=kernel,setting='A' if setting==SETTINGS[0] else 'B',
                             correct=r['correctness_passed'],reference_ms=s['baseline_ms'],
                             latency_ms=r['latency_ms'],vs_reference=s['baseline_ms']/r['latency_ms'],
                             initial_ms=r.get('initial_latency_ms'),vs_initial=r.get('vs_initial'),
                             selected_version=r['selected_version'],budget_s=r['budget_elapsed_s'],
                             response_span_s=r['response_span_s']))
    with (out/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for axis in ('minutes','tokens'):
        fig,axs=plt.subplots(1,2,figsize=(12,4.5))
        for ax,kernel in zip(axs,KERNELS):
            for setting,color,label in zip(SETTINGS,('#2878b5','#9467bd'),('A: no pre-injected expert knowledge','B: with expert knowledge')):
                with (ROOT/kernel/'plots'/f'{setting}.csv').open() as f:
                    data=list(csv.DictReader(f))
                xs=[];ys=[];best=0
                factor=1e6 if axis=='tokens' else 1
                for r in data:
                    metric=r.get('vs_reference',r.get('speedup'))
                    if metric:
                        best=max(best,float(metric));xs.append(float(r[axis])/factor);ys.append(best)
                events=[json.loads(l) for l in (ROOT/kernel/setting/'events.jsonl').read_text().splitlines()]
                end=sum(e.get('total_tokens') or 0 for e in events if e.get('event')=='api_response')/factor if axis=='tokens' else 120
                ax.step(xs+[end],ys+[best],where='post',color=color,label=label)
            ax.axhline(1,color='gray',linestyle=':');ax.set_yscale('log');ax.set_xlim(left=0)
            ax.set_title(kernel);ax.set_ylabel('Best valid speedup vs. PyTorch reference')
            ax.set_xlabel('Elapsed minutes' if axis=='minutes' else 'Cumulative API tokens (millions)')
            ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=7)
        fig.text(.5,.015,'Recorded full evaluations only; flat tails are not new iterations. See operator reports for limitations.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1))
        for ext in ('png','pdf'):fig.savefig(out/f'four-groups-vs-{axis}.{ext}',dpi=180)
        plt.close(fig)

if __name__=='__main__':main()
