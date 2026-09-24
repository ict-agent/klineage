"""Render paired formal curves from sanitized recorded measurements only."""
import csv
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = ('without_memory', 'with_memory')

def stamp(s):
    return datetime.fromisoformat(s)

def main():
    out = ROOT/'plots'
    out.mkdir(exist_ok=True)
    groups = {}
    for setting in SETTINGS:
        unit = ROOT/setting
        result = json.loads((unit/'result.json').read_text())
        events = [json.loads(l) for l in (unit/'events.jsonl').read_text().splitlines()]
        api = [e for e in events if e['event']=='api_response']
        starts = {e['version']:e for e in events if e['event']=='evaluate_start'}
        done = {e['version']:e for e in events if e['event']=='evaluate'}
        start, deadline = stamp(result['model_started_at']), stamp(result['deadline_at'])
        rows = []
        for version in sorted(starts, key=lambda v:int(v[7:])):
            e = done.get(version, {})
            begin = stamp(starts[version]['at'])
            end = stamp(e.get('finished_at', result['deadline_at']))
            overlap = any(v != version and stamp(s['at']) < end and
                          stamp(done.get(v,{}).get('finished_at',result['deadline_at'])) > begin
                          for v,s in starts.items())
            valid = e.get('stage')=='full' and all(e.get(k) for k in ('compile_passed','correctness_passed','profile_passed')) and end<=deadline and not overlap
            latency = e.get('latency_ms') if valid else None
            status = 'valid' if valid else ('incomplete' if not e else ('diagnostic' if e['stage']!='full' else 'failed'))
            rows.append(dict(version=version,stage=e.get('stage','unknown'),status=status,
                             minutes=(end-start).total_seconds()/60,
                             tokens=sum(a.get('total_tokens') or 0 for a in api if stamp(a['at'])<=end),
                             correctness_passed=e.get('correctness_passed'),latency_ms=latency,
                             baseline_ms=result['baseline_ms'],initial_latency_ms=result['initial_latency_ms'],
                             vs_initial=result['initial_latency_ms']/latency if latency else None,
                             vs_reference=result['baseline_ms']/latency if latency else None))
        with (out/f'{setting}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        assert min(r['latency_ms'] for r in rows if r['latency_ms'])==result['latency_ms']
        groups[setting]=(rows,sum(a.get('total_tokens') or 0 for a in api))
    for axis in ('minutes','tokens'):
        fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True,gridspec_kw={'height_ratios':[3,3,1]})
        for setting,color,label in zip(SETTINGS,('#2878b5','#9467bd'),('A: no pre-injected expert knowledge','B: with expert knowledge')):
            rows,total=groups[setting];factor=1e6 if axis=='tokens' else 1
            for ax,metric in zip(axs[:2],('vs_initial','vs_reference')):
                best=0;xs=[];ys=[]
                for r in rows:
                    if r[metric] is not None:
                        best=max(best,r[metric]);xs.append(r[axis]/factor);ys.append(best)
                ax.step(xs+[total/factor if axis=='tokens' else 120],ys+[best],where='post',color=color,label=label)
            for r in rows:
                marker={'valid':'|','failed':'x','diagnostic':'o','incomplete':'^'}[r['status']]
                axs[2].scatter(r[axis]/factor,SETTINGS.index(setting),marker=marker,s=25,color=color)
        for ax in axs[:2]:
            ax.axhline(1,color='gray',linestyle=':');ax.set_yscale('log')
        axs[0].set_ylabel('Best valid speedup\nvs. initial AscendC');axs[1].set_ylabel('Best valid speedup\nvs. PyTorch reference')
        axs[0].set_title('SparseAttention / AscendC / same initial source');axs[0].legend(fontsize=8)
        axs[2].set_yticks([0,1],['A','B']);axs[2].set_ylim(-.5,1.5);axs[2].set_ylabel('Evaluations')
        axs[2].set_xlabel('Elapsed minutes' if axis=='minutes' else 'Cumulative API tokens (millions, including cached input)')
        for ax in axs:ax.set_xlim(left=0);ax.grid(alpha=.2)
        fig.text(.5,.012,'| valid full gate; x failed full gate; o diagnostic; triangle incomplete. Flat tails are not iterations.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.03,1,1))
        for ext in ('png','pdf'):fig.savefig(out/f'speedup-vs-{axis}.{ext}',dpi=180)
        plt.close(fig)

if __name__=='__main__':main()
