from pathlib import Path
import json,hashlib,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
root=Path(__file__).resolve().parent
out=root
summary={}
fig,axes=plt.subplots(2,1,figsize=(11,4),sharex=True)
for ax,s in zip(axes,('without_memory','with_memory')):
 p=root/s;es=[json.loads(l) for l in (p/'events.jsonl').read_text().splitlines()]
 result=json.loads((p/'result.json').read_text());start=datetime.fromisoformat(result['model_started_at'])
 api=[e for e in es if e.get('event')=='api_response']; starts={e['version']:e for e in es if e.get('event')=='evaluate_start'}; ends={e['version']:e for e in es if e.get('event')=='evaluate'}
 summary[s]=dict(result=result,total_tokens=sum(e.get('total_tokens') or 0 for e in api),input_tokens=sum(e.get('input_tokens') or 0 for e in api),output_tokens=sum(e.get('output_tokens') or 0 for e in api),api_pairs_match={e['request_id'] for e in es if e.get('event')=='api_request'}=={e['request_id'] for e in api},api_complete=all(e.get('http_status')==200 and e.get('complete_usage') for e in api),interrupted_versions=sorted(set(starts)-set(ends)))
 for version,event in starts.items():
  a=(datetime.fromisoformat(event['at'])-start).total_seconds()/60
  end=ends.get(version); b=(datetime.fromisoformat(end.get('finished_at') or end['at'])-start).total_seconds()/60 if end else a
  valid=end and all(end.get(k) for k in ('compile_passed','correctness_passed','profile_passed'))
  color='tab:green' if valid else 'tab:red' if end else 'tab:orange'
  marker='o' if valid else 'x' if end else '^'
  ax.plot([a,b],[0,0],color=color,alpha=.25,linewidth=4)
  ax.scatter([b],[0],color=color,marker=marker,s=45)
  ax.annotate(version.replace('version','v'),(b,0),xytext=(0,9 if int(version[7:])%2 else -18),textcoords='offset points',ha='center',fontsize=8)
 ax.set_title(s);ax.set_ylim(-.5,.5);ax.set_yticks([]);ax.set_xlim(0,120);ax.grid(axis='x',alpha=.3)
axes[-1].set_xlabel('Elapsed time (minutes); green=passed, red=failed, orange=no completion record')
fig.tight_layout();fig.savefig(out/'plots/evaluation-timeline.png',dpi=180);fig.savefig(out/'plots/evaluation-timeline.pdf');plt.close(fig)
(out/'audit-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for s in summary:
 print(s,summary[s]['interrupted_versions'],summary[s]['total_tokens'])
