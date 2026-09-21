"""Plot all fixed-gate attempts from the archived experiment, without running kernels."""
from pathlib import Path
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
figure, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
for axis, setting in zip(axes, ('without_memory', 'with_memory')):
    unit = root / setting
    events = [json.loads(line) for line in (unit / 'events.jsonl').read_text().splitlines()]
    result = json.loads((unit / 'result.json').read_text())
    start = datetime.fromisoformat(result['model_started_at'])
    starts = {event['version']: event for event in events if event.get('event') == 'evaluate_start'}
    ends = {event['version']: event for event in events if event.get('event') == 'evaluate'}
    for version, event in starts.items():
        begin = (datetime.fromisoformat(event['at']) - start).total_seconds() / 60
        end = ends.get(version)
        finish = (datetime.fromisoformat(end.get('finished_at') or end['at']) - start).total_seconds() / 60 if end else begin
        passed = end and all(end.get(key) for key in ('compile_passed', 'correctness_passed', 'profile_passed'))
        color = 'tab:green' if passed else 'tab:red' if end else 'tab:orange'
        row = int(version[7:]) % 3
        axis.plot([begin, finish], [row, row], color=color, alpha=.3, linewidth=3)
        axis.scatter([finish], [row], color=color, marker='o' if passed else 'x', s=25)
        axis.annotate(version.replace('version', 'v'), (finish, row), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=7)
    correction = next(e for e in events if e.get('event') == 'protocol_correction')
    elapsed = (datetime.fromisoformat(correction['at']) - start).total_seconds() / 60
    axis.axvline(elapsed, color='gray', linestyle='--', label='Protocol correction')
    axis.set_title(setting)
    axis.set_xlim(0, 120)
    axis.set_ylim(-.4, 2.7)
    axis.set_yticks([])
    axis.grid(axis='x', alpha=.3)
axes[-1].set_xlabel('Elapsed minutes; green=passed, red=failed; dashed line=protocol correction; rows staggered')
figure.tight_layout()
for suffix in ('png', 'pdf'):
    figure.savefig(root / 'plots' / ('evaluation-timeline.' + suffix), dpi=180)
plt.close(figure)
