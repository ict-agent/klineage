"""Plot the selected GQA A/B runs from their recorded gate and API events."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


FIELDS = ('version', 'seconds', 'reported_tokens', 'passed', 'latency_ms', 'best_ms')


def read_run(folder):
    state = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    elapsed = state['started_at_epoch']
    tokens = 0
    best = float('inf')
    rows = []
    counts = {'api_request': 0, 'api_response': 0, 'api_error': 0}

    for line in (folder / 'events.jsonl').read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        kind = event['event']
        if kind in counts:
            counts[kind] += 1
        if kind == 'api_response':
            usage = event.get('usage') or {}
            tokens += usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
        if kind != 'evaluate':
            continue

        passed = event['status'] == 'pass'
        latency = event.get('latency_ms') if passed else None
        if passed and (latency is None or latency <= 0):
            raise ValueError(f'Passing evaluation lacks a valid latency: {folder}')
        if passed:
            best = min(best, latency)
        rows.append({
            'version': event['version'],
            'seconds': round(datetime.fromisoformat(event['at']).timestamp() - elapsed, 6),
            'reported_tokens': tokens,
            'passed': passed,
            'latency_ms': latency if passed else '',
            'best_ms': best if best != float('inf') else '',
        })

    passed = [row for row in rows if row['passed']]
    if len(rows) != len(list((folder / 'versions').glob('version*/report.json'))):
        raise ValueError(f'Evaluation events and frozen versions differ: {folder}')
    if not passed or min(row['latency_ms'] for row in passed) != result['latency_ms']:
        raise ValueError(f'Result does not match valid evaluation events: {folder}')
    if f"version{min(passed, key=lambda row: row['latency_ms'])['version']}" != result['best_version']:
        raise ValueError(f'Best version differs from the recorded result: {folder}')
    return state, result, rows, counts, tokens


def save_plot(data, output, dimension):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scale, xlabel, filename = (
        (60, 'Experiment time (minutes)', 'latency-vs-time.png') if dimension == 'seconds'
        else (1_000_000, 'Reported API tokens (millions)', 'latency-vs-tokens.png')
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    colors = {'A': '#c05d36', 'B': '#087e73'}

    for setting, item in data.items():
        rows = item['rows']
        valid = [row for row in rows if row['passed']]
        x = [row[dimension] / scale for row in valid]
        y = [row['best_ms'] for row in valid]
        end = (120 if dimension == 'seconds' else item['reported_tokens'] / scale)
        if end > x[-1]:
            x.append(end)
            y.append(y[-1])
        ax.step(x, y, where='post', color=colors[setting], linewidth=2,
                label=f"{setting}: {item['result']['latency_ms']:.3f} ms best")
        ax.scatter([row[dimension] / scale for row in valid],
                   [row['latency_ms'] for row in valid], s=13, alpha=0.45,
                   color=colors[setting])

    ax.set(xlabel=xlabel, ylabel='Passing fixed-workload latency (ms)', ylim=(0, 95))
    if dimension == 'seconds':
        ax.set_xlim(0, 120)
    else:
        ax.set_xlim(left=0)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    parser.add_argument('--without', type=Path, default=root / 'without_memory')
    parser.add_argument('--with', dest='with_memory', type=Path, default=root / 'with_memory')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = {}
    for setting, folder in (('A', args.without), ('B', args.with_memory)):
        state, result, rows, counts, tokens = read_run(folder)
        data[setting] = {'state': state, 'result': result, 'rows': rows,
                         'api_counts': counts, 'reported_tokens': tokens}
        name = 'without_memory' if setting == 'A' else 'with_memory'
        with (args.output / f'gqa-{name}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    for dimension in ('seconds', 'reported_tokens'):
        save_plot(data, args.output, dimension)

    summary = {
        setting: {
            'best_version': item['result']['best_version'],
            'initial_ms': item['result']['initial_latency_ms'],
            'best_ms': item['result']['latency_ms'],
            'evaluations': len(item['rows']),
            'passed': sum(row['passed'] for row in item['rows']),
            'reported_tokens': item['reported_tokens'],
            'api_counts': item['api_counts'],
            'first_valid_at_seconds': next(row['seconds'] for row in item['rows'] if row['passed']),
            'best_at_seconds': next(row['seconds'] for row in item['rows']
                                    if row['latency_ms'] == item['result']['latency_ms']),
        }
        for setting, item in data.items()
        for folder in (args.without if setting == 'A' else args.with_memory,)
    }
    lines = [
        '| Kernel | Setting | Correct | Initial (ms) | Best (ms) | vs. Initial |',
        '| --- | --- | --- | ---: | ---: | ---: |',
    ]
    for setting, label in (('A', 'Without Expert Knowledge'), ('B', 'With Expert Knowledge')):
        item = summary[setting]
        lines.append(f"| GQA | {label} | Yes | {item['initial_ms']:.3f} | "
                     f"{item['best_ms']:.3f} | {item['initial_ms'] / item['best_ms']:.3f}x |")
    with (args.output / 'table.md').open('w', encoding='utf-8', newline='') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
