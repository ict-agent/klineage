"""Continue one Codex session until its fixed wall-clock deadline."""
from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

STOP_GRACE = 10
RETRY_DELAY = 10
SPAN_TOLERANCE = 120


def read_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except ValueError:
            continue
    return result


def thread_id(trace: Path) -> str | None:
    ids = {event['thread_id'] for event in read_events(trace)
           if event.get('type') == 'thread.started'}
    if len(ids) > 1:
        raise RuntimeError('Multiple session IDs in one unit; refusing to mix runs.')
    return next(iter(ids), None)


def log_event(out: Path, event: str, **fields) -> None:
    stats = out/'work/.klineage/stats'
    stats.mkdir(parents=True, exist_ok=True)
    with (stats/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event=event, at=datetime.now(UTC).isoformat(), **fields))+'\n')


def run_session(out, work, environment, codex, prompt, budget):
    started = time.monotonic()
    deadline = started + budget
    wall_start = datetime.now(UTC)
    wall_end = wall_start + timedelta(seconds=budget)
    session, attempts = None, 0
    status = dict(model_started_at=wall_start.isoformat(), deadline_at=wall_end.isoformat(),
                  timeout_s=budget)
    log_event(out, 'session_budget', **status)
    while (remaining := deadline-time.monotonic()) > 0:
        command = [codex, '-a', 'never', 'exec']
        if session:
            command += ['resume', session]
        command += ['--json', '--skip-git-repo-check',
                    '--output-last-message', str(out/'final_message.txt'), '-']
        message = prompt if not session else (
            f'Continue this same operator experiment. The fixed deadline is {wall_end.isoformat()}; '
            f'about {remaining:.0f} seconds remain. Keep implementing and measuring candidates '
            'through the unchanged evaluation command in AGENTS.md. A passing kernel or an '
            'earlier final report is not a reason to stop. Try further justified improvements, '
            'preserve every measured version and keep the best valid candidate. Do not idle, '
            'pad the trace, or repeat API requests merely to consume time. No new expert '
            'knowledge is supplied by this continuation. Do not restart the experiment.')
        attempts += 1
        log_event(out, 'session_resume' if session else 'session_start',
                  session_id=session, attempt=attempts, remaining_s=remaining)
        (out/f'continuation-{attempts:03d}.txt').write_text(message)
        status.update(session_id=session, attempts=attempts)
        (out/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        expired = False
        with (out/'trace.jsonl').open('a') as trace, (out/'stderr.log').open('a') as stderr:
            process = subprocess.Popen(command, cwd=work, stdin=subprocess.PIPE, stdout=trace,
                                       stderr=stderr, text=True, start_new_session=True,
                                       env=environment)
            try:
                process.communicate(message, timeout=max(0.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                expired = True
                signal_group(process, signal.SIGTERM)
                try:
                    process.communicate(timeout=STOP_GRACE)
                except subprocess.TimeoutExpired:
                    signal_group(process, signal.SIGKILL)
                    process.communicate(timeout=STOP_GRACE)
        session = thread_id(out/'trace.jsonl') or session
        log_event(out, 'session_segment_end', session_id=session, attempt=attempts,
                  returncode=process.returncode, deadline_reached=expired)
        if expired:
            break
        if not session:
            raise RuntimeError('Codex exited without a session ID; cannot resume the same session.')
        if process.returncode:
            # Preserve provider failures and avoid a rapid retry loop.
            time.sleep(min(RETRY_DELAY, max(0, deadline-time.monotonic())))
    elapsed = time.monotonic()-started
    log_event(out, 'session_budget_end', session_id=session, elapsed_s=elapsed)
    return dict(status, session_id=session, attempts=attempts, resume_count=max(0, attempts-1),
                budget_elapsed_s=elapsed, budget_exhausted=elapsed >= budget)


def signal_group(process, value):
    import os
    try:
        os.killpg(process.pid, value)
    except ProcessLookupError:
        pass


def audit_span(out: Path, budget: float) -> dict:
    events = read_events(out/'events.jsonl')
    responses = [e for e in events if e.get('event') == 'api_response'
                 and e.get('http_status') == 200 and e.get('complete_usage')]
    times = sorted(datetime.fromisoformat(e.get('finished_at') or e['at']) for e in responses)
    span = (times[-1]-times[0]).total_seconds() if len(times)>1 else 0
    evaluations = [e for e in events if e.get('event') == 'evaluate']
    source_hashes = set()
    for version in (out/'versions').glob('version*'):
        files = sorted((version/'submission').rglob('*'))
        digest = hashlib.sha256()
        for source in files:
            if source.is_file():
                digest.update(str(source.relative_to(version)).encode()+b'\0'+source.read_bytes())
        if files:
            source_hashes.add(digest.hexdigest())
    result = dict(first_response_at=times[0].isoformat() if times else None,
                  last_response_at=times[-1].isoformat() if times else None,
                  response_span_s=span, requested_duration_s=budget,
                  response_span_tolerance_s=SPAN_TOLERANCE,
                  response_span_passed=span >= max(0, budget-SPAN_TOLERANCE),
                  completed_evaluations=len(evaluations), unique_candidates=len(source_hashes),
                  iteration_evidence=len(evaluations)>1 and len(source_hashes)>1)
    (out/'duration-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
