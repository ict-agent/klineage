"""Protocol checks for the next Ascend generation runs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, REPO/'scripts/ascend'/f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class InputTests(unittest.TestCase):
    def test_sparse_indices_obey_the_definition(self):
        import torch
        generator = module('gen_inputs')
        definition = json.loads((REPO/'experiment/sparse_attention/problems/definitions/sparse_attention.json').read_text())
        workload = {'axes': {'TOKENS': 17, 'HEADS': 2, 'QK_DIM': 16, 'VALUE_DIM': 8, 'TOPK': 8}}
        indices = generator.build(definition, workload, ['indices'], seed=0)['indices']
        for row, selected in enumerate(indices[:, 0]):
            valid = selected[selected >= 0]
            self.assertEqual(len(valid), min(row + 1, 8))
            self.assertEqual(len(valid.unique()), len(valid))
            self.assertTrue(bool((valid <= row).all()))
            self.assertTrue(bool((selected[len(valid):] == -1).all()))
        again = generator.build(definition, workload, ['indices'], seed=0)['indices']
        self.assertTrue(torch.equal(indices, again))


class SnapshotTests(unittest.TestCase):
    def test_freeze_survives_a_failed_or_changed_submission(self):
        gate = module('eval')
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            source = work/'submission/solution/kernel.asc'
            source.parent.mkdir(parents=True)
            source.write_text('first')
            (work/'submission/config.toml').write_text('[build]\nlanguage="ascendc"\n')
            snapshot = gate.freeze_submission(work)
            source.write_text('second')
            self.assertEqual((snapshot/'submission/solution/kernel.asc').read_text(), 'first')
            self.assertNotEqual(snapshot, gate.freeze_submission(work))

class ProxyTests(unittest.TestCase):
    def test_full_response_trace_and_usage(self):
        import gzip
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.request import Request, urlopen
        proxy_module = module('trace_proxy')
        event = {'type': 'response.completed', 'response': {
            'id': 'response-1', 'model': 'requested-model',
            'usage': {'input_tokens': 5, 'output_tokens': 3, 'total_tokens': 8},
            'output': [{'type': 'function_call', 'name': 'exec_command',
                        'call_id': 'call-1', 'arguments': '{"cmd":"python eval.py"}'}]}}
        frame = ('data: '+json.dumps(event)+'\n\n').encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with proxy_module.AuditProxy(f'http://127.0.0.1:{server.server_port}', root) as proxy:
                    request = Request(proxy.url+'/responses', data=b'{"model":"requested-model"}',
                                      headers={'Authorization': 'Bearer never-log-this'})
                    self.assertEqual(urlopen(request, timeout=10).read(), frame)
                events = [json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
                response = next(x for x in events if x['event']=='api_response')
                self.assertEqual(response['total_tokens'], 8)
                self.assertIn('started_at', response)
                self.assertIn('finished_at', response)
                self.assertTrue(any(x['event']=='api_tool_call' for x in events))
                files = list((root/'api').rglob('*.gz'))
                self.assertEqual(len(files), 2)
                content = b''.join(gzip.decompress(p.read_bytes()) for p in files)
                self.assertIn(frame, content)
                self.assertNotIn(b'never-log-this', content)
        finally:
            server.shutdown()
            server.server_close()

class PlotTests(unittest.TestCase):
    def test_failed_profile_does_not_count_as_success(self):
        plot = module('plot')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = [dict(event='api_request', at='2026-09-21T00:00:00+00:00'),
                      dict(event='evaluate', at='2026-09-21T00:00:01+00:00',
                           finished_at='2026-09-21T00:00:09+00:00', latency_ms=1,
                           correctness_passed=True, compile_passed=True, profile_passed=False)]
            (root/'events.jsonl').write_text('\n'.join(map(json.dumps, events)))
            rows = plot.read_unit(root)
            self.assertEqual(rows[0]['seconds'], 9)
            self.assertNotIn('best', plot.milestones(rows))

class DurationTests(unittest.TestCase):
    def test_resume_uses_same_id_and_appends_trace(self):
        runner = module('session_loop')
        from unittest.mock import patch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = []
            class Process:
                returncode = 0
                def __init__(self, command, **kwargs):
                    calls.append(command)
                    self.output = kwargs['stdout']
                def communicate(self, prompt, timeout):
                    self.output.write(json.dumps(dict(type='thread.started', thread_id='same-session'))+'\n')
                    self.output.flush()
            ticks = iter([0, 0, 1, 1, 2, 2, 3, 4, 4, 4, 4, 4])
            with patch.object(runner.time, 'monotonic', side_effect=lambda: next(ticks, 4)), patch.object(runner.time, 'sleep'), patch.object(runner.subprocess, 'Popen', Process):
                result = runner.run_session(root, root, {}, 'codex', 'initial', 3)
            self.assertGreaterEqual(len(calls), 2)
            self.assertIn('resume', calls[1])
            self.assertIn('same-session', calls[1])
            self.assertEqual(result['session_id'], 'same-session')
            self.assertGreaterEqual(len((root/'trace.jsonl').read_text().splitlines()), 2)

    def test_running_best_never_uses_failure_or_regresses(self):
        plot = module('plot')
        rows = [dict(seconds=i, speedup=v, correctness_passed=ok, compile_passed=True, profile_passed=True)
                for i,v,ok in [(1,2,True),(2,99,False),(3,1,True),(4,4,True)]]
        points = plot.running_best(rows, 'speedup', 'seconds')
        self.assertEqual([p['speedup'] for p in points], [2,2,4])

class DeadlineTests(unittest.TestCase):
    def test_hung_process_stops_at_deadline(self):
        import os
        import sys
        runner = module('session_loop')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root/'fake-codex'
            fake.write_text("#!/bin/sh\nprintf '%s\\n' '{\"type\":\"thread.started\",\"thread_id\":\"deadline-session\"}'\nexec sleep 30\n")
            fake.chmod(0o700)
            result = runner.run_session(root, root, dict(os.environ), str(fake), 'test', 2)
            self.assertTrue(result['budget_exhausted'])
            self.assertEqual(result['session_id'], 'deadline-session')
            self.assertLess(result['budget_elapsed_s'], 12)

    def test_short_response_span_is_not_accepted(self):
        runner = module('session_loop')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            events=[dict(event='api_response', at=t, http_status=200, complete_usage=True)
                    for t in ('2026-09-21T00:00:00+00:00','2026-09-21T01:00:00+00:00')]
            (root/'events.jsonl').write_text('\n'.join(map(json.dumps,events)))
            audit=runner.audit_span(root,7200)
            self.assertEqual(audit['response_span_s'],3600)
            self.assertFalse(audit['response_span_passed'])


if __name__ == '__main__':
    unittest.main()

class DeviceGuardTests(unittest.TestCase):
    def test_busy_device_blocks_start(self):
        guard = module('device_guard')
        output = '''| NPU Chip | Process id | Process name | Process memory(MB) |
| 2 0 | 1834133 | VLLM | 60585 |
| 3 0 | 1838285 | VLLM | 60581 |'''
        self.assertEqual(guard.busy_devices(output), {2, 3})
        with self.assertRaises(RuntimeError):
            guard.require_idle(output, [2, 3])
        guard.require_idle(output, [4, 5])

class SyncTests(unittest.TestCase):
    def test_generated_diagnostics_are_not_mirrored(self):
        from unittest.mock import patch
        gate=module('eval')
        with patch.object(gate.subprocess,'run') as run:
            gate.push_work(Path('/tmp/unit/work'),'host','/tmp/remote')
        args=run.call_args.args[0]
        self.assertIn('extra-info',args)

class LanguageTests(unittest.TestCase):
    def test_triton_policy_rejects_native_bundles(self):
        gate = module('eval')
        gate.check_language('python', 'triton')
        gate.check_language('ascendc', 'unrestricted')
        with self.assertRaises(ValueError):
            gate.check_language('ascendc', 'triton')

    def test_both_prompts_pin_triton(self):
        from unittest.mock import patch
        batch = module('batch')
        with patch.dict('os.environ', {'KLINEAGE_LANGUAGE': 'triton'}):
            for setting in ('without_memory', 'with_memory'):
                fields = batch.template_fields(Path('/tmp/unit/work'), 'fused_add_rmsnorm', setting, Path('/tmp/unit'), '2', 2)
                for filename in ('agents.md.tmpl', 'prompt.md.tmpl'):
                    rendered = batch.render((REPO/'scripts/ascend'/filename).read_text(), fields)
                    self.assertIn('must use Triton', rendered)
                    self.assertNotIn('language is your choice', rendered)
                    self.assertNotIn('Language is unrestricted', rendered)

class TritonGateTests(unittest.TestCase):
    def test_unproven_launch_cannot_be_scored(self):
        gate = module('eval')
        for launches in (None, 0):
            fields = {'correctness_passed': True, 'profile_passed': True, 'latency_ms': 1.0}
            gate.require_triton(fields, launches)
            self.assertFalse(fields['profile_passed'])
            self.assertNotIn('latency_ms', fields)
        fields = {'profile_passed': True, 'latency_ms': 1.0}
        gate.require_triton(fields, 2)
        self.assertTrue(fields['profile_passed'])

class AscendCLanguageTests(unittest.TestCase):
    def test_gate_rejects_python(self):
        gate = module('eval')
        gate.check_language('ascendc', 'ascendc')
        with self.assertRaises(ValueError):
            gate.check_language('python', 'ascendc')

    def test_both_settings_require_ascendc(self):
        from unittest.mock import patch
        batch = module('batch')
        with patch.dict('os.environ', {'KLINEAGE_LANGUAGE': 'ascendc'}):
            for setting in ('without_memory', 'with_memory'):
                fields = batch.template_fields(Path('/tmp/unit/work'), 'fused_add_rmsnorm', setting, Path('/tmp/unit'), '2', 2)
                self.assertIn('must use AscendC', fields['{language_contract}'])
                self.assertIn('--language ascendc', fields['{eval_cmd}'])
