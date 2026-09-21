"""Record Responses requests, complete SSE bodies, usage and tool calls.

HTTP authentication headers are forwarded but never persisted. Compressed
request/response bodies retain the API evidence without duplicating large
contexts as plain text in events.jsonl.
"""
from __future__ import annotations

import gzip
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from klineage.logging import HOP_BY_HOP, complete_events, data_lines, now, record, usage_from_event

READ_SIZE = 65536
UPSTREAM_TIMEOUT = 600


def handler(upstream: str, directory: Path):
    class Proxy(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def forward(self):
            request_id = uuid4().hex
            started_at, started = now(), time.monotonic()
            folder = directory/'api'/request_id
            folder.mkdir(parents=True)
            body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            with gzip.open(folder/'request.json.gz', 'wb') as target:
                target.write(body)
            try:
                requested_model = json.loads(body).get('model')
            except (ValueError, AttributeError):
                requested_model = None
            record('api_request', directory=directory, request_id=request_id,
                   started_at=started_at, method=self.command, path=self.path,
                   requested_model=requested_model)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP and k.lower() != 'accept-encoding'}
            headers['Accept-Encoding'] = 'identity'
            request = Request(upstream.rstrip('/')+self.path, data=body or None,
                              headers=headers, method=self.command)
            fields = dict(request_id=request_id, started_at=started_at,
                          requested_model=requested_model, trace_path=f'api/{request_id}')
            pending, usage = b'', {}
            try:
                try:
                    response = urlopen(request, timeout=UPSTREAM_TIMEOUT)
                except HTTPError as error:
                    response = error
                with response, gzip.open(folder/'response.sse.gz', 'wb') as capture:
                    fields['http_status'] = response.status
                    self.send_response(response.status)
                    for key, value in response.headers.items():
                        if key.lower() not in HOP_BY_HOP:
                            self.send_header(key, value)
                    self.send_header('Transfer-Encoding', 'chunked')
                    self.end_headers()
                    while data := response.read1(READ_SIZE):
                        fields.setdefault('first_byte_at', now())
                        capture.write(data)
                        frames, pending = complete_events((pending+data).replace(b'\r\n', b'\n'))
                        for frame in frames:
                            for payload in data_lines(frame):
                                parsed = usage_from_event(payload)
                                if parsed:
                                    usage = parsed
                                try:
                                    event = json.loads(payload)
                                except ValueError:
                                    continue
                                if event.get('type') != 'response.completed':
                                    continue
                                for item in event.get('response', {}).get('output', []):
                                    if item.get('type') in ('function_call', 'custom_tool_call'):
                                        record('api_tool_call', directory=directory,
                                               request_id=request_id, call=item)
                        self.wfile.write(b'%x\r\n'%len(data)+data+b'\r\n')
                        self.wfile.flush()
                    self.wfile.write(b'0\r\n\r\n')
                    self.wfile.flush()
            except Exception as error:
                fields['error'] = f'{type(error).__name__}: {error}'
                self.close_connection = True
            finally:
                fields.update(usage)
                fields.update(finished_at=now(), duration_s=time.monotonic()-started,
                              complete_usage=bool(usage))
                (folder/'metadata.json').write_text(json.dumps(fields, indent=2)+'\n')
                record('api_response', directory=directory, **fields)

        do_POST = forward
        do_GET = forward
        do_DELETE = forward

    return Proxy


class AuditProxy:
    def __init__(self, upstream: str, directory: Path):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler(upstream, directory))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_port}'

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
