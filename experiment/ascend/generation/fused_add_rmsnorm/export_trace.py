"""Export structural session evidence; never export free-form conversation content."""
import argparse
import collections
import json
import re
from pathlib import Path

SAFE_KEYS = set('type event at timestamp started_at finished_at first_byte_at model_started_at deadline_at id thread_id session_id request_id response_id call_id status name role method requested_model model kernel_name platform version stage timeout_s'.split())
SAFE_VALUE = re.compile(r'^[A-Za-z0-9_.:+/ -]{0,160}$')
HIDDEN_KEYS = set('text message content command aggregated_output arguments error trace_path cwd instructions prompt'.split())

def action(command):
    if 'managed_eval.py' in command:
        return 'fixed_gate_managed'
    if 'scripts/ascend/eval.py' in command:
        return 'fixed_gate'
    if any(s in command for s in ('apply_patch', 'write_text(', "open(p,'w')", 'sed -i', 'cat >')):
        return 'file_edit'
    if command.lstrip().startswith(('cat ', 'head ', 'tail ', 'sed ', 'rg ', 'grep ', 'ls ', 'find ')):
        return 'file_inspection'
    return 'shell_command'

def clean(value, key=''):
    if key in HIDDEN_KEYS and value is not None:
        return '[REDACTED: free-form content]'
    if isinstance(value, dict):
        out = {(k if re.fullmatch(r"[a-z][a-z0-9_]*", k) else "redacted_field_"+str(i)): clean(v, k) for i,(k,v) in enumerate(value.items())}
        if isinstance(value.get('command'),str):
            out['action_category'] = action(value['command'])
        if isinstance(value.get('arguments'),str):
            try:
                args = json.loads(value['arguments'])
                if isinstance(args,dict) and isinstance(args.get('cmd'),str):
                    out['action_category'] = action(args['cmd'])
            except (ValueError,TypeError):
                pass
        return out
    if isinstance(value,list):
        return [clean(x,key) for x in value]
    if isinstance(value,str):
        if key == 'name' and value not in ('exec_command','write_stdin','apply_patch','update_plan','view_image','shell','shell_command'):
            return '[REDACTED]'
        if key in SAFE_KEYS and SAFE_VALUE.fullmatch(value) and not any(x in value for x in ('/Users/','/private/','sk-')):
            return value
        return '[REDACTED]'
    return value

def export(source,out):
    audit={'policy':'structural-trace-v1','raw_content_published':False,'files':{}}
    for name in ('trace.jsonl','session.jsonl','events.jsonl'):
        src=source/name
        assert src.exists(),name
        counts=collections.Counter();rows=[]
        for line in src.read_text().splitlines():
            original=json.loads(line)
            counts[original.get('type',original.get('event','unknown'))]+=1
            sanitized = clean(original)
            if original.get("event") == "evaluate":
                sanitized.setdefault("stage", "full")
            rows.append(sanitized)
        (out/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
        audit['files'][name]={'records':len(rows),'event_counts':dict(counts)}
    for name in ('stderr.log','final_message.txt'):
        src=source/name
        text=src.read_text() if src.exists() else ''
        (out/name).write_text('[REDACTED: original free-form text retained privately; see result.json for verified results.]\n')
        audit['files'][name]={'source_present':src.exists(),'source_lines':len(text.splitlines()),'body_redacted':True}
    (out/'trace-export-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (out/'TRACE_ACCESS.md').write_text('''# Trace 发布范围

trace.jsonl：Codex CLI逐事件结构，保留事件顺序、类型、状态、退出码及用量。
session.jsonl：Codex Session逐记录结构，保留原时间戳和序号。
events.jsonl：API请求/响应时间、HTTP状态、Token用量、工具调用名称与类别、全部固定评测及resume事件。
三个文件保留原记录数量；trace-export-audit.json提供数量核验。版本源码和测量见versions/及result.json。

为保护私有资料，所有自由文本（模型正文、Prompt、工具命令/参数/输出、错误正文等）统一脱敏。工具类别为粗粒度分类，不是原命令。stderr.log和final_message.txt也仅保留脱敏占位，不能视为原文。
本公开包是结构化脱敏Trace，支持时序、调用与测量核验，但不支持还原完整推理/调试正文。原始API请求/响应、Codex Trace及Session完整保留本地；未上传任何私有专家资料。
''')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();export(a.source,a.out)
