"""Rebuild this operator's plots and checksums without model or NPU calls."""
from pathlib import Path
import runpy,json,hashlib
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    for script in ('plot.py','build_techniques.py'):
        runpy.run_path(str(ROOT/'plots'/script),run_name='__main__')
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json' and '__pycache__' not in p.parts}
    (ROOT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
