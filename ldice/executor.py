"""Local subprocess executor. Timeout is a resource limit, NOT a security sandbox."""
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
def extract_source(text):
    text=text.strip()
    if text.startswith('```'):
        match=re.fullmatch(r"```(?:webppl|javascript|js)?\s*\n(.*?)\n```",text,re.S)
        if not match:
            raise ValueError('malformed code fence')
        return match.group(1).strip()
    return text

def parse_probability(stdout):
    lines=[l[len('RESULT_JSON:'):] for l in stdout.splitlines() if l.startswith('RESULT_JSON:')]
    if len(lines)!=1:
        raise ValueError('expected exactly one RESULT_JSON line')
    p=json.loads(lines[0])['probability']
    if type(p) not in (int,float) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('invalid probability (must be finite 0..1)')
    return float(p)

def resolve_webppl(command=None):
    candidate=command or str(ROOT/'node_modules' / '.bin' / 'webppl')
    path=shutil.which(candidate)
    if path is None and command is None:
        path=shutil.which('webppl')
    if path is None:
        raise FileNotFoundError('WebPPL missing; run npm ci in project root')
    return path

def execute(source, timeout=30, command=None):
    executable=resolve_webppl(command)
    with tempfile.TemporaryDirectory(prefix='ldice-') as tmp:
        path=Path(tmp)/'program.wppl';path.write_text(source)
        result=subprocess.run([executable,str(path)],cwd=tmp,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'WebPPL exit {result.returncode}: {result.stderr[-4000:]} {result.stdout[-1000:]}')
    return {'probability':parse_probability(result.stdout),'stdout':result.stdout,'stderr':result.stderr}
