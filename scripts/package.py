"""Build a clean, allowlisted deliverable ZIP from any working directory."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT.parent/(ROOT.name+'.zip')
files=[p for p in ROOT.iterdir() if p.name in {
 'README.md','VERIFICATION.md','requirements.txt','package.json','package-lock.json',
 'run.py','simulate.py','.gitignore'}]
for folder in ('ldice','data','tests','examples','scripts'):
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.jsonl','.txt','.wppl','.md'))
with ZipFile(DEST,'w',ZIP_DEFLATED) as z:
    for p in sorted(files): z.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
print(DEST)
