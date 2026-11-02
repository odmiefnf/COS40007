"""Prepare a clean reference bundle; omit runtime, internal brief and QA images."""
from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
DEST=WORKSPACE/'Nop_bai_PA3_Energy'
assert DEST.resolve().parent==WORKSPACE.resolve()
DEST.mkdir(exist_ok=True)
selected=[]
for path in ROOT.iterdir():
    if path.is_file() and (path.suffix in ['.ipynb','.pdf','.md','.txt'] or path.name in ['.gitignore','.gitattributes']):
        selected.append(path)
for folder in ['data','models','output','tools']:
    for path in (ROOT/folder).rglob('*'):
        if not path.is_file(): continue
        rel=path.relative_to(ROOT)
        if any(part in ['jupyter','pdf_qa','__pycache__'] for part in rel.parts): continue
        if path.suffix in ['.pyc','.log']: continue
        selected.append(path)
for path in selected:
    rel=path.relative_to(ROOT)
    target=DEST/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target)
files={str(p.relative_to(DEST)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='PACKAGE_SHA256.json'}
(DEST/'PACKAGE_SHA256.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
archive=WORKSPACE/'Portfolio_Assessment_3_COS40007_NguyenVanTuanKiet_Reference.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
    for p in sorted(DEST.rglob('*')):
        if p.is_file(): z.write(p,arcname=str(Path(DEST.name)/p.relative_to(DEST)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for rel,digest in files.items():
        assert hashlib.sha256(z.read(DEST.name+'/'+rel)).hexdigest()==digest
print(json.dumps({'folder':str(DEST),'archive':str(archive),'files':len(files)+1,
                  'archive_bytes':archive.stat().st_size,'zip_integrity':'PASS'},indent=2))
