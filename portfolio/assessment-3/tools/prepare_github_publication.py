"""Update publication links while preserving every executed analysis cell/output."""
from pathlib import Path
import hashlib,json,shutil
import nbformat
ROOT=Path(__file__).resolve().parents[1]
REPO='https://github.com/odmiefnf/COS40007'
PA3=REPO+'/tree/main/portfolio/assessment-3'
STEM='Portfolio_Assessment_3_COS40007_NguyenVanTuanKiet'
nbpath=ROOT/(STEM+'.ipynb')
rev=ROOT/'revisions/before_github_publication'
rev.mkdir(parents=True,exist_ok=True)
for p in [nbpath,ROOT/(STEM+'.pdf'),ROOT/'README.md',ROOT/'RUBRIC_MAP.md',ROOT/'GENAI_EVIDENCE.md']:
    target=rev/p.name
    if not target.exists(): shutil.copy2(p,target)
def analysis_hash(nb):
    data=[{'source':c.source,'execution_count':c.execution_count,'outputs':c.outputs}
          for c in nb.cells if c.cell_type=='code']
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
old_cover='''**Repository:** https://github.com/odmiefnf/COS40007 . Assessment 3 is prepared locally;
publication status is recorded in README.md. No Canvas submission is performed.'''
new_cover=f'''**GitHub repository:** [{REPO}]({REPO})  
**Portfolio 3 code, notebook, saved models and results:** [portfolio/assessment-3]({PA3})  
No Canvas submission has been performed.'''
old_d='''The existing project repository is https://github.com/odmiefnf/COS40007 . Local source,
models, results, exact versions, dataset hashes and verification scripts accompany this
notebook. README.md records whether Assessment 3 has been published. A pre-existing
repository link is not evidence that these new files are already online. Preserve
genuine development milestones; do not backdate commits or invent authorship history.'''
new_d=f'''The complete Assessment 3 files are in [portfolio/assessment-3]({PA3}) on the main
branch of [{REPO}]({REPO}), the same repository as Portfolio 1 and 2. The repository
contains the executed notebook, source data, saved models, histories, metrics, figures,
PDF and reproducibility scripts. Real local development milestones are retained in
the integration history with OpenAI Codex attribution; no backdating or student-authorship
claim is made. No Canvas submission has been performed.

Historical execution note: code outputs and model metadata were recorded before
GitHub publication. Their publication flags describe that analysis run. The subsequent
publication is recorded separately in output/github_publication_receipt.json. No
analysis code, predictions, model weights or recorded outputs were changed to add links.'''
nb=nbformat.read(nbpath,as_version=4)
before=analysis_hash(nb)
for c in nb.cells:
    if c.cell_type=='markdown':
        c.source=c.source.replace(old_cover,new_cover).replace(old_d,new_d)
assert PA3 in nb.cells[0].source
assert before==analysis_hash(nb)
nbformat.validate(nb); nbformat.write(nb,nbpath)
builder=ROOT/'tools/build_notebook.py'
s=builder.read_text(encoding='utf-8').replace(old_cover,new_cover).replace(old_d,new_d)
builder.write_text(s,encoding='utf-8')
readme=ROOT/'README.md'
s=readme.read_text(encoding='utf-8')
old='''Assessment 3 is prepared locally in `portfolio3_publication/portfolio/assessment-3`
on branch `assessment-3-reference`. Its new history is independent of the existing
main branch. Commits are made as real milestones complete, explicitly attributed to
OpenAI Codex. No backdating or student-authorship claim is made. No remote push or
Canvas submission has been performed. The report does not present a future subfolder
URL as an already-published artifact.'''
new=f'''Assessment 3 files: {PA3}

The executed notebook, PDF, source data, model files, training histories, results and
reproducibility scripts are published on `main`, alongside the previous assessments.
Local development milestones from `assessment-3-reference` are retained in the
integration history and explicitly attributed to OpenAI Codex. No backdating or
student-authorship claim is made. Publication was expressly requested by the student.
No Canvas submission has been performed.

`output/github_publication_receipt.json` records the verified publication commit.
The analysis/model metadata and stored notebook outputs were recorded before
publication; their publication flags describe that earlier execution run. Repository
link updates preserve all executed code cells, predictions and model artifacts.'''
assert old in s or new in s
readme.write_text(s.replace(old,new),encoding='utf-8')
rubric=ROOT/'RUBRIC_MAP.md'
s=rubric.read_text(encoding='utf-8').replace(
    '| GitHub (4) | Separate local branch with real milestones attributed to Codex | Publication pending |',
    '| GitHub (4) | Main-branch folder linked on PDF cover; real local milestones retained with Codex attribution | Published; see receipt |')
rubric.write_text(s,encoding='utf-8')
record=ROOT/'GENAI_EVIDENCE.md'
s=record.read_text(encoding='utf-8')
addition=f'''\n\n## Subsequent GitHub publication request\n\nExact user prompt: "đẩy lên github của tôi và gắn link vào bài làm như bài trươcs"\n\nThe user expressly requested publishing to the same repository as previous work.\nThe assistant added direct {PA3} links to the report, notebook and documentation,\npreserved all executed analysis code and outputs, retained genuine local milestone\ncommits in the main-branch integration history, and verified published artifacts.\nNo Canvas submission, student signature or independent-authorship declaration was made.\n'''
if '## Subsequent GitHub publication request' not in s: s+=addition
record.write_text(s,encoding='utf-8')
update={'repository':REPO,'assessment_url':PA3,'branch':'main',
    'analysis_code_and_outputs_unchanged':True,'analysis_evidence_sha256':before,
    'canvas_submission_performed':False}
(ROOT/'output/github_link_update.json').write_text(json.dumps(update,indent=2),encoding='utf-8')
print('Updated repository and assessment links; preserved analysis evidence:',before)
