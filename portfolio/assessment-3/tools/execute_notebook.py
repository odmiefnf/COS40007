"""Execute the complete notebook using the interpreter running this script."""
from pathlib import Path
import os,sys,json,time
ROOT=Path(__file__).resolve().parents[1]
config=ROOT/'tools/jupyter'
kernel=config/'kernels/python3'
kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'Python 3 (Portfolio 3)','language':'python'}),encoding='utf-8')
for name,part in [('JUPYTER_PATH',''),('JUPYTER_CONFIG_DIR','config'),('JUPYTER_RUNTIME_DIR','runtime'),
                  ('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib')]:
    os.environ[name]=str(config/part)
os.environ.update(KERAS_BACKEND='torch',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
import nbformat
from nbclient import NotebookClient
path=ROOT/'Portfolio_Assessment_3_COS40007_NguyenVanTuanKiet.ipynb'
nb=nbformat.read(path,as_version=4)
def progress(cell,cell_index,**kwargs):
    print(f'Cell {cell_index+1}/{len(nb.cells)}: '+cell.source[:85].replace('\n',' '),flush=True)
client=NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},on_cell_start=progress)
start=time.perf_counter()
try: client.execute()
finally: nbformat.write(nb,path)
print('Execution completed in',round(time.perf_counter()-start,1),'seconds',flush=True)
