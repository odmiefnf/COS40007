"""AI-generated reference inference; accepts original UCI X1-X8 columns."""
from pathlib import Path
import os,json,pickle,argparse
os.environ.setdefault('KERAS_BACKEND','torch')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
import numpy as np
import pandas as pd
import keras
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
meta=json.loads((ROOT/'models/model_metadata.json').read_text(encoding='utf-8'))
frame=pd.read_csv(args.input)
columns=['X'+str(i) for i in range(1,9)]
missing=set(columns)-set(frame.columns)
if missing: raise ValueError(f'Missing input columns: {sorted(missing)}')
X=frame[columns].copy()
if not np.isfinite(X.to_numpy(dtype=float)).all(): raise ValueError('Input must be finite numeric data.')
X.columns=meta['source_features']
with open(ROOT/'models/ann_preprocessor.pkl','rb') as f: prep=pickle.load(f)
batch=int(meta['selected_ann'].rsplit(' ',1)[1])
model=keras.models.load_model(ROOT/f'models/ann_batch{batch}.h5',compile=False)
prob=model.predict(prep.transform(X).astype('float32'),verbose=0)
pred=prob.argmax(axis=1)
output=pd.DataFrame({'source_row_id':frame.index,'predicted_class':pred,
    'predicted_band':[meta['class_names'][i] for i in pred]})
for i,label in enumerate(meta['class_names']): output['probability_'+label.lower()]=prob[:,i]
output.to_csv(args.output,index=False)
print('Saved',len(output),'predictions to',args.output)
