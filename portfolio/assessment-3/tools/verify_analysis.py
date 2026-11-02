"""Independent verification from saved rows, probabilities, metrics and models."""
from pathlib import Path
import os,json,pickle,hashlib,ast
os.environ.setdefault('KERAS_BACKEND','torch')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,log_loss
import keras
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'output'
meta=json.loads((out/'analysis_manifest.json').read_text(encoding='utf-8'))
raw=pd.read_csv(ROOT/'data/energy_data.csv')
assert hashlib.sha256((ROOT/'data/energy_data.csv').read_bytes()).hexdigest()==meta['source_sha256']
split=pd.read_csv(out/'split_manifest.csv')
prior=pd.read_csv(ROOT/'data/pa2_split_manifest.csv')
assert split.source_row_id.is_unique and len(split)==768
assert (split.pa2_partition==prior.partition).all()
assert split.partition.value_counts().to_dict()=={'fit':429,'test':231,'validation':108}
assert not set(split.loc[split.partition=='fit','source_row_id']) & set(split.loc[split.partition=='test','source_row_id'])
pred=pd.read_csv(out/'holdout_predictions.csv')
assert set(pred.source_row_id)==set(split.loc[split.partition=='test','source_row_id'])
np.testing.assert_array_equal(pred.truth,np.digitize(raw.loc[pred.source_row_id,'Y1'],meta['class_boundaries'],right=True))
scores=pd.read_csv(out/'master_results.csv',index_col=0)
checks={}
X=raw.loc[pred.source_row_id,['X'+str(i) for i in range(1,9)]].copy()
X.columns=meta['source_features']
with open(ROOT/'models/ann_preprocessor.pkl','rb') as f: prep=pickle.load(f)
for name,row in scores.iterrows():
    p=pred[[name+' probability '+str(j) for j in range(3)]].to_numpy(
        dtype='float32' if name.startswith('ANN') else 'float64')
    np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-6)
    assert np.all(p>=0) and np.all(p<=1)
    np.testing.assert_array_equal(p.argmax(axis=1),pred[name])
    for metric,value in [('test_accuracy',accuracy_score(pred.truth,pred[name])),
                         ('test_weighted_f1',f1_score(pred.truth,pred[name],average='weighted')),
                         ('test_macro_f1',f1_score(pred.truth,pred[name],average='macro')),
                         ('test_loss',log_loss(pred.truth,p,labels=[0,1,2]))]:
        np.testing.assert_allclose(row[metric],value,rtol=1e-7,atol=1e-7)
    if name.startswith('ANN'):
        batch=int(name.rsplit(' ',1)[1])
        model=keras.models.load_model(ROOT/f'models/ann_batch{batch}.h5',compile=False)
        assert [l.units for l in model.layers if isinstance(l,keras.layers.Dense)]==[64,32,16,3]
        assert [l.rate for l in model.layers if isinstance(l,keras.layers.Dropout)]==[.2,.2]
        assert model.count_params()==3747
        actual=model.predict(prep.transform(X).astype('float32'),verbose=0)
        history=pd.read_csv(out/f'ann_batch{batch}_epochs.csv')
        assert len(history)<=50 and len(history)>=10
        assert len(history)==int(pd.read_csv(out/'ann_validation.csv',index_col=0).loc[name,'epochs_run'])
    else:
        with open(ROOT/('models/'+name.lower().replace(' ','_')+'.pkl'),'rb') as f: model=pickle.load(f)
        actual=model.predict_proba(X)
    np.testing.assert_allclose(actual,p,atol=1e-6)
    checks[name]='metrics and saved-model predictions match'
group_scores=pd.read_csv(out/'geometry_stress_folds.csv')
for fold in [1,2,3]:
    f=pd.read_csv(out/f'geometry_fold{fold}_predictions.csv')
    for name in ['Random Forest','sklearn MLP','ANN batch 32']:
        value=group_scores.loc[(group_scores.fold==fold)&(group_scores.model==name),'accuracy'].iloc[0]
        np.testing.assert_allclose(value,accuracy_score(f.truth,f[name]))
nb=json.loads((ROOT/'Portfolio_Assessment_3_COS40007_NguyenVanTuanKiet.ipynb').read_text(encoding='utf-8'))
code_cells=[c for c in nb['cells'] if c['cell_type']=='code']
for cell in code_cells:
    ast.parse(''.join(cell['source']))
    assert cell['execution_count'] is not None
    assert all(o['output_type']!='error' for o in cell['outputs'])
verification={'status':'PASS','model_checks':checks,'rows':len(pred),'code_cells_executed':len(code_cells),
    'geometry_fold_metric_checks':9,'source_hash_verified':True,'prior_partition_preserved':True,
    'architecture_verified':True,'student_authorship_or_understanding_verified':False}
(out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification,indent=2))
