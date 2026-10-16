"""Independent consistency checks of stored predictions, tables and notebook outputs."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
for p in [ROOT.parent/'portfolio1/vendor',ROOT.parent/'portfolio2/vendor']:
    sys.path.insert(0,str(p))
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

stem='Portfolio_Assessment_2_COS40007_NguyenVanTuanKiet'
nb=nbformat.read(ROOT/f'{stem}.ipynb',as_version=4)
nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code),'Unexecuted code cell'
errors=[o for c in code for o in c.outputs if o.output_type=='error']
assert not errors,errors
figures=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)
assert figures>=8,figures
manifest=json.loads((ROOT/'output/analysis_manifest.json').read_text(encoding='utf-8'))
tables=[pd.read_csv(ROOT/'output'/f,index_col=0) for f in
    ['first_comparison_table.csv','final_comparison_table.csv']]
assert [len(x) for x in tables]==[5,4]
pred=pd.read_csv(ROOT/'output/holdout_predictions.csv')
checks=[]
for tbl in tables:
    for name,row in tbl.iterrows():
        actual,predicted=pred.Actual,pred[name]
        for metric,fun in [('Test_accuracy',accuracy_score),('Test_macro_F1',f1_score),
            ('Test_macro_precision',precision_score),('Test_macro_recall',recall_score)]:
            kwargs={} if metric=='Test_accuracy' else {'average':'macro','zero_division':0}
            assert np.isclose(fun(actual,predicted,**kwargs),row[metric]),(name,metric)
        checks.append(name)
assert tables[0].CV_accuracy.idxmax()==manifest['selected_tree']
assert tables[1].CV_accuracy.idxmax()==manifest['selected_svm']
splits=pd.read_csv(ROOT/'output/split_manifest.csv')
assert set(splits.loc[splits.partition=='test','source_row_id'])==set(pred.source_row_id)
assert pred.source_row_id.is_unique
foldfiles=['tree_fold_accuracy.csv','svm_outer_fold_accuracy.csv']
for tbl,file in zip(tables,foldfiles):
    folds=pd.read_csv(ROOT/'output'/file,index_col=0)
    assert folds.shape[0]==5
    for name in tbl.index:
        assert np.isclose(folds[name].mean(),tbl.loc[name,'CV_accuracy'])
        assert np.isclose(folds[name].std(ddof=1),tbl.loc[name,'CV_SD'])
summary={'status':'PASS','executed_code_cells':len(code),'error_outputs':len(errors),
    'embedded_png_outputs':figures,'prediction_rows':len(pred),
    'model_metrics_independently_recomputed':len(checks),
    'checks':['notebook schema','all code executed','no error outputs','figures present',
    'all 9 accuracy/precision/recall/F1 values reproduce stored predictions',
    'CV means and SDs reproduce 5 fold scores','CV-only selected procedures',
    'test split IDs equal holdout prediction IDs'],
    'student_verification':'pending'}
(ROOT/'output/verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
