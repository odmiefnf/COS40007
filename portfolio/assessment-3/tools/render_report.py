"""Create the concise PDF explanation from executed, verified analysis artifacts."""
from pathlib import Path
import json,re,pickle
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from PIL import Image as PILImage

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'
DEST=ROOT/'Portfolio_Assessment_3_COS40007_NguyenVanTuanKiet.pdf'
REPO='https://github.com/odmiefnf/COS40007'
PA3=REPO+'/tree/main/portfolio/assessment-3'
M=json.loads((OUT/'analysis_manifest.json').read_text(encoding='utf-8'))
R=pd.read_csv(OUT/'master_results.csv',index_col=0)
CV=pd.read_csv(OUT/'traditional_cv.csv',index_col=0)
V=pd.read_csv(OUT/'ann_validation.csv',index_col=0)
G=pd.read_csv(OUT/'geometry_stress_folds.csv')
COUNTS=pd.read_csv(OUT/'class_counts.csv',index_col=0)
for name,file in [('Body','arial.ttf'),('BodyBold','arialbd.ttf'),('Mono','consola.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
NAVY=colors.HexColor('#19384c'); TEAL=colors.HexColor('#087f8c')
INK=colors.HexColor('#253846'); GRAY=colors.HexColor('#607786'); LIGHT=colors.HexColor('#eef4f5')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyText2',fontName='Body',fontSize=10,leading=14.1,textColor=INK,spaceAfter=8))
styles.add(ParagraphStyle(name='H1x',fontName='BodyBold',fontSize=21,leading=25,textColor=NAVY,spaceAfter=13))
styles.add(ParagraphStyle(name='H2x',fontName='BodyBold',fontSize=12.5,leading=16,textColor=TEAL,spaceBefore=9,spaceAfter=7,keepWithNext=True))
styles.add(ParagraphStyle(name='Smallx',fontName='Body',fontSize=8.4,leading=11.3,textColor=GRAY,spaceAfter=7))
styles.add(ParagraphStyle(name='Cellx',fontName='Body',fontSize=8.2,leading=10.5,textColor=INK))
styles.add(ParagraphStyle(name='CellHeadx',fontName='BodyBold',fontSize=8.2,leading=10.5,textColor=colors.white))
styles.add(ParagraphStyle(name='Codex',fontName='Mono',fontSize=8,leading=11,textColor=INK,spaceAfter=7))
story=[]
def p(text,style='BodyText2'):
    story.append(Paragraph(text,styles[style]))
def title(kicker,text):
    p(kicker.upper(),'Smallx'); p(text,'H1x')
def h(text): p(text,'H2x')
def page(): story.append(PageBreak())
def table(headers,rows,widths=None):
    content=[[Paragraph(escape(str(x)),styles['CellHeadx']) for x in headers]]
    for row in rows: content.append([Paragraph(escape(str(x)),styles['Cellx']) for x in row])
    t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('LEFTPADDING',(0,0),(-1,-1),7),
        ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,0),(-1,0),.6,TEAL),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#d7e4e7'))]))
    story.extend([t,Spacer(1,10)])
def image(name,caption,maxheight=300):
    path=OUT/name
    with PILImage.open(path) as im: w,h=im.size
    scale=min(499/w,maxheight/h)
    story.append(Image(str(path),w*scale,h*scale,hAlign='CENTER'))
    p(caption,'Smallx')
def md_paragraphs(text):
    for block in text.split('\n\n'):
        block=' '.join(block.splitlines()).strip()
        if not block: continue
        if block.startswith('### '): h(block[4:])
        else:
            value=escape(block)
            value=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',value)
            p(value)
def pct(v): return f'{v:.2%}'
def metric_rows():
    return [[name,pct(r.test_accuracy),f'{r.test_weighted_f1:.4f}',f'{r.test_macro_f1:.4f}',f'{r.test_loss:.4f}'] for name,r in R.iterrows()]

title('Portfolio Assessment 3 | Option 3d','Advanced ML and<br/>Deep Learning')
p('Building heating-demand classification','H2x')
p('<b>Nguyen Van Tuan Kiet</b><br/>Student ID: 104772099<br/>COS40007 - Artificial Intelligence Engineering<br/>Swinburne Vietnam | Class identifier: COS40007')
p(f'GitHub repository: <link href="{REPO}" color="#087f8c">github.com/odmiefnf/COS40007</link><br/>Portfolio 3 code, notebook, models and results: <link href="{PA3}" color="#087f8c">portfolio/assessment-3</link>','Smallx')
story.append(Spacer(1,12))
table(['REFERENCE IMPLEMENTATION - AUTHORSHIP STATUS'],[[
    'OpenAI Codex substantially generated the code, experiment, analysis and report. The supplied brief explicitly prohibits whole-code GenAI generation. This reference is not evidence of independent student authorship, and disclosure alone does not satisfy that restriction. Student review and in-person demonstration remain pending.']], [499])
h('Purpose and outcome')
p('Continue the Portfolio 2 Energy Efficiency classification task and compare two traditional-model implementations with three Keras ANN batch-size variants on identical data. Report measured accuracy, class performance, probability loss and a geometry-disjoint stress check.')
p(f'Training CV selects <b>{escape(M["selected_traditional"])}</b>; validation loss selects <b>{escape(M["selected_ann"])}</b>. The observed leader on the reused 231-row comparison set is <b>{escape(M["observed_test_leader"])}</b>. These results do not establish readiness for unseen real buildings.')
h('Acknowledgement of Country')
p('I acknowledge the Traditional Owners of the lands on which Swinburne operates and pay respect to Elders past and present. I recognise the continuing connections of Aboriginal and Torres Strait Islander peoples to Country, culture and community.')

page(); title('A.1-A.2 | Dataset and preparation','A fixed, comparable task')
p('The UCI Energy Efficiency dataset contains 768 simulated building configurations and eight explanatory variables [1]. It is the replacement dataset used for Portfolio 2. No claim is made that it was the dataset originally submitted in Portfolio 1.')
table(['Inputs','Handling','Reason'],[
    ['Compactness; surface, wall and roof areas; height; glazing area','StandardScaler','Six numerical features with different scales.'],
    ['Orientation; glazing distribution','One-hot encoding','Nominal categories should not imply an ordinal distance.'],
    ['Y1 heating load; Y2 cooling load','Excluded from predictors','Y1 defines labels only. Y2 would introduce an unavailable related outcome.']], [178,130,191])
p(f'The fixed PA2 heating-load thresholds are <b>{M["class_boundaries"][0]:.6f}</b> and <b>{M["class_boundaries"][1]:.6f}</b> dataset units. Lower is at or below the first cutoff, Middle is above the first and at or below the second, and Higher is above the second. These are relative demand bands, not official efficiency ratings.')
table(['Partition','Rows','Lower','Middle','Higher','Encoded shape'],[
    [name,int(COUNTS.loc[name].sum()),*COUNTS.loc[name].astype(int).tolist(),f'({int(COUNTS.loc[name].sum())}, 16)']
    for name in ['fit','validation','test']], [89,45,65,65,65,170])
image('01_data_and_classes.png','Figure 1. Actual class counts and fitting-only heating-load distribution. The raw source has no missing, nonfinite or duplicate rows.',235)
p('Source CSV SHA256: '+M['source_sha256'],'Smallx')

page(); title('A.3-A.5 | Traditional ML','Cross-validation and comparison')
p('Preserve the PA2 537/231 development/test membership. Reserve 108 stratified development rows for validation and fit every main model on the remaining 429. Preprocessing is refitted inside each training CV fold. Both traditional models use four configurations and identical stratified five-fold splits, scored by accuracy and weighted F1.')
table(['Model','Selected configuration','CV accuracy ± SD','CV weighted F1'],[
    [name,CV.loc[name,'best_parameters'],f'{CV.loc[name,"cv_accuracy"]:.4f} ± {CV.loc[name,"cv_accuracy_sd"]:.4f}',f'{CV.loc[name,"cv_weighted_f1"]:.4f}'] for name in CV.index], [90,209,112,88])
p('Random Forest uses 300 bootstrap trees with depth/leaf-size control. The sklearn MLP uses ReLU, Adam, L2 regularisation and an internal 10% early-stopping split. It remains a neural network; the report uses the assessment\'s model categories rather than claiming a fundamental distinction from ANN.')
table(['Model','Accuracy','Weighted F1','Macro F1','Crossentropy'],metric_rows()[:2],[135,91,91,91,91])
image('03_master_comparison.png','Figure 2. The required primary metric and weighted F1 across both traditional models and all three ANN variants. The majority-class baseline is shown only on accuracy.',240)
p('Selection uses training CV. Grid-search CV is not nested and may be optimistic for the selected configuration. The 231 comparison rows were already examined in Portfolio 2; their reuse enables like-for-like comparison but not a new untouched external test.','Smallx')

page(); title('B.1-B.2 | Deep learning','The specified ANN architecture')
p('Energy data are non-sequential rows without image structure; option 3d is appropriate. The same 16 encoded input dimensions and the same 429/108/231 rows are used for all main models. Scaling is fitted only on 429 fitting rows. No sequence reshape is applied.')
table(['Layer','Output dimensions','Parameters','Purpose'],[
    ['Input','16','0','Eight source features become 16 encoded inputs.'],
    ['Dense + ReLU','64','1,088','Learn nonlinear feature combinations.'],
    ['Dropout (0.2)','64','0','Regularise the first hidden layer.'],
    ['Dense + ReLU','32','2,080','Combine learned representations.'],
    ['Dropout (0.2)','32','0','Regularise the second hidden layer.'],
    ['Dense + ReLU','16','528','Compact hidden representation.'],
    ['Dense + Softmax','3','51','Probabilities for Lower/Middle/Higher.'],
    ['Total trainable parameters','','3,747','No convolutional or recurrent layers.']], [155,100,78,166])
p('Compilation uses Adam at learning rate 0.001, sparse categorical crossentropy and accuracy. Sparse loss matches integer labels 0/1/2. Softmax probabilities sum to one; dropout is disabled during inference. Keras 3 uses the installed PyTorch CPU backend. The model is saved as a genuine Keras .h5 file and as a native .keras checkpoint [5,6].')
p('The complete executed model.summary() output is stored in the notebook and reproduced as a browser screenshot in the evidence appendix. Layer dimensions and parameter counts above are independently verified from reloaded .h5 artifacts.','Smallx')

page(); title('B.3 | Training and model selection','Batch size is the controlled variable')
p('Three experiments change batch size only: 16, 32 and 64. All start from seed 42 and share the prescribed architecture. Batch size affects the number of optimiser updates per epoch, so equal epochs do not imply equal update counts. Histories retain progress at epoch 10 and later milestones, with a ceiling of 50 epochs.')
table(['Setting','Implementation'],[
    ['Validation','Explicit 108-row stratified validation partition (20% of development).'],
    ['Early stopping','Monitor val_loss; patience 10; restore best weights.'],
    ['Learning-rate reduction','Monitor val_loss; halve rate after 5 plateau epochs; minimum 0.000001.'],
    ['Checkpoint and history','Save minimum-val-loss .keras checkpoint; export .h5; pickle and CSV histories.'],
    ['Selection','Lowest validation crossentropy; comparison-set accuracy is not used to choose batch size.']], [130,369])
table(['ANN variant','Epochs run','Best epoch','Validation loss','Validation accuracy'],[
    [name,int(r.epochs_run),int(r.best_epoch),f'{r.validation_loss:.4f}',pct(r.validation_accuracy)] for name,r in V.iterrows()], [119,70,70,115,125])
p(f'<b>{escape(M["selected_ann"])}</b> is the selected ANN. Each reported main model is trained only on the common fitting partition; no model receives extra validation rows for a final refit. This keeps the comparison controlled.')
p('The script loads the exact minimum-validation-loss checkpoint before evaluation. It does not assume the last epoch is optimal. A learning-rate reduction occurs only if the monitored plateau condition is met; a configured callback is not evidence that it necessarily fired.','Smallx')

page(); title('B.3 | Observed learning','Training and validation histories')
image('02_ann_learning_curves.png','Figure 3. Actual fitting/validation crossentropy (top) and accuracy (bottom). The vertical marker identifies the minimum-validation-loss epoch. Every run is capped at 50 epochs.',355)
for name,r in V.iterrows():
    hist=pd.read_csv(OUT/f'ann_batch{int(r.batch_size)}_epochs.csv')
    first,last=hist.iloc[0],hist.iloc[-1]
    p(f'<b>{escape(name)}:</b> validation loss changes from {first.val_loss:.4f} at epoch 1 to {last.val_loss:.4f} at the last epoch; the saved checkpoint is epoch {int(r.best_epoch)}. Validation accuracy at the selected checkpoint is {r.validation_accuracy:.2%}.')
p('Loss and accuracy measure different properties. Crossentropy can worsen while hard-label accuracy stays constant when incorrect predictions become overconfident. Divergence between fitting and validation is an overfitting signal, but dropout makes training/evaluation curves intrinsically different. These data compare batch-size regimes; they do not isolate the causal benefit of dropout or depth.')

page(); title('B.4 | Evaluation','Class-level behaviour')
best_ann=M['selected_ann']; best_ml=M['selected_traditional']
table(['ANN variant','Accuracy','Weighted F1','Macro F1','Test loss'],metric_rows()[2:],[135,91,91,91,91])
image('04_confusion_matrices.png','Figure 4. Predictions from the CV-selected traditional model and validation-selected ANN on the identical 231 reused comparison rows.',235)
report=pd.read_csv(OUT/(best_ann.lower().replace(' ','_')+'_classification_report.csv'),index_col=0)
table([best_ann,'Precision','Recall','F1','Support'],[
    [label,f'{report.loc[label,"precision"]:.4f}',f'{report.loc[label,"recall"]:.4f}',f'{report.loc[label,"f1-score"]:.4f}',int(report.loc[label,'support'])]
    for label in ['Lower','Middle','Higher']], [135,91,91,91,91])
p('The notebook includes full per-class reports for the selected traditional model and ANN. Class imbalance is modest, but weighted F1 alone can hide minority-class errors; macro F1 and the confusion matrix make the pattern visible. Boundaries between relative classes are sharp even where physical demand changes smoothly.')

page(); title('C.1-C.2 | Master comparison','What the results support')
table(['Model','Accuracy','Weighted F1','Macro F1','Loss'],metric_rows(),[135,91,91,91,91])
ann_acc=R.loc[best_ann,'test_accuracy']; ml_acc=R.loc[best_ml,'test_accuracy']
delta=100*(ann_acc-ml_acc)
p(f'Training CV selects <b>{escape(best_ml)}</b>. Validation loss selects <b>{escape(best_ann)}</b>. ANN minus selected traditional accuracy is <b>{delta:+.2f} percentage points</b>. The observed comparison-set leader is <b>{escape(M["observed_test_leader"])}</b>; validation accuracy, with loss as a tie-breaker, selects <b>{escape(M["validation_selected_candidate"])}</b> as the overall candidate.')
p(f'The paired 2,000-resample row bootstrap interval for the ANN-minus-traditional accuracy difference is [{100*M["paired_accuracy_interval"][0]:.2f}, {100*M["paired_accuracy_interval"][1]:.2f}] percentage points. The exact paired McNemar p-value is {M["paired_mcnemar_p"]:.4f}. These are descriptive calculations on reused, geometry-correlated rows; they cannot establish uncertainty for unseen building shapes.')
image('05_validation_importance.png','Figure 5. Random Forest validation permutation importance, with SD over 15 permutations. Correlated geometry predictors can substitute for one another; this is not a causal ranking.',230)

page(); title('Additional robustness | Different generalisation question','Hold out whole geometries')
p(f'The eight source variables describe only <b>{M["geometry_count"]} underlying building geometries</b>. All <b>{M["row_split_geometry_overlap"]}</b> occur in both development and comparison partitions. Orientation/glazing variants of familiar shapes can therefore make a random split look stronger than prediction for unseen shapes.')
p('Use three GroupKFold outer folds across the full dataset. In each outer training partition, reserve 20% of remaining geometry groups for validation; fit preprocessing and all models only on the remaining groups. Fixed RF (300 trees, unrestricted depth, leaf=1), fixed MLP (64/32, alpha=.01) and fixed ANN (batch 32) avoid importing the row-CV-selected settings. No grouped tuning is claimed.')
gs=G.groupby('model')[['accuracy','weighted_f1']].agg(['mean','std'])
table(['Fixed model','Accuracy mean','Accuracy SD','Weighted F1 mean'],[
    [name,pct(gs.loc[name,('accuracy','mean')]),f'{gs.loc[name,("accuracy","std")]:.4f}',f'{gs.loc[name,("weighted_f1","mean")]:.4f}'] for name in gs.index], [145,115,100,139])
image('06_geometry_stress.png','Figure 6. Three accuracy values per fixed model, each withholding complete geometries. Small fold count and the fixed PA2 label definition limit interpretation.',250)
p('<b>Do not deploy yet.</b> This simulation-only evidence lacks independent real-building validation, probability calibration, drift checks and operational cost assessment. A building-level external sample and a clear classification-versus-regression objective are needed before practical use.')

page(); title('C.3 | Reflection questions','Interpretation and next experiments')
h('How did architecture choices affect performance?')
p('Dense 64/32/16 layers learn nonlinear feature combinations; three softmax outputs match the target classes. Dropout 0.2 regularises the first two hidden layers. Batch size changes gradient noise and update frequency; validation histories provide direct evidence for that comparison. There was no width/depth or no-dropout ablation, so the results cannot quantify how much dropout or additional layers caused any improvement.')
h('Did deep learning outperform traditional ML? Why?')
p(f'The selected ANN scores {ann_acc:.2%} accuracy versus {ml_acc:.2%} for the selected traditional model on reused rows, a difference of {delta:+.2f} percentage points. The master table adds weighted/macro F1 and probability loss. A Random Forest can capture threshold interactions efficiently in small tabular data, while an ANN depends on optimisation and regularisation. These are plausible mechanisms, not isolated causal findings. The sklearn MLP is also a neural network. The dataset and validation design do not justify a universal claim of deep-learning superiority.')
h('With more time, what would improve performance?')
p('Collect independent real-building observations and evaluate complete unseen geometries. Use nested grouped cross-validation for model selection and repeat seeds. Test dropout and architecture ablations, probability calibration, ordinal penalties and sensitivity near the fixed cutoffs. Evaluate continuous heating-load regression if that better serves the practical objective. Any physical feature engineering should be fitted and assessed within the validation design, with Y1/Y2 kept out of predictors.')
h('Would the best model be deployed in production?')
p('No. Strong scores for familiar geometry do not establish performance for new building shapes or climates. The classes are dataset-relative rather than regulatory labels, and the 231 rows have been reused from an earlier assessment. External building-level validation, calibrated probabilities and defined error costs are required first.')
p('These are assistant-drafted technical reflections. They do not certify that the student personally derived the conclusions or completed the required in-person studio demonstration.','Smallx')

page(); title('E | Mandatory report-only template','GenAI Reflection')
h('1. GenAI Tools Used')
p('<b>OpenAI Codex (GPT-6-based agent in the Codex desktop app; exact served build not exposed):</b> retrieved the brief; inspected prior work; designed and generated the full notebook and supporting code; executed models; checked artifacts; drafted the interpretation, report and demo guide. No other generative AI tool is claimed. Python, scikit-learn, Keras, PyTorch and ReportLab were execution libraries.')
h('2. Specific Tasks & Prompts')
p('<b>Task 1:</b> access Canvas and complete Portfolio 3. Exact user prompt:', 'Smallx')
p(escape('"truy cập vào cavas swindburne của tôi, tìm kiếm assignment Portfolio Assessment 3: Advanced ML - Regression/Classification using ML and Deep Learning rồi hoàn thiện bài tập"'))
p('<b>Task 2:</b> locate the exact assignment. Exact URL supplied:', 'Smallx')
p('"https://swinburne.instructure.com/courses/78034/assignments/803620"')
p('<b>Task 3:</b> publish to the same GitHub repository and add links to the work. Exact user prompt:', 'Smallx')
p(escape('"đẩy lên github của tôi và gắn link vào bài làm như bài trươcs"'))
p('The user replied "ok" to using COS40007 temporarily as the class identifier. No separate student-written debugging prompts are invented. The full record is in GENAI_EVIDENCE.md.')
h('3. Code Attribution only for debugging and helping find errors')
p('<b>The actual use exceeds that category.</b> All new notebook modelling sections, experiment design, verification/inference/rendering scripts and drafted interpretations are substantially AI generated. Describing them as only debugging assistance would be inaccurate. The brief explicitly prohibits whole-code generation; this package is therefore labelled a reference. Disclosure alone does not remove that restriction or demonstrate independent student authorship.')
h('4. Learning Reflection')
p('<b>How could GenAI help learning?</b> The executed example makes fold-specific scaling, validation-based checkpointing and model reload checks inspectable. It can support a student-led explanation and reconstruction of the workflow.')
p('<b>What challenges did it create?</b> Full generated code exceeds the brief\'s allowed scope. The distinction between reusable row comparison and unseen-geometry validation also needs careful review. A Windows library-loading issue required compatible library versions; no security setting was changed.')
p('<b>What specific skills were developed?</b> Student skill development has not been assessed. Relevant skills to demonstrate are target definition, leakage control, five-fold CV, interpreting softmax/crossentropy, reading learning curves and reloading model artifacts. No personal learning experience, independent authorship or completed student verification is asserted.')

page(); title('B.5 and D | Reproducibility and repository','Artifacts that can be checked')
table(['Artifact','Content / verification'],[
    ['One Jupyter notebook','Full Parts A-C, D, references, executed code and actual stored outputs.'],
    ['random_forest.pkl; sklearn_mlp.pkl','Fitted preprocessing plus estimator; probabilities checked against exported rows.'],
    ['ann_batch16/32/64.h5 and .keras','Complete ANN variants; layer units, dropout rates and 3,747 parameters verified on reload.'],
    ['ann_preprocessor.pkl; metadata JSON','Fitted encoder/scaler, source feature order, class order, fixed thresholds and versions.'],
    ['ANN history .pkl / epoch CSVs','Training/validation losses and accuracy; epoch count and selected checkpoint.'],
    ['Metrics, predictions and figures','Comparison tables and probability-level evidence; independent metric recalculation.']], [183,316])
h('Exact execution environment')
table(['Component','Version'],[[k,v] for k,v in M['versions'].items()],[170,329])
p('Run: python -m pip install -r requirements.txt; then Restart Kernel and Run All from the notebook folder. The data are included. tools/predict.py demonstrates inference on a CSV containing X1-X8. tools/verify_analysis.py independently recalculates metrics and checks stored model predictions and executed cells.')
h('GitHub development record')
p(f'Code, the executed notebook, saved models, results and reproducibility files are available in <link href="{PA3}" color="#087f8c">portfolio/assessment-3 on the main branch</link> of <link href="{REPO}" color="#087f8c">odmiefnf/COS40007</link>, the same repository as Portfolio 1 and 2. Real local development milestone commits are retained in the integration history and attributed to OpenAI Codex. Publication was separately requested by the student; it does not establish independent student authorship. No history was backdated or invented.')
p('No Canvas submission has been performed. Canvas displayed a 1 November 23:59 due date and 5 November 23:59 availability cutoff when accessed on 2 November 2026. The printed May date in the option PDF is stale relative to the live assignment; displayed times use the Canvas account setting.','Smallx')

if (OUT/'evidence_architecture.png').exists():
    page(); title('Notebook evidence | B.2','Executed architecture summary')
    p('This screenshot captures the rendered output of the executed notebook cell. The full uncropped browser view, including source code, is also retained in output/evidence_architecture_full.png. It is shown separately from the structured table so that the original model output remains legible.')
    image('evidence_architecture.png','Screenshot 1. Actual saved notebook output viewed in a local browser. The screenshot has been cropped to the summary, with no changes to values.',470)
    p('The notebook shows the exact input shape, dropout and compilation code. Reload verification checks dense units [64, 32, 16, 3], dropout rates [0.2, 0.2] and 3,747 trainable parameters. The parameter count matches the manually explained architecture.')
if (OUT/'evidence_results.png').exists() and (OUT/'evidence_training.png').exists():
    page(); title('Notebook evidence | A.4 and B.3','Executed comparisons and checkpoints')
    p('Browser views of the saved notebook outputs; the comparison view displays the primary metric columns. Complete tables and raw values remain in the notebook and CSV files.','Smallx')
    image('evidence_results.png','Screenshot 2. Actual executed comparison table and selection outputs; displayed columns are a subset of the complete stored table.',335)
    image('evidence_training.png','Screenshot 3. Actual epoch counts, checkpoint epochs and validation results for all three batch sizes.',248)

page(); title('Sources and traceability','References and rubric map')
refs=[
    '[1] Tsanas, A. and Xifara, A. (2012). Energy Efficiency [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C51307. CC BY 4.0. https://doi.org/10.24432/C51307',
    '[2] Swinburne Canvas. Portfolio Assessment 3: Advanced ML - Regression/Classification using ML and Deep Learning. Course 78034, assignment 803620; option 3d PDF. Accessed 2 November 2026. https://swinburne.instructure.com/courses/78034/assignments/803620',
    '[3] Scikit-learn. RandomForestClassifier API. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html',
    '[4] Scikit-learn. Neural network models (supervised). https://scikit-learn.org/stable/modules/neural_networks_supervised.html',
    '[5] Keras. Training and evaluation with built-in methods. https://keras.io/guides/training_with_built_in_methods/',
    '[6] Keras. Callbacks API and ModelCheckpoint. https://keras.io/api/callbacks/ ; https://keras.io/api/callbacks/model_checkpoint/'
]
for text in refs: p(escape(text),'Smallx')
table(['Rubric / brief component','Evidence'],[
    ['Traditional models and visualisation','A.3-A.5; two pipelines, five-fold CV, master table and Figure 2.'],
    ['Deep-learning preparation','A.1-A.2 and B.1; source hash, exact rows, class counts, shape and fitted-only scaling.'],
    ['Architecture and training','B.2-B.3; exact 64/32/16 network, callbacks, 3 batch sizes, actual histories.'],
    ['Deep-learning evaluation','B.4; accuracy/loss, per-class report, confusion matrix and Figure 3 curves.'],
    ['Final summary / reflection','C.1-C.3; selected versus observed winners, limits and three reflection questions.'],
    ['Model saving','B.5; .h5, .pkl and native checkpoints; independent reload verification.'],
    ['GitHub integration','Direct link on the cover; complete code, models, results and retained milestone history.'],
    ['GenAI reflection','E; exact four-part template with truthful full-assistance disclosure.'],
    ['Studio demo','Student-led in-person performance remains outstanding (20 rubric points).']], [170,329])
p('The report explains actual computed artifacts; it does not certify compliance with the independent-authorship restriction. Read GENAI_EVIDENCE.md and the Vietnamese demo guide before deciding how to use the reference.','Smallx')

def footer(canvas,doc):
    canvas.saveState()
    w,h=A4
    canvas.setStrokeColor(TEAL); canvas.setLineWidth(1.4); canvas.line(48,h-28,w-48,h-28)
    canvas.setFont('Body',7.3); canvas.setFillColor(GRAY)
    canvas.drawString(48,h-20,'COS40007 | PORTFOLIO ASSESSMENT 3 | REFERENCE')
    canvas.setStrokeColor(colors.HexColor('#d8e4e7')); canvas.setLineWidth(.4); canvas.line(48,37,w-48,37)
    canvas.drawString(48,24,'Nguyen Van Tuan Kiet | 104772099 | 2 November 2026')
    canvas.drawRightString(w-48,24,str(doc.page))
    canvas.restoreState()
doc=SimpleDocTemplate(str(DEST),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=48,bottomMargin=49,
    title='Portfolio Assessment 3 - Building Heating-Demand Classification (Reference)',
    author='Nguyen Van Tuan Kiet; substantial AI assistance disclosed',subject='COS40007 Option 3d; actual executed results')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('Created',DEST)
