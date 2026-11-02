
### C.1 - Master results
The master table above contains both required traditional models and all three
ANN configurations, with accuracy, weighted/macro F1 and crossentropy on the same
231 rows. Traditional five-fold CV scores are reported separately from ANN validation.

### C.2 - Best model and deployment
**Random Forest** is the training-CV-selected traditional model. **ANN batch 16**
has the lowest ANN validation loss. Their reused comparison-set accuracies are
**93.07%** and **86.15%**: ANN minus traditional is **-6.93 percentage
points**. The observed comparison-set leader is **Random Forest**; the overall
candidate selected by validation accuracy (loss breaks ties) is **Random Forest**.
Selection criteria and observed test ranking answer different questions.

An ensemble can capture thresholds and interactions efficiently in a small tabular
dataset. A neural network learns smoother shared representations but needs optimisation
and regularisation. These mechanisms are plausible explanations, not experimentally
isolated causal claims. The paired row bootstrap interval for ANN minus traditional
is [-11.69, -2.16] percentage points, with exact paired
McNemar p=0.0090. Reused rows and correlated geometries limit inference.

**Production deployment is not justified.** All 12 building geometries appear
on both sides of the inherited row split. Fixed-model geometry-disjoint mean accuracies
are RF 64.06%, MLP 67.84%,
and ANN 69.27%. These stress results concern unseen geometries,
whereas random row scores concern familiar geometry with different configurations.
The data are simulated, labels are relative terciles, and no external building-level
validation, calibration or monitoring has been completed.

### C.3 - Architecture, comparison and next steps
**Architecture choices:** Dense 64/32/16 layers learn nonlinear combinations; three
softmax outputs match the classes. Dropout 0.2 regularises both large hidden layers.
Changing batch size changes update frequency and gradient noise. Validation results
compare these batch sizes only; there is no no-dropout or width ablation, so the
experiment cannot attribute improvement specifically to dropout or depth. At epoch
10 the histories show early learning; the saved checkpoint uses the best validation
loss within at most 50 epochs rather than assuming that the final epoch is best.

**Did deep learning win?** The selected ANN's accuracy difference is -6.93
percentage points on the reused row comparison. Compare probability loss and class
reports as well as accuracy. A single split cannot support a universal claim that
deep learning is superior; the scikit-learn MLP is also a neural network.

**With more time:** obtain independent real buildings and hold out complete geometries;
use nested grouped CV for tuning; repeat seeds; ablate dropout/width; check probability
calibration and ordinal errors near fixed cutoffs. If the practical objective is
continuous heating demand, compare regression with classification before operational
use. Engineer physically meaningful ratios only inside validated pipelines and assess
their incremental value rather than including Y1 or Y2 as predictors.
