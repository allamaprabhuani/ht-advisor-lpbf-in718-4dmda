# Publication strategy: ML/DL contribution from the 4DMDA LPBF fatigue dataset

## Current evidence base in this folder

- Main spreadsheet: `ML LPBF 718 (29 04 2026).xlsx`
- Extracted mechanical table: `ml_project/data/sheet1_mechanical_raw.csv`
- Extracted composition table: `ml_project/data/sheet2_composition_raw.csv`
- Literature PDFs: 14 local PDFs in `downloaded_pdfs/`
- S-N figure mining assets: `ml_project/figures/SN_candidates.csv`, `ml_project/figures/sn_confirmed/`, and `ml_project/digitize_sn.py`

The structured table currently has 23 usable mechanical rows. Only 8 rows contain a count of fatigue-life points, and the composition table has only 1 populated alloy/powder row. This is not enough for a standalone deep neural network trained only on tabular data. The stronger contribution is a data-centric ML paper that combines:

1. structured process-property data from the spreadsheet,
2. digitised S-N fatigue data from the confirmed figures,
3. literature metadata such as heat treatment, surface condition, specimen orientation, stress ratio, temperature, and reference source,
4. uncertainty-aware ML rather than overclaiming deterministic prediction from a small dataset.

## Best publishable contribution

Recent literature already includes ML and physics-guided ML studies for AM/LPBF Inconel 718 fatigue-life prediction. Therefore, the paper should avoid presenting generic ML prediction as the novelty. The stronger publishable angle is reproducible data curation, S-N curve digitisation from literature figures, grouped validation by source paper, and uncertainty-aware prediction.

The availability of new fatigue-test data changes the strongest contribution. The work can be framed as a literature-to-lab learning problem: a model learns broad trends from published LPBF Inconel 718 data, then is adapted and validated on local specimens produced and tested under controlled laboratory conditions. This supports a stronger claim about domain shift, reproducibility, and data-efficient fatigue characterisation.

### Proposed title

**An uncertainty-aware machine-learning framework for fatigue-life prediction of laser powder bed fused Inconel 718 from heterogeneous literature data**

### Core novelty

The paper should not claim that the novelty is simply "using ML for fatigue prediction." That is already common. The defensible new contribution is:

- a curated, source-traceable LPBF Inconel 718 fatigue dataset assembled from structured tables and digitised S-N curves;
- a reproducible pipeline for converting literature figures into machine-readable fatigue data;
- a physics-informed feature representation for heat treatment, orientation, surface state, stress ratio, temperature, and tensile properties;
- uncertainty-aware fatigue-life or fatigue-strength prediction that quantifies when the model is extrapolating;
- model interpretation showing which processing and post-processing variables most strongly affect fatigue performance.

This is more publishable than a black-box deep learning model because the available dataset is small and heterogeneous.

## Experimental-data leverage strategy

### Delta-learning for literature-to-lab domain shift

#### Problem

Literature fatigue data are heterogeneous because different groups use different machines, powder batches, scan strategies, specimen geometries, surface states, and fatigue-test procedures. A model trained on literature data may therefore show a systematic bias when applied to local LPBF specimens.

#### Novelty

Train a global model on literature-derived data, then use local experimental data to learn the residual error between the global prediction and the measured local result:

`Delta = y_local - y_global`

The final prediction becomes:

`y_corrected = y_global + Delta_model(local_process_features)`

#### Scientific claim

This directly quantifies the domain shift between published AM fatigue data and local manufacturing/testing conditions. That is a stronger contribution than reporting prediction accuracy alone because it addresses reproducibility and transferability in AM fatigue modelling.

#### Requirements

- Literature data with source identifiers and condition metadata.
- Local experiments with the same core feature schema.
- Validation that separates literature-only training, local-only training, and literature-plus-delta correction.

### Active learning for fatigue-test selection

#### Problem

Fatigue testing is expensive and slow. A fixed test matrix may waste tests in regions where the model is already confident and may undersample regions where uncertainty is high.

#### Novelty

Use the literature-trained model to predict the S-N response for the planned local material state before testing. Select stress amplitudes, temperature conditions, or heat-treatment cases where predictive uncertainty is largest or where model alternatives disagree.

#### Scientific claim

The study can demonstrate an AI-guided experimental workflow that reduces the number of fatigue tests needed to characterise a new LPBF material condition.

#### Practical implementation

- Fit an uncertainty-aware model on literature data.
- Predict S-N curves for the local heat-treatment candidates.
- Select fatigue stress levels using uncertainty sampling, expected model improvement, or targeted uncertainty reduction around design-relevant lives such as `10^5`, `10^6`, or `10^7` cycles.
- Compare the information gained from active-learning-selected tests with a conventional evenly spaced stress-level design.

### Transfer learning through pretraining and fine-tuning

#### Problem

Local fatigue datasets may contain only a small number of tests, but the broader literature contains useful information about stress-life behaviour, heat treatment, surface condition, and temperature effects.

#### Novelty

Pretrain a neural or hybrid model on digitised literature S-N data, then fine-tune selected layers or condition-dependent parameters using local fatigue results.

#### Scientific claim

The paper can test whether process-property-fatigue representations learned from the broader AM literature can be adapted to a local manufacturing environment with limited new data.

#### Caution

This should be presented as transfer learning only if the pretrained model and fine-tuned model are both implemented and compared against baselines. If the local dataset remains very small, delta-learning with uncertainty-aware classical ML may be more robust than deep transfer learning.

### Revised title option

**Bridging literature and laboratory fatigue data: a transfer-learning and delta-correction framework for LPBF Inconel 718 with experimental validation**

## Recommended ML problem formulation

### Primary target

Predict fatigue strength at fixed lives, for example:

- stress amplitude at `10^5` cycles,
- stress amplitude at `10^6` cycles,
- stress amplitude at `10^7` cycles where available.

This is better than directly predicting raw cycles-to-failure because S-N data from different papers are sparse, censored, and measured under different stress levels.

### Secondary target

Predict log fatigue life:

`log10(Nf) = f(stress amplitude, R ratio, temperature, surface condition, orientation, heat treatment, UTS, YS, elongation, hardness, grain size)`

Use this only after enough S-N points have been digitised.

## Recommended model stack

Use a tiered model comparison:

1. Baseline empirical S-N model: Basquin-type regression per condition where enough points exist.
2. Classical ML: Random Forest, XGBoost/LightGBM or Gradient Boosting, Support Vector Regression.
3. Uncertainty model: Gaussian Process Regression, Quantile Regression Forest, or conformal prediction wrapped around gradient boosting.
4. Optional deep learning contribution: small tabular neural network or neural additive model only after expanding the digitised dataset; report it honestly as a benchmark, not the main claim.

The most defensible "deep learning" element is not a large predictor trained on 23 rows. It is a computer-vision-assisted digitisation workflow for extracting S-N points from published figures, followed by uncertainty-aware ML on the enlarged dataset.

## Feature engineering that can create a scientific contribution

Create features that reflect metallurgy rather than raw text labels:

- `build_orientation`: vertical, horizontal, diagonal, unknown.
- `post_processing`: as-built, stress relieved, solution treated, direct aged, HIP, HIP+solution+aged.
- `heat_treatment_temperatures`: max temperature, aging temperature, aging duration, number of thermal steps.
- `surface_condition`: as-built, machined, polished, shot-peened, surface treated.
- `test_temperature_C`: room temperature, 550 C, 650 C, etc.
- `stress_ratio_R`.
- tensile proxies: UTS, YS, elongation, hardness, grain size.
- source/reference ID for grouped validation.

Grouped validation by paper/source is important. Random row splits may overestimate performance because points from the same figure or paper are not independent.

## Specific paper structure

### Abstract claim

This study develops a source-traceable machine-learning framework for fatigue-property prediction of LPBF Inconel 718 using heterogeneous literature data. Structured mechanical-property tables are combined with digitised S-N curves from local literature PDFs. The framework compares empirical S-N fitting, tree-based ML, and uncertainty-aware regression under leave-one-source-out validation. Model interpretation is used to quantify the influence of post-processing, surface condition, build orientation, stress ratio, and tensile properties on fatigue resistance.

### Research questions

1. Can heterogeneous literature fatigue data for LPBF Inconel 718 be harmonised into a usable ML dataset?
2. Which process and post-processing variables dominate fatigue strength after accounting for stress ratio and temperature?
3. Does uncertainty-aware ML outperform empirical Basquin-style models under leave-one-paper-out validation?
4. Where are the largest data gaps that prevent reliable fatigue-life prediction?

### Methods

1. Data curation from the two Excel workbooks and 14 downloaded PDFs.
2. Semi-automated S-N curve detection and digitisation using the existing `digitize_sn.py` workflow.
3. Unit harmonisation and metadata encoding.
4. Model training with grouped cross-validation by source paper.
5. Model interpretation using SHAP/permutation importance and partial dependence.
6. Uncertainty quantification using Gaussian Process Regression, quantile models, or conformal prediction.

### Expected figures

- Dataset map: number of S-N points by paper, heat treatment, orientation, surface state, and temperature.
- S-N curve reconstruction examples from confirmed figures.
- Predicted versus measured fatigue strength with leave-one-source-out validation.
- Uncertainty bands showing extrapolation regions.
- Feature-importance plot linking fatigue performance to HIP, solution/aging, surface treatment, and tensile strength.

## Alternative contribution angles

### Proposal 0: AI expert system for heat-treatment recipe recommendation

This is a strong direction because it moves from passive prediction to engineering decision support.

#### Proposed title

**A physics-informed AI expert system for heat-treatment recipe recommendation in additively manufactured Inconel 718**

#### Problem

AM metallic parts often require post-processing routes such as stress relief, HIP, solution treatment, direct ageing, double ageing, machining, polishing, or surface treatment. The correct recipe depends on alloy, build condition, surface state, target property, temperature regime, and defect/microstructure state. Trial-and-error heat-treatment optimisation is expensive, and purely data-driven models are weak when experimental data are sparse.

#### Novelty

Develop an inverse-design AI expert system that recommends heat-treatment recipes instead of only predicting properties. The system should combine:

- empirical metallurgy rules,
- literature-derived experimental data,
- fatigue and tensile-property prediction models,
- uncertainty estimation,
- explainable recommendations.

#### How it would work with the current folder

- Inputs: alloy, AM process, build orientation, surface state, available tensile properties, target fatigue/property requirement, and operating temperature.
- Knowledge base: empirical rules for HIP, solution treatment, ageing, surface treatment, grain coarsening, precipitation strengthening, defect closure, and surface-driven fatigue debit.
- Prediction layer: ML models estimating UTS, YS, elongation, hardness, fatigue strength, or S-N curve parameters.
- Recommendation layer: ranks possible recipes such as ST+DA, HIP+DA, HIP+ST+DA, machining+DA, or surface treatment routes.
- Output: recommended heat-treatment sequence, estimated property outcome, confidence level, and explanation.

#### Scientific claim

The contribution is a hybrid physics-informed recommender that uses sparse literature data without pretending that a black-box model has learned the full process-property-fatigue relationship.

### Proposal 1: Physics-Informed Neural Networks for S-N curve prediction

#### Proposed title

**Physics-informed deep learning for robust fatigue-life prediction of LPBF Inconel 718 from sparse literature data**

#### Problem

Traditional ML models often treat fatigue prediction as a pure regression task. With scarce AM fatigue data, standard neural networks can overfit and may predict physically inconsistent behaviour, such as fatigue life increasing as stress amplitude increases or unrealistic S-N curve shapes.

#### Novelty

Introduce a Physics-Informed Neural Network (PINN) or physics-guided neural model that embeds known fatigue laws, such as Basquin or Coffin-Manson relationships, directly into the loss function. The physics constraint regularises the model when experimental data are sparse.

#### How it works with the current data

- Inputs: stress amplitude, stress ratio, temperature, UTS, YS, hardness, elongation, composition where available, orientation, surface condition, and heat-treatment features.
- Output: predicted number of cycles to failure, `Nf`, or predicted S-N curve parameters.
- Data loss: error between measured/digitised fatigue points and predicted values.
- Physics loss: penalty when the predicted S-N curve violates expected Basquin-type monotonic behaviour or deviates strongly from a valid log stress-log life relationship.
- Optional constraints: fatigue strength should not increase with increasing stress amplitude; predictions should remain within plausible tensile-strength bounds; endurance-limit behaviour can be enforced where appropriate.

#### Recommended formulation

For high-cycle fatigue, use a Basquin-guided model:

`log10(Nf) = a(condition) + b(condition) * log10(sigma_a)`

The neural network predicts condition-dependent `a` and `b`, rather than directly predicting every point without structure. This keeps the model physically interpretable and reduces overfitting.

#### Risks and controls

- Do not call it a full PINN unless the physics residual is clearly defined and implemented.
- Use leave-one-paper-out or leave-one-condition-out validation.
- Compare against a plain neural network, random forest/boosting model, and Basquin regression baseline.

### Supporting angle: Automated literature-figure digitisation for AM fatigue

Focus the paper on the computer-vision pipeline: detecting S-N figures in PDFs, calibrating axes, extracting marker points, and producing a reusable dataset. This is strong if the digitisation workflow is made robust and validated against manually extracted points.

### Proposal 2: Multi-modal text-to-fatigue architecture using NLP and regression

#### Proposed title

**A multi-modal deep learning framework bridging textual heat-treatment protocols and mechanical properties for AM fatigue-life prediction**

#### Problem

Heat treatments in AM are complex process sequences. In the current spreadsheet, heat treatments appear as text such as:

`1 h at 980C + air cool --> 8 h at 720C + furnace cool to 620C --> 8 h at 620C + air-cooled to room temperature`

Standard ML struggles with this because the heat-treatment route is not a simple categorical variable. The number of steps, temperatures, hold times, cooling rates, and sequence order all affect microstructure and fatigue performance.

#### Novelty

Use a multi-modal model that combines a text branch for heat-treatment protocols with a numerical branch for mechanical, composition, and test-condition data.

#### How it works with the current data

- Text branch: encode the `heat_treatment` column using a domain-specific parser, pretrained language model embedding, 1D-CNN, LSTM, or transformer-style encoder.
- Numerical branch: use UTS, YS, modulus, hardness, elongation, composition, R ratio, temperature, orientation, surface condition, and stress amplitude.
- Fusion layer: concatenate text embeddings with numerical features.
- Output: fatigue-life prediction, fatigue strength at fixed cycles, or S-N curve parameters.

#### More defensible version

Because the current table is small, the first implementation should not rely only on generic NLP embeddings. A stronger approach is:

1. parse heat-treatment text into structured features such as max temperature, total hold time, number of thermal steps, ageing temperature, HIP flag, pressure, cooling method, and sequence order;
2. compare this structured parser against simple text embeddings;
3. use the comparison to show whether NLP adds value beyond metallurgical feature engineering.

#### Risks and controls

- A large language model embedding may overfit if the dataset remains small.
- The paper should report ablation tests: no heat-treatment feature, categorical heat-treatment feature, parsed heat-treatment features, and text-embedding features.
- The text-to-fatigue claim becomes credible only after digitised S-N data increase the number of labelled examples.

### Supporting angle: Physics-informed fatigue-life ML

Combine Basquin parameters with ML residual correction. The model first fits an S-N relationship and then learns residual effects from heat treatment, orientation, surface state, and tensile properties. This is scientifically stronger than a generic black-box regressor.

### Proposal 3: Transfer learning from wrought/conventional to AM Inconel 718

#### Proposed title

**Bridging the gap: deep transfer learning from conventional to laser powder bed fusion Inconel 718 for accelerated fatigue modelling**

#### Problem

LPBF fatigue data are expensive and scarce, but conventional wrought/cast Inconel 718 has a larger historical fatigue-property literature. A model trained only on LPBF data may overfit, while a model trained only on wrought data may not capture AM-specific defects, anisotropy, surface roughness, and heat-treatment effects.

#### Novelty

Train a base model on a larger conventional Inconel 718 fatigue dataset, then fine-tune it using the LPBF dataset extracted from the current Excel workbook and digitised S-N curves. The research question is whether transferable fatigue representations can reduce the amount of AM-specific data needed.

#### How it works with the current data

- Source task: conventional/wrought/cast Inconel 718 fatigue-life or S-N curve prediction.
- Target task: LPBF Inconel 718 fatigue prediction.
- Shared inputs: stress amplitude, temperature, R ratio, UTS, YS, hardness, heat treatment, and composition.
- AM-specific inputs: build orientation, surface state, defect descriptors where available, HIP, and AM process metadata.
- Transfer strategy: pretrain on source data, freeze early layers or use feature embeddings, then fine-tune on LPBF data.

#### Validation design

- Baseline 1: train only on LPBF data.
- Baseline 2: train only on conventional data and test on LPBF data.
- Proposed model: pretrain on conventional data and fine-tune on LPBF data.
- Evaluation: leave-one-source-out validation and uncertainty/error analysis on LPBF data.

#### Risks and controls

- This proposal requires a separate conventional Inconel 718 fatigue dataset. It is not yet present in the folder.
- Domain shift must be handled explicitly because AM fatigue is strongly affected by defects, surface roughness, residual stress, and anisotropy.
- If the conventional dataset cannot be assembled, this should remain a future-work extension rather than the main paper.

### Supporting angle: High-temperature fatigue transfer learning

Use room-temperature fatigue data as the source domain and 550-650 C data as the target domain. This is promising because the folder includes high-temperature fatigue references, but it needs enough digitised points.

### Supporting angle: Defect and microstructure image learning

Possible only if the extracted PDF figures include enough SEM/defect images with matched fatigue outcomes. At present this is higher risk because labels may not map cleanly from image to specimen.

## Minimum dataset needed before writing results

For a conference paper:

- at least 80-150 digitised S-N points,
- at least 8-12 distinct source conditions,
- clear metadata for orientation, heat treatment, surface condition, R ratio, and temperature,
- grouped validation by source paper or condition.

For a journal paper:

- 250+ digitised fatigue points if possible,
- stronger metadata completeness,
- validation against manually digitised reference points,
- uncertainty quantification and comparison to empirical fatigue models.

## Immediate next steps

1. Use the confirmed S-N figures in `ml_project/figures/sn_confirmed/` to digitise fatigue points into CSV files.
2. Build a single harmonised dataset with one row per fatigue test point.
3. Add metadata columns from the Excel sheet and paper titles.
4. Fit baseline Basquin models and grouped ML models.
5. Prepare a 4DMDA abstract around the data-centric ML framework and early results.
