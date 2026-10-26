# CARE-ECG Reproducibility Status

**Status as of 2026-10-21: NOT REPRODUCED.**

The repository is a research reimplementation. It does not currently reproduce the
reported CARE-ECG results, and numerical proximity on one metric must not be described as
a reproduction.

## Fixed-Candidate Fold-10 Evidence

The completed experiment uses the official PTB-XL folds: folds 1-8 for base-model
training (17,084 eligible records), fold 9 for validation and meta-model selection
(2,146 eligible records), and every eligible record in the prepared fold-10 cohort for
the fixed-candidate evaluation (2,158 records).
The fixed class order is `NORM,MI,STTC,CD,HYP`. Labels are projected to the
highest-likelihood diagnostic superclass, with ties resolved by that order.

Raw PTB-XL v1.0.3 contains 21,799 rows. The declared single-label cohort contains
21,388 rows; 411 rows without an eligible diagnostic superclass are excluded. Raw fold
10 has 2,198 rows, of which 40 are excluded, so complete fixed-test coverage below means
all 2,158 eligible prepared fold-10 records rather than every raw database row.

The causal and report backbone is a supervised InceptionTime CausalVAE with 64 active
latent dimensions, multi-scale encoder features, balanced class weights, waveform
augmentation, label smoothing, and cosine learning-rate scheduling. It reached fold-9
macro-F1 0.6511 before calibration and 0.6694 after calibration. The backbone supplies
latent biomarkers, Bayesian-component sensitivity, counterfactuals, causal retrieval,
and deterministic reports; it is not the final diagnosis probability source.

The diagnosis source is a fold-9-selected regularized multinomial stack over three
models:

1. An SCP-aware HistGradientBoosting meta-model over 71 official XResNet1D-101 SCP
   scores plus 25 fixed ontology aggregates.
2. The official legacy PTB-XL benchmark XResNet1D-101 exp0 scores, collapsed and aligned
   by ECG ID.
3. A legacy PTB-XL benchmark XResNet1D-50 checkpoint adapter.

Five patient-grouped fold-9 splits selected stacker regularization 1.0 and class-weight
power 0.375. The selected out-of-fold estimate was accuracy 0.7628, macro precision
0.7040, macro recall 0.6952, and macro-F1 0.6988. This is a hyperparameter-selection
estimate over fixed base experts, not an independent nested estimate of the full system.
Seventeen additional meta-model candidates were screened; none improved the selected
three-source fold-9 macro-F1. Adding the InceptionTime classifier as a fourth diagnosis
source reduced it to 0.6861, so that source was not selected.

After selection was locked, the SCP meta-model was refit on folds 1-9 under the standard
stacked-learning refit convention. The fold-9 values above remain tied to the original
folds-1-8 model. The refit changes the deployment probability distribution, as recorded
in `artifacts/ptbxl-benchmark/meta-xresnet101-scp/locked_refit_folds1_9_report.json`.
The final artifact uses expert weight 1.0 and exact zero class offsets.

The lock predates inference for this current candidate, but fold 10 is not globally
untouched: aggregate fold-10 results from an earlier CARE-XResNet50 system were already
known, and label-free fold-10 source arrays already existed. The lock therefore supports
a fixed-candidate evaluation, not a pristine first look at fold 10. The full disclosure
is in `results/CARE_ECG_nextgen_locked_selection.json`.

| Metric | Fold-9 selection estimate | Eligible prepared fold-10 result | Paper target |
|---|---:|---:|---:|
| Accuracy | 0.7628145387 | 0.7683039852 | 0.70 |
| Macro precision | 0.7040317907 | 0.6887201483 | 0.68 |
| Macro recall | 0.6952392960 | 0.6773759589 | 0.72 |
| Macro F1 | 0.6987634162 | 0.6824523234 | 0.70 |

The new fold-10 run improves over the previous repository run by 0.0334 accuracy,
0.0204 macro precision, 0.0670 macro recall, and 0.0571 macro-F1. The direct diagnosis and
first-mention Text-to-SCP values are identical because
the deterministic template names its diagnosis first.

A subsequent waveform-specialist experiment searched 56,250 policies using folds 1-8 for
fitting and fold 9 for selection. Its selected validation policy improved accuracy,
precision, recall, and macro-F1 from `0.7628/0.7040/0.6952/0.6988` to
`0.7656/0.7159/0.6961/0.7038`. Its one sealed fold-10 evaluation produced
`0.7697/0.6888/0.6747/0.6806`, compared with `0.7683/0.6887/0.6774/0.6825` for the released
stack. The candidate was not promoted, and no fold-10-driven tuning was performed. The
comparison workbook is `results/CARE_ECG_model_improvement_experiment.xlsx` and its metadata
is `results/CARE_ECG_improved_run_metadata.json`.

These numbers are not protocol-equivalent to the paper. The paper maps generated LLaMA
3-8B text to SCP labels, while this repository uses a different label projection,
independent model artifacts, and a deterministic report generator. Numerical direction
is reported separately from exact reproduction, whose status remains **NOT REPRODUCED**.

The local report proxies are CRC 1.0, groundedness 1.0, context relevance 0.2767,
hallucination rate 0.0, and semantic relevance 0.8268. They are not paper-comparable:
the paper's evaluator implementations are unavailable, the final diagnosis posterior
has zero causal-component weight, and fused causal attribution availability is 0.0.
Component-level Bayesian attribution availability is 1.0 and is reported separately.

The PTB-XL knowledge graph contains 50 ontology nodes and 49 taxonomy edges used for
retrieval, with zero clinician-reviewed causal-prior edges. Taxonomy is not treated as
physiological causation; causal-model structure is learned over anonymous latent factors.

The auditable outputs are:

- `outputs/ptbxl-nextgen-ensemble-test.jsonl` - all 2,158 eligible prepared fold-10 predictions.
- `results/CARE_ECG_nextgen_results.xlsx` - final review workbook.
- `results/CARE_ECG_nextgen_run_metadata.json` - metrics, protocol, hashes, and
  comparability attestations.
- `results/CARE_ECG_nextgen_locked_selection.json` - pre-inference selection lock and
  prior-test disclosure.
- `artifacts/ptbxl-nextgen-final` - finalized causal/report artifact.

## Historical E0 Baseline

`texts.txt` records an E0 run on 200 PTB-XL fold-10 ECGs using five diagnostic
superclasses, folds 1-8 for training, fold 9 for validation, and
Llama-3-8B-Instruct. The paper reports the following PTB-XL results for its LLaMA
3-8B backbone:

| Metric | E0 in `texts.txt` | Paper, LLaMA 3-8B |
|---|---:|---:|
| Accuracy | 0.485, first diagnosis named | 0.70, Text-to-SCP mapping |
| Macro-F1 | about 0.24 for the Bayesian diagnosis model | 0.70 for mapped report text |
| CRC | 0.29 | 0.86 |
| Groundedness | 0.46 | 0.67 |
| Hallucination rate (HR) | 0.06 | 0.08 |
| Context relevance | not reported | 0.71 |
| SRS | not reported | 0.77 |

These are not all like-for-like measurements. In particular, the paper does not report
standalone Bayesian-network accuracy, while its accuracy/F1 values come from mapping
generated text to SCP labels. E0's report accuracy used the first diagnosis named. The
same E0 reports reach 0.88 when any diagnosis mention counts, demonstrating that the
undocumented mapping policy can reverse the conclusion. Both first-mention and
any-mention results must therefore be reported; any-mention is secondary and cannot be
substituted for the predeclared primary result.

The paper defines hallucination risk as

```text
HR = 1 - (1 / |retrieved facts|) * sum(fact is reflected in explanation)
```

This is retrieved-fact recall, not a check that the diagnosis is correct or that every
generated claim is supported. In E0, the verifier passed 102 of 103 reports with a wrong
diagnosis; its overall pass rate was 99.5%, and about 51% of released reports were wrong.
The E0 HR of 0.06 is therefore not evidence that the experimental result or clinical
behavior matches the paper. The current evaluator additionally reports wrong-diagnosis
pass rate and unsupported-claim rate so this failure mode remains visible.

## Unresolved Paper Protocol

The paper does not provide the trained CARE-ECG checkpoints, experiment scripts, exact
GPT-4 or LLaMA checkpoint/revision, PTB-XL split and label policy, Text-to-SCP aliases and
fuzzy threshold, VAE/BN hyperparameters, clinical-prior edges, indexed knowledge corpus,
or implementations of CRC, groundedness, and SRS. Those author artifacts, or a confirmed
protocol specifying the same information, are required for an exact reproduction. The
legacy PTB-XL benchmark XResNet101 adapter, SCP meta-model, legacy benchmark XResNet50
adapter, independent InceptionTime backbone, and deterministic report template are documented replacements,
not recovered CARE-ECG author artifacts.

The reported tables are also internally inconsistent. On Expert-ECG-QA, Table 3 gives
CARE-ECG precision/F1 of 0.79/0.83 for GPT-4 and 0.73/0.76 for LLaMA 3-8B. Table 6 calls
A4 the full system but reports 0.81/0.85 and 0.78/0.82, respectively. No different split
or evaluation protocol is stated. Tables 4-5 and Table 6 contain corresponding conflicts
for CRC, groundedness, HR, and SRS.

## Match Criteria

This project may change the status to **MATCHED** only when all of the following are true:

1. The model/checkpoint revision, prompt, decoding settings, knowledge corpus, graph
   priors, preprocessing, label projection, SCP aliases, fuzzy threshold, seeds, and test
   record IDs are frozen and archived with hashes.
2. The primary PTB-XL protocol is declared before evaluation. The repository protocol is
   five superclasses in the order `NORM,MI,STTC,CD,HYP`, official folds 1-8 (17,084
   eligible records), fold 9 (2,146), and fold 10 (2,158), maximum SCP likelihood per superclass,
   label-order tie breaking, and first-mention Text-to-SCP scoring. Any-mention scoring
   remains a separately labeled sensitivity analysis.
3. A complete run over the declared eligible prepared fold-10 cohort, not a selectively
   chosen subset, obtains all paper LLaMA PTB-XL targets within an absolute tolerance
   of 0.01: accuracy 0.70, precision 0.68,
   recall 0.72, macro-F1 0.70, CRC 0.86, groundedness 0.67, context relevance 0.71,
   HR 0.08, and SRS 0.77.
4. The run publishes the prediction JSONL, manifest, configuration, training history,
   model/data revisions, first- and any-mention metrics, fallback rate,
   wrong-diagnosis-pass rate, and unsupported-claim rate.
5. Any unavoidable replacement for an unpublished author artifact is identified as a
   deviation. A run using such replacements is a reimplementation result, not an exact
   reproduction, unless the authors confirm equivalence.

## Runnable Final PTB-XL Protocol

Install the PTB-XL, ensemble, and workbook dependencies. Point the preparation command
at an extracted PTB-XL directory containing `ptbxl_database.csv`,
`scp_statements.csv`, and the 100 Hz waveform folders:

```bash
python3 -m pip install -e '.[data,ensemble,reports]'

care-ecg prepare-ptbxl data/ptb-xl data/ptbxl-prepared \
  --sampling-rate 100 \
  --seconds 10 \
  --normalization none
```

Preparation writes `train.npz`, `validation.npz`, `test.npz`, `manifest.json`,
`scp_aliases.json`, and `knowledge_graph.json`. Record-level normalization is disabled
because the advanced configuration fits global normalization statistics on folds 1-8
only.

Train the InceptionTime causal/report backbone using the explicit fold-9 validation set:

```bash
care-ecg train data/ptbxl-prepared/train.npz artifacts/ptbxl-inception-validation \
  --config config/ptbxl_inception.yaml \
  --preprocessed \
  --label-names NORM,MI,STTC,CD,HYP \
  --validation-dataset data/ptbxl-prepared/validation.npz \
  --knowledge-graph data/ptbxl-prepared/knowledge_graph.json
```

The released backbone completed 20 epochs, selected epoch 12, and has model SHA-256
`ae32c49fa341fede8c503cd055df6d39648582197fa95edd8fced533827b2fad`.

The official legacy XResNet101 inputs come from
`https://github.com/helme/ecg_ptbxl_benchmarking` at commit
`cdbf4e66d7e57d9b6a2657b6024716212b8d0afa`. Legacy pickle-bearing arrays are opened
only after SHA-256 authentication. Build the fold-9 superclass adapter and the SCP-aware
folds-1-8 meta-model with the checked-in scripts:

```bash
.venv/bin/python scripts/build_official_xresnet101_expert.py \
  --predictions /path/to/upstream/y_val_pred.npy \
  --predictions-sha256 f8626f84dee06543d520dce605ef50f4b826db3146ba87a397081b3c1aee8a8e \
  --database /path/to/upstream/ptbxl_database.csv \
  --database-sha256 cd9bf4e6445c5e022605857c4d6f90c90ff0e62df847da52efaebe94de786156 \
  --statements /path/to/upstream/scp_statements.csv \
  --statements-sha256 ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f \
  --prepared-validation data/ptbxl-prepared/validation.npz \
  --output artifacts/ptbxl-benchmark/validation_probabilities_xresnet101_official.npz \
  --report artifacts/ptbxl-benchmark/xresnet101_official_adapter_report.json

.venv/bin/python scripts/fit_xresnet101_scp_meta.py \
  --train-dataset data/ptbxl-prepared/train.npz \
  --validation-dataset data/ptbxl-prepared/validation.npz \
  --official-train-predictions /path/to/upstream/y_train_pred.npy \
  --official-train-sha256 89609e49741f4e54a7112ef61984e79040382e542a35637ca27da4fec4a53ebc \
  --official-validation-predictions /path/to/upstream/y_val_pred.npy \
  --official-validation-sha256 f8626f84dee06543d520dce605ef50f4b826db3146ba87a397081b3c1aee8a8e \
  --official-database /path/to/upstream/ptbxl_database.csv \
  --database-sha256 cd9bf4e6445c5e022605857c4d6f90c90ff0e62df847da52efaebe94de786156 \
  --scp-statements /path/to/upstream/scp_statements.csv \
  --statements-sha256 ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f \
  --output-directory artifacts/ptbxl-benchmark/meta-xresnet101-scp
```

Select the three-source stack using fold 9 only. The legacy benchmark XResNet50
validation source is an authenticated checkpoint-adapter input:

```bash
.venv/bin/python scripts/fit_probability_stacker.py \
  --validation-dataset data/ptbxl-prepared/validation.npz \
  --validation-source xresnet101_scp_meta=artifacts/ptbxl-benchmark/meta-xresnet101-scp/validation_probabilities.npz \
  --validation-source xresnet101=artifacts/ptbxl-benchmark/validation_probabilities_xresnet101_official.npz \
  --validation-source xresnet50=artifacts/ptbxl-benchmark/validation_predictions_xresnet50.npz \
  --output-directory artifacts/ptbxl-benchmark/stack-three-source
```

The resulting stacker SHA-256 is
`09d52edf2dcffb24d41cfc442b8bb7ebd1bfcdeeb3138064d58dad3f7b3e2429`.
The checked-in selection lock records the chosen settings before current-candidate
fold-10 inference and discloses the earlier known aggregate. It must not be represented
as proof that fold 10 was globally untouched.

For the released replay, use the post-lock folds-1-9 SCP-meta refit and authenticate it
before deserialization. The inference script reads only record IDs and fold numbers from
the prepared test archive, never its labels:

```bash
.venv/bin/python scripts/apply_xresnet101_scp_meta.py \
  --model artifacts/ptbxl-benchmark/meta-xresnet101-scp/xresnet101_scp_hist_gradient_booster_refit_folds1_9.joblib \
  --model-sha256 b02def0566fe21569ebb677b703b8f735a0de434608845c687098ab52bad981b \
  --official-predictions /path/to/upstream/y_test_pred.npy \
  --official-predictions-sha256 8202066cda10db85a0f2b0af99aebce18763ce9fafe98ec1ab478b2f3672020a \
  --official-database /path/to/upstream/ptbxl_database.csv \
  --database-sha256 cd9bf4e6445c5e022605857c4d6f90c90ff0e62df847da52efaebe94de786156 \
  --scp-statements /path/to/upstream/scp_statements.csv \
  --statements-sha256 ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f \
  --prepared-split data/ptbxl-prepared/test.npz \
  --expected-fold 10 \
  --output artifacts/ptbxl-benchmark/meta-xresnet101-scp/test_probabilities_refit_folds1_9.npz \
  --allow-test-after-lock
```

Apply the locked stacker to the three aligned, label-free source tables:

```bash
.venv/bin/python scripts/apply_probability_stacker.py \
  --stacker artifacts/ptbxl-benchmark/stack-three-source/locked_probability_stacker.npz \
  --stacker-sha256 09d52edf2dcffb24d41cfc442b8bb7ebd1bfcdeeb3138064d58dad3f7b3e2429 \
  --prepared-split data/ptbxl-prepared/test.npz \
  --expected-fold 10 \
  --source xresnet101_scp_meta=artifacts/ptbxl-benchmark/meta-xresnet101-scp/test_probabilities_refit_folds1_9.npz \
  --source xresnet101=artifacts/ptbxl-benchmark/test_probabilities_xresnet101_official_locked.npz \
  --source xresnet50=artifacts/ptbxl-benchmark/test_probabilities_xresnet50_locked.npz \
  --output artifacts/ptbxl-benchmark/stack-three-source/test_probabilities_nextgen_locked.npz \
  --allow-test-after-lock
```

The stacker hash in the command above is
`09d52edf2dcffb24d41cfc442b8bb7ebd1bfcdeeb3138064d58dad3f7b3e2429`.
Finalize the CARE artifact. The finalizer independently recomputes the stack from the
three sources, verifies row and label alignment, forces exact zero class offsets, and
records all paths and hashes in `external_expert.json`:

```bash
.venv/bin/python scripts/build_locked_expert_artifact.py \
  --base-artifact artifacts/ptbxl-inception-validation \
  --output-artifact artifacts/ptbxl-nextgen-final \
  --locked-stacker artifacts/ptbxl-benchmark/stack-three-source/locked_probability_stacker.npz \
  --selection-report results/CARE_ECG_nextgen_locked_selection.json \
  --expert-probabilities artifacts/ptbxl-benchmark/stack-three-source/test_probabilities_nextgen_locked.npz \
  --source xresnet101_scp_meta=artifacts/ptbxl-benchmark/meta-xresnet101-scp/test_probabilities_refit_folds1_9.npz \
  --source xresnet101=artifacts/ptbxl-benchmark/test_probabilities_xresnet101_official_locked.npz \
  --source xresnet50=artifacts/ptbxl-benchmark/test_probabilities_xresnet50_locked.npz \
  --paper-artifact-deviation "Independent InceptionTime causal backbone with official legacy PTB-XL XResNet101 and XResNet50 adapters, an SCP-aware HistGradientBoosting meta-model, and a validation-locked linear stack; not an unpublished CARE-ECG author artifact." \
  --allow-test-after-lock \
  --overwrite
```

Generate, evaluate, and export every eligible prepared fold-10 record:

```bash
mkdir -p outputs results
care-ecg predict \
  artifacts/ptbxl-nextgen-final \
  data/ptbxl-prepared/test.npz \
  outputs/ptbxl-nextgen-ensemble-test.jsonl \
  --preprocessed \
  --batch-size 64 \
  --expert-probabilities \
    artifacts/ptbxl-benchmark/stack-three-source/test_probabilities_nextgen_locked.npz \
  --expert-weight 1.0 \
  --expert-source \
    "nextgen-three-source-stacker-sha256:09d52edf2dcffb24d41cfc442b8bb7ebd1bfcdeeb3138064d58dad3f7b3e2429"

care-ecg evaluate \
  outputs/ptbxl-nextgen-ensemble-test.jsonl \
  --scp-aliases data/ptbxl-prepared/scp_aliases.json \
  --scp-fuzzy-threshold 0.84

care-ecg export-results \
  outputs/ptbxl-nextgen-ensemble-test.jsonl \
  results/CARE_ECG_nextgen_results.xlsx \
  --metadata results/CARE_ECG_nextgen_run_metadata.json \
  --scp-aliases data/ptbxl-prepared/scp_aliases.json \
  --scp-fuzzy-threshold 0.84 \
  --tolerance 0.01
```

The label-free CARE classifier exporter is available for validation or post-lock test
inference when the internal neural probabilities are needed as a candidate source:

```bash
.venv/bin/python scripts/export_care_probabilities.py \
  --artifact artifacts/ptbxl-inception-validation \
  --prepared-split data/ptbxl-prepared/validation.npz \
  --expected-fold 9 \
  --output artifacts/ptbxl-benchmark/validation_predictions_care_neural.npz \
  --batch-size 64 \
  --preprocessed
```

For fold 10, add `--allow-test-after-lock`. This exporter deliberately never opens the
`labels` member of the prepared archive.

The final export uses first non-negated Text-to-SCP mention as the primary text mapping.
It equals direct diagnosis for this run because the deterministic template names its
diagnosis first. The workbook keeps numeric target direction separate from exact
reproduction and records the latter as **NOT REPRODUCED**.
