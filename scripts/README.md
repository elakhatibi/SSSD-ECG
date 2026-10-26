# Validation-Locked Ensemble Tools

These scripts implement the next-generation PTB-XL diagnosis ensemble while keeping
validation selection separate from fold-10 inference. Legacy official prediction
pickles and scikit-learn models are deserialized only after their SHA-256 hashes match
operator-supplied or pinned values.

## Protocol Boundary

- Base training: eligible records from official PTB-XL folds 1-8, 17,084 records.
- Validation and meta-model selection: eligible fold-9 cohort, 2,146 records and 1,917 patients.
- Fixed-candidate evaluation: every eligible prepared fold-10 record, 2,158 records.
- Label order: `NORM,MI,STTC,CD,HYP`.
- Stacker objective: pooled patient-grouped OOF macro-F1, then accuracy.
- Selected settings: regularization 1.0, class-weight power 0.375, zero class offsets.
- Final diagnosis policy: three-source expert weight 1.0.

The selection lock is `results/CARE_ECG_nextgen_locked_selection.json`, SHA-256
`fc5f38167c35bd7f3e99d4433f68724f7db18f670d7a96a9329639c245410889`.
It was written before inference for the current candidate. It also discloses that an
earlier CARE-XResNet50 aggregate fold-10 result was already known and that label-free
fold-10 source arrays already existed. Fold 10 is therefore not presented as globally
untouched.

Raw PTB-XL v1.0.3 contains 21,799 rows; 411 lack an eligible diagnostic superclass and
are excluded by the declared cohort projection. Fold 10 contains 2,198 raw rows and
2,158 eligible prepared rows, so full coverage here refers to that prepared cohort.

The selected fold-9 estimate is accuracy 0.7628145387, macro precision 0.7040317907,
macro recall 0.6952392960, and macro-F1 0.6987634162. The full fixed-candidate fold-10
result is 0.7683039852, 0.6887201483, 0.6773759589, and 0.6824523234, respectively.
These are reimplementation results; exact CARE-ECG paper reproduction remains
**NOT REPRODUCED**.

## Waveform-specialist improvement experiment

`fit_waveform_multiclass_specialists.py` fits class-specific gradient-boosting specialists
from 252 fixed waveform summaries plus eight demographic/missingness features. It searches
56,250 fold-9 policies, requires both recall and macro-F1 to improve over the locked stack,
writes the selection lock before fold-10 inference, and refits the selected specialists on
folds 1-9. `apply_waveform_multiclass_specialists.py` performs label-free inference;
`evaluate_locked_probabilities.py` permits one fold-10 evaluation; and
`export_model_improvement_experiment.py` creates the comparison workbook.

The selected validation policy improved accuracy/precision/recall/F1 from
`0.7628/0.7040/0.6952/0.6988` to `0.7656/0.7159/0.6961/0.7038`. Its sealed fold-10
recall/F1 were `0.6747/0.6806`, compared with `0.6774/0.6825` for the released stack, so
the candidate was not promoted. The audit-ready result is
`results/CARE_ECG_model_improvement_experiment.xlsx`; the full decision record is
`results/CARE_ECG_improved_run_metadata.json`.

## Locked Artifacts

| Artifact | SHA-256 |
|---|---|
| InceptionTime CARE model | `ae32c49fa341fede8c503cd055df6d39648582197fa95edd8fced533827b2fad` |
| Three-source stacker | `09d52edf2dcffb24d41cfc442b8bb7ebd1bfcdeeb3138064d58dad3f7b3e2429` |
| Folds-1-9 SCP meta-model | `b02def0566fe21569ebb677b703b8f735a0de434608845c687098ab52bad981b` |
| Fold-10 SCP-meta probabilities | `c49678af679455588e5f3f151981098b6ee53eda18285598eb4961e9fdb8d023` |
| Fold-10 XResNet101 probabilities | `c6a7a2e736f55912a6a68b1befe1a446d3613f7d9ff57966e9984b8ed571475e` |
| Fold-10 XResNet50 probabilities | `0821097fc6dd489bf5c05ce2e64127436ce871174e02d83fe3e7677e6209500a` |
| Fold-10 stacked probabilities | `5ba129ef27f2250adfaadec17c5e67d211f781e99597e9a78b448d095ac32f7d` |
| Final prediction JSONL | `c4adfd3bc242c080fd17ec3d49686cc7ee4bb62a3bc559cfe6aa762fa3375ecc` |

## Build Validation Sources

The official legacy XResNet101 scores come from
`https://github.com/helme/ecg_ptbxl_benchmarking` at commit
`cdbf4e66d7e57d9b6a2657b6024716212b8d0afa`. Substitute local paths to that release in
the commands below. The hashes authenticate the exact arrays and metadata tables used in
the completed run.

Build the direct fold-9 superclass adapter:

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
```

Fit the SCP-aware HistGradientBoosting model on folds 1-8 and apply it to fold 9. Its
fixed feature contract is 71 official SCP scores plus 25 ontology aggregates:

```bash
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

## Fit the Validation Stack

This command reads fold-9 labels for patient-grouped cross-fitting and hyperparameter
selection. It rejects test-named inputs and never opens fold 10:

```bash
.venv/bin/python scripts/fit_probability_stacker.py \
  --validation-dataset data/ptbxl-prepared/validation.npz \
  --validation-source xresnet101_scp_meta=artifacts/ptbxl-benchmark/meta-xresnet101-scp/validation_probabilities.npz \
  --validation-source xresnet101=artifacts/ptbxl-benchmark/validation_probabilities_xresnet101_official.npz \
  --validation-source xresnet50=artifacts/ptbxl-benchmark/validation_predictions_xresnet50.npz \
  --output-directory artifacts/ptbxl-benchmark/stack-three-source
```

The reported OOF metrics reuse the predictions that selected the stacker's
hyperparameters. They are selection estimates over fixed base experts, not an
independent nested estimate of the complete system.

## Export Label-Free CARE Probabilities

Use the internal InceptionTime classifier as a candidate source without exposing split
labels. Choose exactly one preprocessing mode. `--preprocessed` asserts that signals
already satisfy the saved artifact's lead, duration, sample-rate, and normalization
contract; otherwise supply their raw `--source-rate`.

```bash
.venv/bin/python scripts/export_care_probabilities.py \
  --artifact artifacts/ptbxl-inception-validation \
  --prepared-split data/ptbxl-prepared/validation.npz \
  --expected-fold 9 \
  --output artifacts/ptbxl-benchmark/validation_predictions_care_neural.npz \
  --batch-size 64 \
  --preprocessed
```

For fold 10 or any test-named input/output, add `--allow-test-after-lock` only after the
selection record is frozen. The split reader accesses `signals`, `record_ids`, and
`folds`, never `labels`. Output rows retain input order and include deterministic
inference metadata plus artifact-tree, config, and model digests. This source was
screened but not selected because the four-source fold-9 macro-F1 was 0.6860684609.

## Generate Post-Lock Fold-10 Sources

The released SCP meta-model was refit on folds 1-9 after validation lock. Its report at
`artifacts/ptbxl-benchmark/meta-xresnet101-scp/locked_refit_folds1_9_report.json`
separates the original fold-9 evidence from the refit's deployment-distribution audit.
Authenticate that model before label-free fold-10 application:

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

Apply the locked stacker to the three aligned label-free tables. Newly fit stackers
include `feature_names` and `source_names`; older NPZ candidates without those fields
are intentionally incompatible.

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

## Finalize and Evaluate

The finalizer verifies that the released expert probabilities are a recomputation of the
locked stacker over the named sources, checks row and label order, forces exact zero
offsets, and records provenance in `external_expert.json`:

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

Generate all 2,158 eligible prepared fold-10 reports and export the review workbook:

```bash
care-ecg predict \
  artifacts/ptbxl-nextgen-final \
  data/ptbxl-prepared/test.npz \
  outputs/ptbxl-nextgen-ensemble-test.jsonl \
  --preprocessed \
  --batch-size 64 \
  --expert-probabilities artifacts/ptbxl-benchmark/stack-three-source/test_probabilities_nextgen_locked.npz \
  --expert-weight 1.0 \
  --expert-source "nextgen-three-source-stacker-sha256:09d52edf2dcffb24d41cfc442b8bb7ebd1bfcdeeb3138064d58dad3f7b3e2429"

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

The final workbook path is
`results/CARE_ECG_nextgen_results.xlsx`. It separates directional
numeric comparisons from exact paper reproduction and records the latter as
**NOT REPRODUCED**.
