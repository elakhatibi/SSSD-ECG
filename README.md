# CARE-ECG reproduction

This repository is an independent implementation of the pipeline described in
**“CARE-ECG: Causal Agent-based Reasoning for Explainable and Counterfactual ECG
Interpretation”**. It is not the authors' official code, and it is not a medical device.

The paper specifies the system at an architectural level but does not release code,
trained weights, data splits, preprocessing constants, graph priors, latent semantics,
knowledge documents, or most training hyperparameters. Consequently, this repository
implements the described modules while making replacement engineering choices explicit
in [`config/default.yaml`](config/default.yaml) and
[`config/ptbxl_inception.yaml`](config/ptbxl_inception.yaml). The completed prepared-cohort
PTB-XL experiment and a separately locked waveform-specialist improvement experiment are
reported with their validation and held-out measurements below. The strongest held-out result
remains the released three-source stack. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the
protocol boundary and complete provenance.

## Implemented architecture

The physiological stream is:

1. Configurable multi-lead preprocessing, resampling, and fixed-duration crop/pad.
2. A supervised temporal CausalVAE with an InceptionTime multi-scale encoder for the
   completed experiment; residual, BiLSTM, and Mamba backbones remain available.
3. An encoder-feature classifier trained jointly with the ELBO using KL warm-up,
   balanced class weights, label smoothing, five waveform augmentations, cosine learning
   rate decay, early stopping, and best-weight restoration.
4. Training-fold global normalization when configured, with fitted statistics saved for
   validation and inference.
5. Training-set quantile discretization of posterior-mean latent biomarkers.
6. Clinical-prior graph initialization plus greedy K2/BDeu structure refinement.
7. Bayesian-estimated categorical CPTs and exact variable elimination for
   `P(diagnosis | discrete biomarkers)`.
8. Bayesian-component driver ranking by local posterior sensitivity, with traceable
   directed paths and explicit attribution scope in every prediction.
9. Markov-blanket-constrained single-factor counterfactual search for a requested
   diagnosis, with exhaustive search retained where needed.
10. Batched neural encoding and deterministic saved-artifact inference.

The final diagnostic probabilities come from a validation-locked multinomial stack over
three sources: an SCP-aware HistGradientBoosting meta-model over legacy XResNet101
scores, the official PTB-XL benchmark XResNet1D-101 `exp0` predictions, and a legacy
PTB-XL benchmark XResNet1D-50 checkpoint adapter. Seventeen additional meta-model
candidates and the InceptionTime
probability source were screened on fold 9; none improved patient-grouped out-of-fold
macro F1. The InceptionTime model therefore supplies latent biomarkers, Bayesian
sensitivity, counterfactuals, retrieval, and reports, while the locked stack supplies
the final diagnosis. These external sources are replacements, not the CARE-ECG authors'
unavailable XResNet101 artifact.

The agentic stream is:

1. A History Agent that uses the latest patient-specific ECG, or the nearest latent
   surrogate when no longitudinal record exists, to report latent shifts.
2. A Diagnosis Agent that packages graph posterior, drivers, paths, and counterfactuals.
3. Diagnosis-aware causal RAG that enriches the query with driver states, retains
   generic and matching-class evidence, and excludes metadata-explicit competing-class
   facts.
4. A Response Agent using repository-defined fact-tagged prompt fields.
5. Evidence verification using both the paper's retrieved-fact HR and a separate
   unsupported-claim score.
6. A deterministic, evidence-only fallback when generated text fails either safety
   check, with the fallback and abstention states exposed in JSON.

The default generator is a deterministic offline report template. It differs from the
paper's LLaMA report generator. An optional OpenAI Responses API
adapter is included; it explicitly disables response storage and never sends raw ECG
samples. Operators remain responsible for privacy, governance, and model eligibility.

## Installation

Create an environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional integrations:

```bash
python -m pip install -e '.[data]'    # PTB-XL via pandas + WFDB
python -m pip install -e '.[rag]'     # sentence-transformer retrieval
python -m pip install -e '.[openai]'  # OpenAI Responses API
python -m pip install -e '.[local-llm]' # local Hugging Face / LLaMA-style model
python -m pip install -e '.[reports]' # Excel result export
python -m pip install -e '.[ensemble]' # validation-locked scikit-learn stackers
python -m pip install -e '.[dev]'     # pytest + ruff
```

Mamba is intentionally not a hard dependency because its supported CUDA/PyTorch matrix
is platform-specific. Install a compatible `mamba-ssm`, then set
`model.backbone: mamba`. The repository default is `resnet`; the BiLSTM path implements
the fallback explicitly named in the paper.

## Dataset contract

The generic training CLI accepts an NPZ file with:

- `signals` (required): float array `[N, 12, T]` or `[N, T, 12]` when
  `--source-rate` is supplied.
- `labels` (required): one categorical label per record. Strings are encoded in sorted
  order; integer labels can use `--label-names`.
- `patient_ids`, `timestamps`, `record_ids`, `folds` (optional): aligned arrays.

The paper writes `y` as a categorical diagnosis. Multi-label ECG records therefore
need a declared task projection (for example, PTB-XL superclass or highest-likelihood
SCP statement). `care_ecg.data.iter_ptbxl_diagnostic_superclasses` preserves the raw SCP
likelihoods used for the projection, and `records_to_arrays` performs an explicit
highest-likelihood projection over a supplied label order. This avoids silently treating
a multi-label benchmark as single-label.

Knowledge is JSONL, one object per line:

```json
{"doc_id":"guideline-1","text":"A paraphrased, validated clinical fact.","source":"stable source identifier or URL","metadata":{"topic":"rhythm"}}
```

The bundled five-document corpus only explains CARE-ECG's method and limitations. It is
for smoke tests, not clinical use. A production experiment must index licensed,
versioned, clinician-reviewed ECG guidelines, textbook excerpts, and SCP mappings.

## PTB-XL protocol

The completed experiment used 100 Hz waveforms and training-fold global normalization.
Install the data dependencies, extract PTB-XL, and build the fixed five-superclass
splits without record-level normalization:

```bash
care-ecg prepare-ptbxl data/ptb-xl data/ptbxl-prepared \
  --sampling-rate 100 \
  --seconds 10 \
  --normalization none
```

This writes eligible diagnostic-superclass records from folds 1-8 to `train.npz`
(17,084 records), fold 9 to `validation.npz` (2,146 records), and fold 10 to `test.npz`
(2,158 records). Raw PTB-XL contains 21,799 rows; 411 rows lack an eligible diagnostic
superclass and are excluded by this declared cohort definition. In fold 10, that is 40
of 2,198 raw rows. Preparation also writes a dataset manifest, `scp_aliases.json`, and
a metadata-derived knowledge graph. The projection
chooses the diagnostic superclass with the highest SCP likelihood in the fixed order
`NORM,MI,STTC,CD,HYP`; ties follow that order.

The generated PTB-XL graph has 50 ontology nodes and 49 taxonomy edges for retrieval.
It has zero clinician-reviewed causal-prior edges and explicitly sets
`causal_claims: false`. The experiment still learns a Bayesian network over anonymous
latent factors, but it does not relabel PTB-XL taxonomy links as physiological causes.

Train the InceptionTime internal model without allowing validation records to fit the
global normalizer, discretizer, or Bayesian network:

```bash
care-ecg train data/ptbxl-prepared/train.npz artifacts/ptbxl-inception-validation \
  --config config/ptbxl_inception.yaml \
  --preprocessed \
  --label-names NORM,MI,STTC,CD,HYP \
  --validation-dataset data/ptbxl-prepared/validation.npz \
  --knowledge-graph data/ptbxl-prepared/knowledge_graph.json
```

The ensemble utilities authenticate legacy arrays and serialized estimators by SHA-256,
fit the SCP-aware model, perform patient-grouped validation of the probability stack,
and apply it without opening fold-10 labels. The exact staged commands and NPZ contracts
are documented in [`scripts/README.md`](scripts/README.md). Selection is frozen in
[`results/CARE_ECG_nextgen_locked_selection.json`](results/CARE_ECG_nextgen_locked_selection.json)
before current-candidate fold-10 inference. The lock also discloses that an earlier
repository system's aggregate fold-10 result was already known, so fold 10 is not
presented as a globally untouched test set.

The finalizer copies the InceptionTime artifact, verifies the locked stack and all source
probabilities, sets zero post-selection class offsets, and records their hashes in
`artifacts/ptbxl-nextgen-final/external_expert.json`. Run that fixed artifact over all
2,158 eligible prepared fold-10 records and export the audited workbook with:

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

care-ecg evaluate outputs/ptbxl-nextgen-ensemble-test.jsonl

care-ecg export-results \
  outputs/ptbxl-nextgen-ensemble-test.jsonl \
  results/CARE_ECG_nextgen_results.xlsx \
  --metadata results/CARE_ECG_nextgen_run_metadata.json \
  --scp-aliases data/ptbxl-prepared/scp_aliases.json \
  --scp-fuzzy-threshold 0.84 \
  --tolerance 0.01
```

The export requires the `reports` optional dependency and writes the final workbook to
[`results/CARE_ECG_nextgen_results.xlsx`](results/CARE_ECG_nextgen_results.xlsx). Its
primary classification values use the first non-negated Text-to-SCP mention and equal
the direct diagnosis metrics because the deterministic report template states that
diagnosis first. This remains different from the paper's Text-to-SCP scoring of
LLaMA-generated reports.

For other external classifiers, the expert NPZ contract is strict:
`probabilities` has shape `[N,C]`, `record_ids` has shape `[N]` in exactly the same
order as the prediction dataset, and `label_order` has shape `[C]`. Probability rows
must be finite, nonnegative, and sum to one. Columns are reordered to the trained label
order after the label sets are verified as identical.

All three `predict` expert options are required together. The source, weight, file hash,
input distributions, and resulting probabilities are recorded under
`metadata.diagnosis_fusion.external_expert` for each prediction. The CLI never selects a
weight on the prediction set; compare and lock candidates using validation records
before generating test predictions.

Use `--preprocessed` only for arrays already produced under the configured lead,
sampling-rate, duration, normalization, and orientation contract. Otherwise provide
`--source-rate` and let the CLI perform preprocessing.

## Train and run

Copy and edit the default configuration, then train all learned components:

```bash
care-ecg train data/train.npz artifacts/run-1 \
  --config config/default.yaml \
  --knowledge data/ecg_knowledge.jsonl \
  --label-names NORM,AFIB,MI \
  --prior-edges data/clinical_prior_edges.json
```

`clinical_prior_edges.json` is a JSON list such as:

```json
[["z2", "diagnosis"], ["z7", "diagnosis"]]
```

Latents are anonymous unless they have been anchored using external biomarker
supervision. Do not name a latent “QTc” or “ST elevation” merely because it correlates
with that measurement.

Run the complete saved pipeline:

```bash
care-ecg infer artifacts/run-1 data/example.npy \
  "Which diagnosis is most supported, and why?" \
  --source-rate 500 \
  --counterfactual-target MI \
  --output outputs/example.json
```

Pass `--history data/prior_ecgs.npz --patient-id PATIENT_ID` to activate true
longitudinal comparison. If the history file contains no matching patient, the History
Agent uses the nearest latent record as the paper's surrogate-history fallback.

Use the optional API-backed response agent:

```bash
export OPENAI_API_KEY='...'
care-ecg infer artifacts/run-1 data/example.npy \
  "Which diagnosis is most supported, and why?" \
  --openai --model gpt-4
```

For a locally available LLaMA-compatible Hugging Face checkpoint, use
`--generator huggingface --model /path/to/checkpoint`. Model weights and access are not
bundled.

The model ID stays configurable because account availability changes. `gpt-4`,
temperature `0.3`, and a 600-token limit are repository defaults for the optional OpenAI
adapter; they do not reproduce the paper's LLaMA 3-8B generator.

For a no-data smoke run after installing PyTorch:

```bash
care-ecg demo --epochs 1 --output artifacts/demo
```

The demo uses synthetic sine patterns and has no medical meaning.

## Evaluation

Validation results used to select the completed system were:

| Validation component | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| InceptionTime, raw | 0.7335 | 0.6477 | 0.6619 | 0.6511 |
| InceptionTime, calibrated | 0.7423 | 0.6604 | 0.6879 | 0.6694 |
| Three-source stack, patient-grouped OOF | 0.7628 | 0.7040 | 0.6952 | 0.6988 |

The stack values cross-fit the stacker over fixed base-expert outputs on fold 9; they are
selection estimates, not a fully nested independent estimate of the complete system.
After locking the sources, regularization, class weighting, and zero class offsets, the
ensemble was evaluated on all 2,158 eligible records in the prepared fold-10 cohort:

| Metric | Held-out result | Paper target |
|---|---:|---:|
| Accuracy | 0.768303985171455 | 0.70 |
| Macro precision | 0.6887201482768676 | 0.68 |
| Macro recall | 0.67737595888676 | 0.72 |
| Macro F1 | 0.6824523233999099 | 0.70 |

All four values improve on the previous repository run: accuracy by 0.0334, macro
precision by 0.0204, macro recall by 0.0670, and macro F1 by 0.0571. The first-mention
Text-to-SCP values equal the direct classifier values
because the repository template emits its diagnosis first; the paper instead maps
free-form LLaMA reports. Different model artifacts, label projection details, report
generator, and metric implementations also prevent an exact protocol match. The final
status is therefore **NOT REPRODUCED**.

An additional 56,250-candidate waveform-specialist search used folds 1-8 for fitting and
fold 9 for selection. It improved validation accuracy/precision/recall/F1 from
`0.7628/0.7040/0.6952/0.6988` to `0.7656/0.7159/0.6961/0.7038`. The candidate was then
evaluated once on sealed fold 10, where recall/F1 were `0.6747/0.6806` versus
`0.6774/0.6825` for the released stack, so it was not promoted and no post-test tuning was
performed. The complete comparison is in
[`results/CARE_ECG_model_improvement_experiment.xlsx`](results/CARE_ECG_model_improvement_experiment.xlsx).

[`care_ecg.metrics`](src/care_ecg/metrics.py) implements:

- accuracy and macro precision/recall/F1;
- causal retrieval coverage (CRC);
- lexical groundedness and context relevance;
- the paper's retrieved-fact hallucination rate (HR);
- a separate unsupported-claim rate and lexical semantic-relevance proxy;
- negation-aware Text-to-SCP mapping with both strict first-mention and permissive
  any-mention scores;
- verifier pass rate and wrong-diagnosis pass rate.

CRC is comparable to the paper target only when both component and fused causal
attribution availability are 1.0. Records with no causal drivers receive CRC 0 rather
than vacuous perfect coverage. In the final run, local CRC is 1.0 and Bayesian-component
attribution availability is 1.0, but fused attribution availability is 0 because the
locked diagnostic expert has weight 1.0. CRC is therefore not comparable to the paper's
0.86 target.

The other final local report proxies are groundedness 1.0, context relevance 0.2767,
hallucination rate 0.0, unsupported-claim rate 0.0, semantic relevance 0.8268, verifier
pass rate 1.0, and wrong-diagnosis pass rate 1.0. They are deterministic
repository-defined lexical checks, not paper-comparable report measurements. The Excel
workbook keeps numeric proximity, directional targets, and exact-reproduction status in
separate fields so these distinctions are not collapsed into a single claim.

The input to `care-ecg evaluate predictions.jsonl` is one JSON object per prediction.
Classification metrics require `expected` and `diagnosis`; grounding metrics additionally
require `query`, `rationale`, and `retrieved_facts`. These lightweight overlap metrics
mirror the paper's descriptions, but the paper does not publish exact metric code or
thresholds. Clinician grading and waveform-criterion checkers remain necessary for a
credible clinical evaluation.

## Adaptive security evaluation

The proposal in `attack_models/Adaptive_Security_of_CARE_ECG_Proposal.docx` is implemented
under [`src/care_ecg/security`](src/care_ecg/security). It includes F0, F1, F1-CF, NES,
SimBA, Square, and adaptive K1/K2/K3 attacks; S-infinity, S2, spectral, and physiological
constraint sets; V1/V2/V3 defenses; E0-E7 experiment APIs; patient-level uncertainty; and
hash-chained query artifacts with a terminal manifest. The complete protocol and command
examples are in [`attack_models/README.md`](attack_models/README.md).

The physiological constraint is conditioned on each clean ECG rather than a fixed-rate template:
multilead consensus R peaks determine patient RR and wave timing, while the clean waveform supplies
lead-specific morphology and polarity. A nominal RR is retained only as a deterministic fallback
for traces without sufficient beat evidence.

The checked-in [validation](results/security-all-attacks-validation-final-v5-20261021/security_results.xlsx)
and [held-out](results/security-all-attacks-heldout-final-v5-20261021/security_results.xlsx)
security workbooks exercise E0/E1 only, with T1, budget 1, seed 42, and one
analysis-eligible case across all 36 attack/constraint arms. They are engineering integration
evidence, not primary clinical-security estimates. Primary
E0-E7 conclusions require the independent report adjudication, preserved-finding evidence,
intervention callbacks, transfer victim, and authenticated history specified by each API.

F0 uses constraint-native random draws whose magnitude is calibrated from non-F0 validation
submissions for each constraint and budget, then frozen into the validation lock. Held-out
runs bind the exact model tree, code tree, dataset hash, preprocessing contract, and fold
policy. The bundled signal-delta measurement checker supports engineering analysis only;
primary clinical IRR additionally requires independent all-claims clean-report review and
target-specific preserved-finding evidence through the Python API.

## Artifact layout

The final training, validation-selection, and locked-inference workflow writes:

```text
artifacts/ptbxl-inception-validation/
├── causal_vae.pt
├── causal_graph.json
├── validation_metrics.json
└── validation_predictions.npz

artifacts/ptbxl-nextgen-final/
├── manifest.json
├── config.yaml
├── causal_vae.pt
├── discretizer.json
├── causal_graph.json
├── knowledge_index.json
├── knowledge_graph.json
├── normalization.json
├── training_history.json
└── external_expert.json

artifacts/ptbxl-benchmark/
├── meta-xresnet101-scp/
│   └── xresnet101_scp_hist_gradient_booster_refit_folds1_9.joblib
└── stack-three-source/
    ├── locked_probability_stacker.npz
    └── test_probabilities_nextgen_locked.npz

outputs/
└── ptbxl-nextgen-ensemble-test.jsonl

results/
├── CARE_ECG_nextgen_locked_selection.json
├── CARE_ECG_nextgen_run_metadata.json
├── CARE_ECG_validation_meta_search.json
└── CARE_ECG_nextgen_results.xlsx
```

`external_expert.json` records the three source artifacts, locked stacker, probability
recomputation check, zero offsets, SHA-256 provenance, and the explicit deviation from
the paper's unavailable classifier artifact. The workbook contains aggregate and
per-class results, a confusion matrix, paper comparisons, run metadata, and the
reproduction checks used to retain the **NOT REPRODUCED** status.

All inference outputs include probabilities, continuous and discretized biomarkers,
ranked driver effects, graph paths, retrieved facts, hallucination score, fallback state,
history delta, optional counterfactual, and an explicit research-only warning.
