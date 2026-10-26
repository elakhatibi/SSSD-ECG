# Adaptive Security Evaluation

This directory contains the research proposal and the runnable CARE-ECG implementation of
its waveform-only threat model. The implementation treats the complete waveform-callable
internal CARE pipeline as the success oracle. Differentiable surrogates guide optimization,
but never decide whether an attack succeeded.

## What Is Implemented

| Proposal item | Implementation |
| --- | --- |
| F0 random control | constraint-native draws with per-constraint/budget validation-matched magnitude |
| F1 white-box diagnosis attack | targeted L-infinity/L2 PGD in raw mV |
| F1-CF | PGD plus a continuous counterfactual latent-bin objective |
| F2 black-box diagnosis attack | antithetic NES, SimBA coordinate search, and Square-style search |
| F3 adaptive attack | K1 decision feedback, K2 grey-box, and K3 verifier-aware objectives |
| S-infinity / S2 | projected raw-mV norm constraints |
| S-spec | amplitude, spectral-energy, and temporal-slew constraints |
| S-phys | clean-conditioned, bounded P/QRS/T amplitude and timing, drift, and RR-jitter parameterization |
| V1 / V2 / V3 | shipped, independently measured, and authenticated longitudinal verifiers |
| Clinical preservation | target-aware checker contract with explicit primary/exploratory endpoint status |
| Query protocol | hard complete-system budgets and first-incorrect-release stopping |
| Audit trail | hash-chained JSONL plus separately hashed raw-mV NPZ waveforms |
| Statistics | patient bootstrap IRR, paired delta IRR, rounds-to-pass, coverage, and selective risk |

The public implementation lives in `src/care_ecg/security/`. Its central success predicate is:

1. The final pipeline output is released.
2. The released text contains the target clinical claim, with negation handled by the claim oracle.
3. The candidate is in the declared waveform constraint set.
4. Independent evidence names the same patient, record, and target and explicitly identifies the
   preserved baseline findings.

Every full pipeline submission consumes one query, including rejected and clinically inadmissible
candidates. E1 and E5 wrap the victim with an independent counter, reject any mismatch between
victim calls and the returned attempt trace, and recompute every submitted waveform's membership
against the exact constraint instance passed to the attack. E5 additionally wraps each verifier
in an orchestrator-owned observer: every victim query must call that verifier exactly once, return
its typed `ReleaseDecision`, and return the exact clinical snapshot that was checked with the same
release state and score. A second isolated verifier independently replays every decision on the
same snapshot and context. The clean and selected adversarial snapshots are canonicalized and
bound to the E0 input and recorded query output before downstream scoring. The attacker receives a
separate defense-configuration copy, while the victim-bound threshold is locked for the complete
run.

S-phys derives a deterministic patient profile from the unmodified raw-mV input: robustly
normalized leads vote on consensus R peaks, local RR intervals place the P/QRS/T windows, and the
observed waveform supplies lead-specific morphology and polarity. Its eight bounded parameters
remain P/QRS/T amplitude, P/QRS/T timing, baseline drift, and alternating-beat RR jitter. The
configured nominal RR is used only when the clean trace does not contain enough beat evidence.
NumPy and Torch rendering share this canonical clean-signal basis, including an interior float32
projection margin, so serialized attack waveforms replay against the same membership oracle.

`AttackResult.success` and the primary IRR summaries are reserved for this full definition. The
bundled local measurement checker compares signal-derived deltas only, so its candidate releases are
recorded as `exploratory_success` and never promoted to primary successes.

## Run E0 And E1

Install the report extra for Excel output, then run validation fold 9:

```bash
pip install -e '.[reports]'
care-ecg security-run \
  artifacts/ptbxl-nextgen-final \
  data/ptbxl-prepared/validation.npz \
  results/security-validation \
  --security-config config/security_ptbxl.yaml \
  --phase validation
```

The default configuration is the full F0/F1/F1-CF/F2/F3 by
S-infinity/S2/S-spec/S-phys matrix at budgets 1, 10, 50, and 100. It is intentionally large.
Validation executes non-F0 arms first, freezes their median native perturbation distance for
each constraint and budget, and only then runs F0. Held-out F0 reads those values from the lock.
A short end-to-end engineering run can be made explicit without changing the primary config:

```bash
care-ecg security-run \
  artifacts/ptbxl-nextgen-final \
  data/ptbxl-prepared/validation.npz \
  results/security-smoke \
  --phase validation \
  --target-ids T1,T3 \
  --attacks f0,f1_pgd,f2_square,f3_k3 \
  --constraints s_inf,s_phys \
  --budgets 1,10 \
  --seeds 42 \
  --max-candidates-per-target 2 \
  --max-eligible-per-target 1
```

Outputs are:

- `security_results.json`: protocol manifest, E0 observations, E1 observations, and patient-level summaries.
- `security_results.xlsx`: concise manifest, summary, E0, and E1 worksheets.
- `security_events.jsonl`: append-only, hash-chained query events.
- `security_events.jsonl.manifest.json`: terminal record-count, record-hash, and log-hash seal.
- `security_events_waveforms/`: compressed raw-mV waveform artifacts with content hashes.

Verify the audit trail independently:

```bash
care-ecg security-verify-log results/security-validation/security_events.jsonl
```

Completed all-arm engineering-validation workbooks are available at
`results/security-all-attacks-validation-final-v5-20261021/security_results.xlsx` and
`results/security-all-attacks-heldout-final-v5-20261021/security_results.xlsx`. Each covers T1,
budget 1, seed 42, one analysis-eligible case, nine attack implementations, and four
constraints. All 36 submitted attack waveforms in each run satisfy their declared set and both
event logs have valid terminal manifests. These runs are E0/E1 integration evidence, not primary
clinical-security estimates: their local report and preservation evaluators are exploratory, so
primary IRR is correctly recorded as not evaluable rather than zero.

V2 can be evaluated with `--defense v2`. It uses the NumPy measurement implementation for the
release gate and a separately instantiated SciPy implementation for exploratory clinical deltas.
The artifact-plus-NPZ CLI also uses an exploratory diagnosis/target-absence E0 heuristic because the
prepared NPZ has no independently adjudicated all-claims report labels. These classifications are
written into JSON, Excel, and the event log.

## Freeze Before Fold 10

Create a lock from the completed validation result. The lock independently verifies the sealed
query log and binds the YAML protocol, validation result, model tree, code tree, validation dataset,
preprocessing contract, cohort controls, and F0 calibration by SHA-256:

```bash
care-ecg security-lock \
  config/security_ptbxl.yaml \
  results/security-validation/security_results.json \
  results/security-validation/security_protocol.lock.json
```

Held-out evaluation refuses to run without that lock and rejects changed inputs:

```bash
care-ecg security-run \
  artifacts/ptbxl-nextgen-final \
  data/ptbxl-prepared/test.npz \
  results/security-heldout \
  --security-config config/security_ptbxl.yaml \
  --phase heldout \
  --validation-lock results/security-validation/security_protocol.lock.json
```

The validation lock and event-log terminal manifest are auditable integrity records, not
cryptographic signatures or substitutes for preregistration and access control.

## E0-E7 Mapping

| ID | Runnable entry point | Required inputs |
| --- | --- | --- |
| E0 | `run_e0_clean_baseline` / `run_e0` | `CohortCase` sequence and victim provider |
| E1 | `run_e1_attack_matrix` / `run_e1` | E0-eligible cases, attacks, constraints, budgets, seeds |
| E1b | `run_e1b_staged_ablations` / `run_e1b` | primary successful selected attacks and named ablated victim providers |
| E2 | `run_e2_retrieval_mediation` / `run_e2` | successful primary-eligible, clinically preserved source pairs and a context/retrieval replay callback |
| E3 | `run_e3_latent_restoration` / `run_e3` | primary successful pairs and latent replay callback; top-k plus random-k controls |
| E4 | `run_e4_counterfactual_alignment` / `run_e4` | primary successful pairs and minimal counterfactual delta provider |
| E5 | `run_e5_matched_coverage_reattack` / `run_e5` | exactly V1/V2/V3, S-phys, shared typed `E5CalibrationCase` inputs, typed all-claims correctness evidence, fresh F3/victim factories, and an optional append-only logger |
| E6 | `run_e6_transfer_replay` / `run_e6` | successful perturbations, a second callable victim, and an independent all-claims clean-report evaluator |
| E7 | `run_e7_authenticated_history_stress` / `run_e7` | primary successful pairs, one common pre-defense replay callback, signed prior history, current timestamp, and optional tamperer |

E0 and E1 have a generic artifact-plus-NPZ CLI because their inputs share a stable file contract.
`security-evaluate` remains an alias for `security-run`.
E1b-E7 are explicit Python APIs: each requires an experiment-specific intervention callback,
calibration set, independently trained transfer victim, or authenticated prior record. This keeps
missing experimental inputs from being silently replaced with placeholders.

E5's primary comparison is intentionally narrow: it rejects missing or extra verifier arms,
requires an S-phys constraint, and constructs a fresh victim and re-optimized F3 attack for every
case, seed, and verifier. Primary matched-coverage output additionally requires
`calibration_cases`: each `E5CalibrationCase` binds one unique patient/record/target key to one
clean snapshot, the V1/V2/V3 inputs needed for that case, and optional independent all-claims
correctness evidence. The orchestrator calls every bound verifier on those same cases and
snapshots, derives its score, applies the matched-coverage threshold, then calls it again to record
the actual release decisions. A typed run is primary only when those recomputed release counts
match across V1/V2/V3 and attain the nearest cohort-quantized target coverage. Defense-specific
contexts cannot override the shared waveform, patient, record, sample rate, or timestamp.
Precomputed `calibration_scores`, `calibration_case_keys`, and
related legacy arrays remain available only as explicitly exploratory inputs, even when their
keys align. A custom reoptimization validator can veto a run, but cannot replace the observed
fresh factory invocation, complete-system queries, or verifier decisions. Primary re-attack
provenance requires the repository's exact audited `F3AdaptiveAttack` implementation with its
defense-specific objective applied. For custom factories, the orchestrator's causal probe first
requires an identical seeded factual replay, then intervenes only on the attack-facing verifier
decisions and checks whether the submitted query trace changes. This is diagnostic evidence only:
custom Python executes in the orchestrator process and can inspect runtime objects, frames, and
closures, so the probe is not an isolation attestation and cannot promote a custom factory to the
primary E5 endpoint. The built-in path snapshots its expected class and `run` method before any
attack factory call and requires exact identity immediately before and after execution. Arbitrary
process tampering before `run_e5` begins is outside this in-process boundary and requires external
process isolation and attestation. Merely calling a verifier, changing between fixed queries, or naming an object
F3 does not establish primary defense-conditioned reoptimization. Pass an
`AppendOnlySecurityLogger` through `logger=` to persist every complete-system submission with its
verifier arm, seed, budget, constraint, and calibrated threshold. Clean and attacked
selective-risk estimates are emitted only from
independent `ReportCorrectnessResult` assessments that cover all report claims; without that
evidence the risk fields remain unavailable rather than falling back to diagnosis or
target-claim heuristics.

```python
from care_ecg.security import (
    run_e1b,
    run_e2,
    run_e3,
    run_e4,
    run_e5,
    run_e6,
    run_e7,
)
```

Every runner returns an `ExperimentRun`; `run_e0` returns a `CleanBaselineRun` containing both the
run and its eligible cases. Call `.to_dict()` on an `ExperimentRun` for a JSON-ready result.

## Scope Boundaries

The attack CLI targets the waveform-callable internal CARE model. The repository's strongest final
benchmark stack includes cached probabilities keyed to record IDs; those values cannot be honestly
recomputed after changing a waveform, so they are excluded and this is recorded in every result.

The bundled `ClinicalAdmissibilityChecker` is a local signal-delta implementation. It does not
represent PTB-XL+ reference measurements and is deliberately ineligible for the primary endpoint.
A primary Python run must supply a target-aware checker that accepts `case`, `patient_id`,
`record_id`, `target`, or the bundled `ClinicalCaseContext`, then returns an
`AdmissibilityResult(primary_endpoint_eligible=True)` containing independent
`ClinicalPreservationEvidence` for that exact context. Boolean and mapping-only checker outputs are
kept exploratory. E0 likewise requires an independent evaluator returning
`ReportCorrectnessResult` with all report claims evaluated against a named reference standard.
The diagnosis-plus-target-absence fallback remains explicitly exploratory for E0 and is never
used to calculate E5 selective risk. E2 admits a source pair only when its attack succeeded, both
the clean case and attack are primary-endpoint eligible, and the selected perturbation is
clinically admissible. E6 uses the same typed `ReportCorrectnessResult` contract to establish clean
correctness on the transfer victim through `report_correctness_evaluator`. Its heuristic fallback
and legacy `clean_correctness` callback are reported only as exploratory, with the primary transfer
estimate marked `not_evaluable`.

The primary cohort requires explicit adult evidence in each `CohortCase.metadata` (`age`,
`age_years`, `patient_age`, or `is_adult`). Cases without usable adult metadata remain available
for exploratory analysis but are marked not evaluable for primary endpoints. The standard prepared
NPZ format has no age field, so CLI E0/E1 runs remain exploratory unless that metadata is supplied
upstream. Primary patient-level runs require `patient_ids`; `--allow-record-level` is explicitly
marked exploratory in the output. E1b/E3/E4/E6/E7 independently revalidate the selected successful
attempt's typed patient-record-target preservation evidence; legacy source pairs remain
exploratory. E2 replay callbacks must return a `VictimSnapshot` or a mapping containing both
`released` and `target_claim_present`; a bare boolean or `ReleaseDecision` cannot establish the
claim endpoint. V3 additionally requires a protected HMAC key, a genuine same-patient prior ECG,
and a current timestamp strictly after the authenticated history timestamp. E7 applies V2 and both
V3 arms to the same explicit pre-defense replay snapshot. E6 requires a second waveform-callable
model.
