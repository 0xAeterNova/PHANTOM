# Project PHANTOM research report template

> Replace every bracketed field. Use “not measured” instead of inventing a result. Attach immutable manifests; do not attach restricted data or sensitive examples.

## 1. Record

- Title:
- Authors and roles:
- Date:
- Repository commit/tag:
- Configuration and environment lock:
- Study/ethics approval identifier, if applicable:
- Data/model/legal reviewers:
- Intended audience and use:
- Prohibited interpretations:

## 2. Executive summary

State the research question, exact construct, data, method, primary evidence, negative results, limitations, and decision. Explicitly say this is not a medical assessment.

## 3. System and claim boundary

- Components evaluated:
- Mock, heuristic, experimental, or validated status of each:
- Inputs/outputs:
- Human oversight:
- What the system cannot conclude:
- Changes since the model card:

## 4. Data cards and governance

| Dataset | Official source/version | Manifest hash | License/access decision | Population/construct | Split unit |
|---|---|---|---|---|---|
| [name] | [link/version] | [SHA-256] | [decision + reviewer/date] | [limits] | [speaker/subject/etc.] |

Describe consent provenance, media rights, inclusion/exclusion, missingness, withdrawals, retention, access, deletion, and why the data is proportionate.

## 5. Methods

- Preprocessing and quality gates:
- Label mapping and why labels are comparable:
- Architecture and initialization:
- Calibration and abstention:
- Fusion weights/learner:
- Temporal handling:
- Dialogue/safety policy version:
- Hyperparameter budget and seeds:
- Hardware and software:
- Exact commands:

## 6. Split and leakage audit

Describe group-aware splitting, duplicates/near-duplicates, shared scenes/sessions, preprocessing leakage, test access controls, and all deviations.

## 7. Results

### 7.1 Per modality

Report macro F1, UAR, per-class precision/recall/F1, confusion matrix, calibration, coverage-risk, quality-versus-error, latency, memory, and confidence intervals.

### 7.2 Fusion

Report single modalities, ablations, missing modalities, contradictions, correlated errors, temporal cases, and out-of-domain results.

### 7.3 Safety and dialogue

Report false reassurance, false escalation, correction, advice permission, diagnostic/medication/dependency boundaries, prompt injection, underage safeguards, and crisis-route results.

### 7.4 Fairness and validity

For each justified slice, report definition/provenance, count, missingness, interval, performance/calibration, practical harm, and limits. Explain omitted slices.

### 7.5 Optional attributes

Keep separate from affect. State necessity review, consent, broad-band/presentation construct, abstention, calibration, fairness, and tests proving no downstream influence. If not evaluated, say disabled.

## 8. Human-factors and accessibility

Report consent comprehension, sensor awareness, stop/delete/correction discoverability, over-trust, emotional impact, keyboard/screen-reader/zoom/contrast findings, and participant withdrawal.

## 9. Security and privacy

- Threat-model version and changes:
- Upload/decoder and resource-limit tests:
- Log and temporary-file inspection:
- Session isolation/expiry/deletion evidence:
- Dependency/SBOM/weight provenance:
- Findings, severity, owner, due date:

## 10. Failures and negative results

List failed runs, unusable classes/slices, unexpected harms, incidents, unresolved uncertainty, and hypotheses rejected. Do not hide inconvenient evidence.

## 11. Limitations

Cover construct validity, annotation, sampling, languages/accents/cultures, skin tones/lighting, ages/disabilities/neurotypes, devices/environments, acted versus spontaneous behavior, calibration, correlated modalities, privacy boundary, and crisis-router limits.

## 12. Release decision

- Decision: [mock only / continue research / limited experimental release / stop]
- Evidence supporting decision:
- Release-stopping criteria assessed:
- Required mitigations and owners:
- Rollback trigger and procedure:
- Claim wording approved by:

## 13. Reproduction manifest

List exact artifact paths/hashes, commands, expected runtime, hardware, seeds, and actual pass/fail outputs. Cite each dataset/model under upstream requirements.
