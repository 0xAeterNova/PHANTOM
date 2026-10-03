# Project PHANTOM: eight-week implementation plan

## Status and assumptions

This plan treats Project PHANTOM as an experimental, local-first research prototype. The initial milestone is a deterministic mock system, not a trained affect-recognition product. No medical, psychiatric, identity, age, or gender claim is in scope. Raw media and cloud upload remain off by default. Every real-data activity requires documented consent provenance, legal review, and an approved data card.

Current repository baseline:

- typed schemas, consent/session lifecycle, confidence-aware fusion, controlled dialogue, privacy/safety modules, API/UI/CLI surfaces, and mock robot behavior;
- deterministic mock audio/vision controls and lightweight text behavior;
- no production model weights, real training result, benchmark, clinical validation, or approved deployment.

## Definition of done

At the end of eight weeks, a research milestone is complete only if:

1. mock behavior remains reproducible and all safety/privacy tests pass;
2. any experimental real model has a traceable dataset card, split manifest, configuration, checkpoint provenance, evaluation report, and explicit limitations;
3. uncertainty, calibration, missing modalities, contradictions, and subgroup limitations are reported;
4. an independent reviewer signs off on threat-model and dialogue red-team findings;
5. documentation never represents model output as internal truth or medical assessment.

## Work plan

| Week | Goal | Deliverables | Exit evidence |
|---|---|---|---|
| 1 | Safe foundation | Schemas; consent states; in-memory session deletion/expiry; mock orchestrator; redacted logging | Unit tests for defaults, consent denial, expiry, deletion, and log redaction |
| 2 | Independent modality interfaces | Audio, vision, and text contracts; quality/abstention states; no-face/multiple-face handling; synthetic fixtures | CPU mock E2E; invalid-input tests; modality outputs remain separate |
| 3 | Fusion and dialogue | Weighted late fusion; temporal smoothing; contradictions; controlled response policy; user correction | Missing/low-quality/conflict tests; prompt examples; boundary tests |
| 4 | API, UI, and robot adapter | Validated endpoints; bounded uploads; consent UI; sensor indicators; mock robot | API malicious-input tests; keyboard walkthrough; stop/delete demonstration |
| 5 | Data and model governance | Dataset cards; license/access decision log; split policy; model registry format | Legal/ethics approval or explicit rejection for each candidate source |
| 6 | Experimental baselines | Speaker/subject-independent training adapters; checkpoint metadata; calibration and abstention | Reproducible run record; no performance claim without saved evidence |
| 7 | Evaluation and red teaming | Modality ablations; conflicts; fairness slices where legitimate; safety and privacy attack tests | Signed evaluation report; unresolved findings have owners and severity |
| 8 | Hardening and release candidate | Dependency scan; container; documentation audit; release manifest; rollback plan | Clean CI, reproducible mock demo, security review, non-medical release notes |

## Milestones

### M1 — Mock demonstrator

A user can consent, create a session, supply any supported mock modality combination, see quality/confidence/contributions, correct the estimate, obtain a bounded supportive response, and delete the session. This milestone says nothing about real-world affect accuracy.

### M2 — Research harness

Training/evaluation interfaces create immutable run metadata and refuse undocumented data. Target metrics include macro F1, unweighted average recall, per-class precision/recall, calibration error, abstention coverage/risk, quality-versus-error, modality ablations, contradiction behavior, latency, memory, and applicable fairness slices.

### M3 — Reviewed experimental adapter

At least one modality adapter may be labeled experimental only after a reproducible evaluation. It must abstain under poor quality, report target-population limits, and remain replaceable. Optional visual attributes are a separate workstream and cannot influence fusion or dialogue dignity.

### M4 — Research release

The release includes exact commands, artifact hashes, dependency lock/provenance, data/model cards, safety results, known failures, and a rollback path. “Research release” does not mean production-ready or medically validated.

## Decision gates

- **Data gate:** official source, license, access terms, population, consent provenance, redistribution, commercial restrictions, and automated-download permission verified.
- **Model gate:** architecture and weight license verified separately; training data provenance known; checksum recorded.
- **Evaluation gate:** leakage-resistant splits and calibration completed; uncertainty and negative results published.
- **Safety gate:** crisis, prompt-injection, correction, underage, dependency-forming, stereotyping, and false-reassurance tests reviewed.
- **Deployment gate:** use case, operator, retention, jurisdiction, authentication, incident response, accessibility, and human escalation approved.

Failure at a gate pauses that workstream; it does not justify lowering safeguards.

## Responsibility matrix

| Area | Accountable role | Required reviewer |
|---|---|---|
| Dataset/license decision | Data steward | Legal/ethics reviewer |
| Perception model | ML lead | Independent evaluation lead |
| Consent/retention | Privacy lead | Security reviewer |
| Dialogue/safety | Responsible-AI lead | Domain safety reviewer |
| Robot hardware | Robotics lead | Physical-safety operator |
| Release | Maintainer | Security + responsible-AI reviewers |

## Risk register

| Risk | Early signal | Mitigation |
|---|---|---|
| Users over-trust affect labels | Definitive wording or high confidence | Forced uncertainty language, abstention, correction control, usability test |
| Dataset leakage | Same speaker/subject across splits | Group-aware split manifests and duplicate checks |
| Biometric retention | Media or embeddings appear in logs/cache | Data-flow tests, deny storage by default, deletion and expiry |
| Demographic stereotyping | Optional attributes affect emotion/dialogue | Schema separation, code tests, feature disabled |
| Crisis false reassurance | Concerning text receives ordinary coaching | Safety route before dialogue; adversarial phrase tests; human escalation |
| Crisis over-routing | Quoted/research text triggers emergency copy | Context tests, calm clarification, clear non-clinical limitations |
| Malicious upload | Decoder crash or resource exhaustion | Byte limits, signature checks, isolated decoding, timeouts |
| Model supply-chain compromise | Untracked or changed weight | Allowlist, checksum, provenance, read-only artifact store |
| Physical robot harm | Unbounded motion/volume | Safe command allowlist, emergency stop, operator supervision |

## After week eight

Do not move directly to public deployment. First run participatory and independent evaluation, resolve high-severity findings, verify jurisdiction-specific obligations, and decide whether the proposed use case is proportionate. If evidence is inadequate, keep the system in mock/research mode.
