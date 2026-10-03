# Model card: Project PHANTOM prototype

## Model details

- **Name:** Privacy-Preserving Human Affect and Natural-Tone Observation Module
- **Version:** development prototype; use the Git commit SHA for an exact version
- **Type:** modular orchestration, deterministic mock perception, optional untrained log-Mel CNN/CRNN architecture definitions, heuristic text/safety routing, confidence-aware late fusion, controlled response policy
- **Status:** experimental research software
- **License:** code license in `LICENSE`; all future data and weights require separate review

There is no single trained “PHANTOM model” in this repository. No pretrained weights or benchmark results are bundled. Audio and vision behavior in the default runnable path is synthetic/mock. The optional audio CNN and CNN-plus-bidirectional-RNN (CRNN) are trainable architecture and checkpoint-adapter code only; their presence is not evidence of a completed training run or useful affect inference. Confidence values in mock mode are user-specified test values.

## Intended use

- local, consent-based demonstration of multimodal software architecture;
- research on uncertainty, missing modalities, late fusion, corrections, and safe dialogue policy;
- test harness for future replaceable, documented perception adapters;
- hardware-independent robot-output prototyping through the mock adapter.

Human review and a clearly disclosed research context are required.

## Out-of-scope and prohibited use

Do not use Project PHANTOM to:

- diagnose, screen, triage, treat, or predict a medical or psychiatric condition;
- determine a person's true emotion, honesty, intent, dangerousness, competence, or identity;
- infer biological sex or gender identity;
- make employment, education, insurance, credit, housing, policing, immigration, legal, punishment, access-control, or medical decisions;
- covertly monitor people, recognize identities, track faces, or retain face embeddings;
- manipulate, shame, sexualize, form dependency, or exploit a user, especially a child;
- replace local emergency services or a qualified human professional.

## Inputs and outputs

Potential inputs are consent settings, preferences, text, and mock or experimental audio/image observations. Each modality returns a label, confidence, quality, optional distribution, availability, temporal consistency, and abstention reason. Fusion returns a distribution, confidence, uncertainty flag, per-modality contributions, and non-medical explanation.

Optional age-band and perceived-gender-presentation results are separate outputs. They are disabled by default and must never enter affect fusion. Self-report takes priority.

## Current components

| Component | Current evidence-backed claim |
|---|---|
| Session manager | In-process, bounded, expiring state with explicit deletion |
| Audio | Deterministic mock path, log-Mel preprocessing, and optional CNN/CRNN training/inference scaffolding; no bundled checkpoint or validated classifier |
| Vision | Deterministic mock path and safe ambiguity/quality interface; no validated classifier |
| Text | Lightweight, local heuristic observations; not a general semantic model |
| Safety router | Conservative phrase patterns and configurable messages; not a clinical classifier |
| Late fusion | Transparent quality/confidence/consistency-weighted baseline with abstention |
| Dialogue | Deterministic policy/templates with correction and permission controls |
| Optional visual attributes | Schema/consent interface only; no approved trained capability |
| Robot | Command protocol and non-physical mock adapter |

## Training data

None. The repository does not contain datasets or weights and does not claim a completed training run. The neural audio architectures start from random parameters unless a separately trained checkpoint is loaded, and random parameters must never be exposed as a working classifier. Candidate sources are documented in [docs/dataset_cards.md](docs/dataset_cards.md); unresolved facts are tracked in [RESEARCH_REQUIRED.md](RESEARCH_REQUIRED.md).

## Evaluation status

Software tests can verify deterministic behavior such as consent enforcement, missing-modality fusion, contradictions, deletion, and dialogue boundaries. They do **not** validate affect recognition.

Before any real-model claim, publish:

- group-aware split manifests (speaker-independent for speech and subject-independent for faces);
- a controlled CNN-versus-CRNN comparison using identical log-Mel preprocessing,
  speaker splits, seeds, calibration data, and a declared tuning budget;
- macro F1, unweighted average recall, per-class precision/recall, and confusion matrices;
- calibration error, reliability plots, abstention coverage/risk, and quality-versus-error;
- modality ablations and contradictory-signal behavior;
- latency and peak memory on declared CPU/GPU hardware;
- appropriate fairness slices with sample sizes and uncertainty;
- false reassurance, false escalation, correction, and prompt-injection results;
- negative and failed results.

No metric should be described as clinical accuracy.

## Ethical considerations

Affect annotation captures how annotators interpreted a performance or expression, not an objective internal state. Expression differs across cultures, languages, accents, skin tones, ages, disabilities, neurotypes, devices, environments, and individuals. Acted datasets may not represent spontaneous interaction. Label taxonomies can erase mixed or context-dependent experiences.

Visual age and presentation labels are particularly sensitive. Binary “gender” datasets do not validate gender identity or the project's broader presentation taxonomy. Any future research needs a proportionality assessment, participatory review, separate consent, abstention, and evidence that the feature benefits users rather than merely increasing surveillance.

## Known limitations

- Mock outputs cannot generalize to real people or sensors.
- Quality and confidence scores are not calibrated probabilities by default.
- Fusion assumes modality evidence can be combined; correlated errors can create false confidence.
- Simple temporal smoothing may suppress meaningful changes.
- Multiple faces cannot be assigned automatically.
- Text patterns can miss indirect safety language and misread quotations, negation, slang, or other languages.
- Dialogue templates cannot understand the full situation and may feel generic.
- In-memory deletion does not control copies held by clients or infrastructure.
- The prototype is not hardened for untrusted public traffic or physical robot control.

## Privacy

Default sessions are local, short-lived, and in memory. Cloud upload and optional attribute analysis/storage are disabled. Raw media and face embeddings are not retained by the core session manager. Operators must still audit framework temporary files, proxies, access logs, crash dumps, browser caches, and OS-level capture.

## Safety behavior

Ordinary affect output never triggers the crisis route. Concerning user-provided language is assessed before response generation. Crisis routing stops ordinary coaching, avoids detailed harmful content, and encourages immediate human/local support using operator-configured, verified resources. Because the router is not comprehensive, users must be told it cannot monitor or guarantee safety.

## Human oversight

An operator must be able to stop sensors and robot output, delete the session, correct an estimate, inspect contributions, and escalate to a human. User correction overrides the system's affect wording. High-impact decisions are categorically prohibited rather than delegated to human review.

## Release checklist

- [ ] Exact commit and configuration recorded
- [ ] All real datasets and weights have approved cards and hashes
- [ ] No personal data, secrets, data, or weights committed
- [ ] Safety, privacy, API, and mock E2E suites pass
- [ ] Real-model evaluation reproduced independently
- [ ] Accessibility and consent comprehension reviewed
- [ ] Local resources verified for deployment country
- [ ] Threat model and incident response approved
- [ ] Claims audit finds no diagnostic or definitive-emotion wording

## Contact

Use the private process in [SECURITY.md](SECURITY.md) for vulnerabilities. Use repository issues only for non-sensitive reproducible software defects.
