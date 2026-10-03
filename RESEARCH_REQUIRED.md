# Research required

This is the blocking research register. A name in this file is a **candidate, not an approved dependency or endorsement**. Do not download data or weights, begin human-subject research, or advertise a model capability until the applicable item is resolved and the decision is recorded.

## Priority 0: before any real-data run

- Verify the current official license, access terms, redistribution rules, commercial-use status, data-subject consent/provenance, and automated-download permission for every dataset.
- Review media rights separately from repository code licenses. A code repository license does not necessarily license audiovisual source material.
- Decide whether local ethics/IRB review is required for recorded voices, faces, minors, sensitive attributes, or interactive safety studies.
- Define deletion/withdrawal handling and dataset version/hash records.
- Approve a leakage-resistant split strategy before looking at test labels.
- Document the exact affect construct: portrayed expression, annotator perception, self-report, or another target. Do not relabel it “true emotion.”

## Dataset review status

Two candidates have enough official-source information for preliminary cards: RAVDESS and CREMA-D. They are **not automatically approved**. MELD, IEMOCAP, AffectNet, UTKFace, and FairFace retain unresolved rights, suitability, or governance questions. See [docs/dataset_cards.md](docs/dataset_cards.md).

| Research category | Current decision |
|---|---|
| Speech expression | Compare RAVDESS and CREMA-D with spontaneous, multilingual alternatives; acted English speech alone is insufficient |
| Facial expression | No source approved; verify access and consent provenance, especially for web-collected faces |
| Multimodal conversation | No source approved; resolve audiovisual copyright and dialogue-context leakage |
| Broad age band | No source approved; establish necessity and proportionality before model selection |
| Perceived gender presentation | No source approved; existing binary labels do not match the construct or safeguards |
| Conversational safety | Use synthetic policy tests now; any human-language corpus needs privacy, license, and harm-taxonomy review |

## Pretrained model and weight research

For each candidate adapter, verify:

- official model card and immutable weight source;
- code license and weight license (they may differ);
- training-data provenance and restrictions;
- supported language/sample rate/image preprocessing/label map;
- known subgroup limitations and calibration evidence;
- checksum/signature and supply-chain ownership;
- CPU memory, latency, and offline behavior;
- whether the model sends telemetry or downloads code dynamically;
- whether its labels can be mapped without pretending equivalence.

No pretrained speech-emotion, facial-expression, age-band, or presentation estimator is approved or bundled. Generic representation models do not become validated affect models merely by adding a classifier head.

The new log-Mel CNN and CNN-plus-bidirectional-RNN (CRNN) implementations are
architecture candidates, not an approved model choice. Compare them with the
same speaker-independent splits, preprocessing, seeds, tuning budget,
calibration, abstention policy, and declared CPU latency/memory measurements
before selecting either architecture or approving any checkpoint.

## Crisis and support-resource research

- Replace the global placeholder with country/region resources verified by a qualified local owner.
- Define a review cadence, source-of-truth URL, last-verified date, supported languages, hours, phone/text accessibility, and outage fallback.
- Evaluate indirect language, slang, multilingual inputs, negation, quotations, third-person concern, coercion, and adversarial prompt injection.
- Measure false reassurance and false escalation with trained human reviewers. Never optimize only keyword recall.
- Establish an incident procedure for a public deployment; this prototype cannot monitor users or dispatch help.

## Optional visual attributes

Before even an experimental implementation:

1. document a user benefit that cannot be achieved through self-report or preferences;
2. run a data-protection and human-rights impact assessment;
3. validate broad age-band boundaries and the presentation label taxonomy with affected communities;
4. exclude identity recognition and persistent embeddings by design;
5. create separate consent, evaluation, storage, and deletion paths;
6. prove through tests that outputs never affect emotion, dignity, access, punishment, or high-impact decisions;
7. provide `unknown` and `analysis-disabled` and calibrate abstention;
8. prioritize self-reported age and form of address.

Until these steps pass, keep the feature disabled and describe it as an interface placeholder.

## Evaluation research

- Select calibration and selective-classification methods before tuning thresholds.
- Predefine minimum sample sizes and uncertainty intervals for fairness slices; suppress misleading tiny groups.
- Test cross-corpus and out-of-domain shifts across languages, accents, devices, noise, lighting, pose, occlusion, disability, and expression style.
- Evaluate correlated modality errors and adversarial contradictions.
- Conduct consent-comprehension, correction, accessibility, and over-trust studies.
- Define release-stopping thresholds for privacy leakage, unsafe dialogue, crisis false reassurance, and demographic disparity.

## Open engineering questions

- Which media decoders can be sandboxed reliably across supported platforms?
- How will worker timeouts and memory ceilings be enforced in deployment?
- How will signed weight manifests and rollback work?
- Is server-side processing justified, or should all perception remain on-device?
- What telemetry is strictly necessary, and can it be aggregate-only?
- How will browser camera/microphone state be reconciled with server consent state?

## Completion evidence

Resolve an item by linking an official source, recording reviewer/date/version, stating the decision and rationale, and attaching reproducible evidence. “Widely used,” a third-party blog, or a model hub mirror is not sufficient.
