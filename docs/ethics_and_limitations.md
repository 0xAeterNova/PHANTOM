# Ethics and limitations

## Positioning

Project PHANTOM is an experimental affect-aware supportive interaction prototype. It is not a psychiatrist, psychologist, therapist, doctor, diagnostic instrument, medical device, lie detector, identity system, or emergency service.

Outputs should be framed as possibilities:

- “The available signals may indicate…”
- “The model is uncertain.”
- “You may be feeling…”
- “Would you like to tell me more?”
- “This is not a medical assessment.”

Never say a face or voice proves a feeling, that a person definitely has a condition, or that the assistant is acting as their clinician.

## The construct problem

Emotion-related labels can mean at least four different things:

1. an actor was instructed to portray an expression;
2. observers perceived an expression;
3. a participant self-reported an experience;
4. a model mapped patterns to a dataset label.

These are not interchangeable. None reveals an objective private mental state. A model can reproduce annotator conventions while being wrong about the person, context, or meaning. Reports must name the exact target and avoid “emotion detection” as a truth claim.

## Sources of error and inequity

Performance may differ across:

- languages, dialects, accents, speech differences, and code-switching;
- cultures, display rules, and individual expression styles;
- skin tones, lighting, pose, occlusion, camera quality, and assistive devices;
- ages, disabilities, neurotypes, fatigue, medication effects, and temporary illness;
- microphones, compression, noise, reverberation, speaking style, and silence;
- spontaneous versus acted data and laboratory versus everyday settings.

Dataset balance does not guarantee fairness. Labels may encode stereotypes; subgroup categories may be imposed, incomplete, or too small for reliable metrics. Aggregate accuracy can hide severe harms.

## Multimodal limitations

More modalities do not automatically improve validity. They may share the same bias, react to the same recording artifact, or contradict for legitimate reasons. Late fusion can turn correlated errors into false confidence. PHANTOM therefore reports contributions and abstains on weak or conflicting evidence, but thresholds still require calibration.

Missing modalities are expected. Absence must not be interpreted as a negative signal. Quality scores are engineering estimates, not proof that an input is meaningful.

## Dialogue risks

Supportive phrasing can still:

- validate an incorrect interpretation;
- feel intrusive or patronizing;
- encourage over-trust or emotional dependency;
- miss the user's actual request;
- offer an exercise at the wrong time;
- provide false reassurance or excessive crisis escalation.

The policy asks rather than declares, honors correction, requests permission before grounding, and encourages human support when appropriate. It must never claim exclusivity, guilt a user into returning, simulate romance/sexuality with a minor, or use age/presentation stereotypes.

Optional generative models are downstream of the controlled policy. Transcripts and retrieved text are untrusted input and cannot override boundaries. Generated text must be checked and fall back safely.

## Crisis-language routing

The bundled router is a conservative pattern-based backstop. It is not a suicide-risk assessment and cannot understand all euphemisms, languages, quotations, temporal context, or coercion. It can miss urgent language or route benign discussion.

When immediate-safety language is routed:

- stop ordinary coaching;
- use concise, calm, non-graphic wording;
- encourage local emergency services or a trusted nearby person;
- show only deployment-verified, country-appropriate resources;
- do not promise monitoring, confidentiality beyond reality, dispatch, or rescue.

Affect labels alone never trigger this route. Operators must not infer crisis from a face or tone.

## Age and perceived presentation

Do not infer biological sex or gender identity. The only permitted future presentation outputs are masculine-presenting, feminine-presenting, androgynous or ambiguous, unknown, and analysis-disabled. Permitted broad age bands are child, teenager, young adult, adult, older adult, unknown, and analysis-disabled.

These features:

- are disabled by default and currently lack an approved trained capability;
- require separate explicit consent and storage permission;
- include confidence and abstention;
- never influence affect conclusions, dignity, support level, access, or high-impact decisions;
- yield to self-reported age and preferred address;
- cannot establish whether someone is legally a minor.

For a potentially underage user, use stronger privacy and safety defaults based primarily on self-report/context—not a visual estimate. Do not generate romantic, sexual, manipulative, or dependency-forming dialogue.

## Prohibited contexts

Do not use the project for medicine, therapy replacement, hiring, worker/student monitoring, grading, insurance, credit, housing, law enforcement, border control, punishment, access control, deception detection, interrogation, advertising manipulation, covert surveillance, or any decision about rights or opportunities.

Consent does not make a disproportionate or harmful use ethical. In many power relationships, consent is not freely given.

## Privacy limitations

Local/in-memory defaults reduce exposure but do not eliminate it. Raw uploads transit the client, network stack, multipart parser, memory, and decoder. Browsers and operating systems control sensor indicators. Clients, proxies, crash dumps, swap, backups, screenshots, and observability tools may retain copies. Application deletion cannot erase copies it does not control.

No persistent face embedding is needed or permitted. Cloud processing is unavailable in the default build.

## Evidence and claims

Software tests support claims about code paths only. Mock outputs are synthetic. No bundled benchmark supports real-world affect, age, presentation, fairness, or clinical performance. Do not use “accurate,” “human-level,” “unbiased,” “safe,” or “production-ready” without a defined population, protocol, uncertainty, reproduced evidence, and scope.

## Responsible research questions

Before adding a feature, ask:

1. What user benefit requires this inference?
2. Can self-report or a direct preference achieve it with less intrusion?
3. What does the label actually represent?
4. Who is missing or harmed by the dataset?
5. Can users decline, correct, stop, and delete without penalty?
6. What happens when the system is wrong, uncertain, or attacked?
7. What evidence would stop release?

If the benefit does not outweigh the risk, do not build the feature.
