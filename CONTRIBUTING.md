# Contributing

Project PHANTOM accepts changes that strengthen a consent-based experimental prototype. Contributions must preserve non-medical positioning, local-first privacy defaults, explicit uncertainty, and separation of optional visual attributes from affect.

## Before opening a change

1. Read the [model card](MODEL_CARD.md), [ethics guidance](docs/ethics_and_limitations.md), [data governance](docs/data_governance.md), and [security policy](SECURITY.md).
2. Search existing issues without posting sensitive data.
3. For a new dataset, weight, sensor, cloud path, demographic feature, LLM provider, or robot action, open a design proposal first. Include purpose, alternatives, data flow, licenses, consent, abuse cases, rollback, and tests.
4. Never add real human media or transcripts. Use synthetic arrays, generated geometric images, and invented text fixtures.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[api,demo,dev]"
pre-commit install
```

## Development rules

- Keep modules replaceable, typed, configuration-driven, and safe without optional ML packages.
- Enforce consent at the service boundary, not only in the UI.
- Never log raw media, full transcript content, secrets, embeddings, or optional attributes.
- Return an abstention instead of forcing a label.
- Do not feed age-band or presentation output into affect, dialogue dignity, eligibility, or safety routing.
- Treat user correction as authoritative for subsequent wording.
- Route user-provided safety language before ordinary supportive generation.
- Keep crisis resources configurable and locally verified; do not invent contact details.
- Mark mock, heuristic, experimental, and validated components exactly.

## Tests

Run:

```powershell
pytest
ruff check .
ruff format --check .
mypy src app
```

Add meaningful tests for changed behavior. Safety-sensitive changes should include positive, negative, contextual, multilingual where supported, correction, and adversarial cases. A test must not assert a definitive internal emotion.

## Data/model contribution checklist

- [ ] Official source, version, license, access, commercial and redistribution terms documented
- [ ] Data-subject provenance and population limitations documented
- [ ] Automated downloading explicitly permitted
- [ ] Checksum and expected structure available
- [ ] Speaker/subject-independent split plan
- [ ] Training/evaluation preprocessing and label mapping documented
- [ ] Weight license and training-data provenance verified separately
- [ ] Privacy, bias, misuse, and rollback review approved

Do not commit the artifact itself unless the repository owner and upstream terms explicitly allow it. Default to external, user-confirmed preparation.

## Pull requests

Keep changes focused. Include:

- the user/research problem and why the change is proportionate;
- files and data flows changed;
- exact verification commands and actual outcomes;
- privacy, safety, accessibility, and compatibility impacts;
- mock-versus-real status;
- screenshots only with synthetic data;
- limitations and follow-up work.

Reviewers may block a technically correct change that weakens consent, privacy, uncertainty, safety, or project positioning.
