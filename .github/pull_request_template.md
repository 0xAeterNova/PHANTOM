## Summary

Describe the change, why it is needed, and the user-visible outcome.

## Validation

- [ ] `python -m ruff format --check .`
- [ ] `python -m ruff check .`
- [ ] `python -m mypy src/phantom`
- [ ] `python -m pytest -m "not slow and not optional_ml"`
- [ ] Relevant safety, integration, and optional-ML checks were run or the omission is explained.

## Responsible-AI and privacy review

- [ ] No credentials, model weights, datasets, real recordings, face images, or personal transcripts are committed.
- [ ] Sensor activation and analysis remain subject to explicit, granular consent.
- [ ] Raw biometric data is not logged or retained by default.
- [ ] Outputs express uncertainty and do not claim diagnosis, identity, biological sex, or true internal state.
- [ ] Age-band and perceived-gender-presentation analysis remain disabled by default and do not influence affect fusion.
- [ ] Crisis-routing and user-correction behavior remain intact, or changes are covered by safety tests.
- [ ] New dependencies, model sources, and datasets have documented licenses and integrity checks.

## Risk and rollback

State the likely failure modes, compatibility impact, and a safe rollback approach.
