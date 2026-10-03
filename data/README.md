# Data directory

No dataset or human recording is included. Everything below `data/` is ignored except documentation/placeholders. Never force-add raw media, transcripts, faces, personal information, restricted annotations, or secrets.

## Zones

- `raw/`: untouched, user-obtained source under its official terms; read-only after intake.
- `interim/`: manifests, validated structure, and reversible preprocessing.
- `processed/`: split-specific derived features suitable for training.

These names are organizational, not proof of anonymization. Voice, face, transcript, derived embeddings, and sensitive inferences can remain personal/biometric data.

## Intake gate

Before copying anything:

1. complete/approve the source card in [docs/dataset_cards.md](../docs/dataset_cards.md);
2. verify official license, access, media rights, commercial/redistribution terms, consent provenance, and automated-download permission;
3. obtain ethics/privacy approval where needed;
4. predefine speaker/subject-independent splits and retention/deletion;
5. use an encrypted, access-controlled volume with sufficient capacity.

The script does not download:

```powershell
python scripts/prepare_data.py "Official dataset name" C:\approved\source --accept-license
```

`--accept-license` is an operator confirmation, not legal validation. The script hashes an existing directory into an interim manifest. Compare expected structure and official checksums manually; stop on any mismatch.

## Storage

Do not use a personal cloud drive merely because local disk is low. First verify organizational approval, provider terms/data-processing agreement, jurisdiction, encryption, account access, sync clients, sharing, backups, retention, and deletion. Prefer an approved encrypted research volume. Never upload project participant data through browser automation.

## Splits and leakage

Group speech by speaker and faces by subject. Keep sessions/scenes and near-duplicates together. Freeze the test manifest before tuning. Record all excluded/corrupt files rather than silently dropping them.

## Synthetic tests

Tests should generate arrays/WAV bytes, geometric images, and invented non-graphic text at runtime. Do not add photographs or voices of real people, even “sample” files.

## Removal

Deleting local data may not delete synced copies, backups, derived datasets, caches, manifests containing paths, or trained weights. Follow [docs/data_governance.md](../docs/data_governance.md).
