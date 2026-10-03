# Notebooks

Notebooks are for exploration, not authoritative training or production inference. None is bundled because the repository has no approved dataset.

Rules:

- use synthetic/mock data until a data card is approved;
- set seeds and import reusable code from `src/phantom`;
- clear outputs, widget state, paths, tokens, images, audio, transcripts, and metadata before commit;
- never embed a dataset, weight, face, voice, participant identifier, or support conversation;
- record commit, environment, configuration, dataset/split manifest hashes, and exact commands;
- move durable logic and tests into typed modules;
- report failed and negative results.

Use the scripts for reproducible runs. A notebook screenshot or completed cell is not benchmark evidence.
