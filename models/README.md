# Model artifacts

No pretrained or trained weight is included or approved. The runnable mock demonstration does not need one. Keep local artifacts out of Git.

Before using a weight, record:

- official immutable source, model card, code license, weight license, and training-data provenance;
- architecture, task/construct, labels, preprocessing, supported language/input, and known limitations;
- file SHA-256/signature, safe serialization format, dependency versions, and whether remote code is required;
- evaluation/calibration evidence on the declared target population;
- owner, access, retention, deletion, rollback, and incident process.

Never load an untrusted pickle or enable remote model code. Prefer a safe tensor format, read-only artifact mount, allowlisted hash, offline/local-files-only loading, and a sandboxed worker.

## Local Piper TTS voice

The real-time profile expects the ignored `models/tts/ar_JO-kareem-low.onnx`
and matching `.onnx.json`. Provision them only through:

```powershell
python scripts\setup_piper_tts.py
```

The script pins an immutable upstream revision, exact byte sizes, MD5 values,
and SHA-256 values. The runtime rechecks SHA-256 before loading. These local
files are not committed or included in the MIT-licensed source distribution;
read [`docs/third_party_tts.md`](../docs/third_party_tts.md) for the separate
runtime and voice-data licensing boundary.

## Spectrogram CNN/CRNN checkpoints

The optional audio architectures consume log-Mel sequences and do not ship
with weights. A compatible local checkpoint must contain tensor weights rather
than a serialized executable model object. Its reviewed sidecar metadata must
record at least:

- checkpoint format/version and SHA-256;
- architecture (`cnn` or `crnn`) and all shape-defining hyperparameters;
- feature definition, including sample rate, Mel-bin count, FFT, frame, hop,
  frequency range, trained maximum frame count, and normalization;
- canonical label order and the source-dataset label mapping;
- dataset/split manifest hashes, seed, training commit, dependency versions,
  and whether any initialization was pretrained;
- calibration/abstention configuration and held-out evaluation status;
- requested and resolved training device.

Load tensor weights onto the explicitly resolved inference device. A checkpoint
created on CUDA must remain loadable for reviewed CPU inference without
executing arbitrary serialized code. Reject architecture, feature-shape, label,
or format mismatches instead of guessing. A random initialization or a
checkpoint whose provenance is missing must never be exposed as a working
classifier.

Suggested local layout:

```text
models/
  registry.yaml          # untracked approved artifact metadata
  audio/<name>/<sha>/
  vision/<name>/<sha>/
  fusion/<name>/<sha>/
```

An exported checkpoint is experimental until its run manifest and [MODEL_CARD.md](../MODEL_CARD.md) evidence are complete. Optional age/presentation weights require a separate governance decision and must never feed affect fusion.
