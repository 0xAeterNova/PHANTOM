# Third-party text-to-speech notice

Project PHANTOM routes replies containing an Arabic letter, including mixed
Arabic/English replies, to a local Piper neural voice. English-only replies use
Windows SAPI in native Windows installations or eSpeak NG in Linux containers.
No reply text is sent to a remote TTS service.
TTS network access is limited to the model-provisioning command, run explicitly
for native installations or automatically at real Docker startup.

## Piper runtime

- Package: `piper-tts==1.6.0`
- Upstream: <https://github.com/OHF-Voice/piper1-gpl>
- Installed package license metadata: `GPL-3.0-or-later`

Piper is an optional real-time runtime dependency with terms separate from
PHANTOM's MIT-licensed source. Anyone distributing a combined installation,
binary, container, or appliance must review and satisfy Piper's current license
and notices; this document is an engineering inventory, not legal advice.

## Linux English speech runtime

- Debian package: `espeak-ng` (installed from Debian Bookworm repositories)
- Upstream: <https://github.com/espeak-ng/espeak-ng>
- License: `GPL-3.0-or-later`, with the package's complete copyright notices at
  `/usr/share/doc/espeak-ng/copyright` inside the real image
- English output uses `espeak-ng --stdout --stdin`; no host sound device is opened.

The real image contains GPL-licensed runtimes even though model weights are
downloaded separately. Before publishing or sending that binary image, review
the applicable corresponding-source, license, and notice obligations for Piper,
eSpeak NG, and every other included dependency. A link in this notice alone is
not a claim of complete license compliance. The included GitHub publishing
workflow intentionally publishes only the lightweight demo image; users can
build the real image locally from source after reviewing the upstream terms.

## Jordanian Arabic voice

- Repository: <https://huggingface.co/rhasspy/piper-voices>
- Immutable revision: `6249c8a9178e606f0de19227d5426e5dfaf9fc9e`
- Voice card: <https://huggingface.co/rhasspy/piper-voices/blob/6249c8a9178e606f0de19227d5426e5dfaf9fc9e/ar/ar_JO/kareem/low/MODEL_CARD>
- Declared voice facts: `ar_JO`, one speaker, low quality, 16,000 Hz, fine-tuned
  from the U.S. English Lessac low-quality voice

| Artifact | Bytes | MD5 | SHA-256 |
|---|---:|---|---|
| `ar_JO-kareem-low.onnx` | 63,201,294 | `d335cd06fe4045a7ee9d8fb0712afaa9` | `2887e9d68b125965c747e1371fa21e1cef19555ea98d0795a0d5d71188b13890` |
| `ar_JO-kareem-low.onnx.json` | 5,022 | `465724f7d2d5f2ff061b53acb8e7f7cc` | `da328e52896826135508f797c1c77b45b35117e967c71befc377d654f100f328` |

The Hugging Face repository metadata displays an MIT tag. The individual Kareem
voice card identifies its dataset source as
<https://github.com/AliMokhammad/arabicttstrain/> and says to consult that source
for the dataset license; it does not state a dataset license in the voice card
itself. Do not infer that repository metadata resolves every right in the
underlying recordings, speaker performance, training data, or generated model.
Review the current upstream materials before redistribution or public deployment.

## Provisioning and integrity

Run `python scripts/setup_piper_tts.py`. The script downloads from the immutable
revision above, rejects non-HTTPS redirects, enforces exact byte limits, checks
both requested MD5 values and SHA-256, and replaces the local target only after
verification. PHANTOM then checks SHA-256 again before first model load. MD5 is
retained for compatibility with the supplied artifact record; SHA-256 is the
stronger integrity control.
