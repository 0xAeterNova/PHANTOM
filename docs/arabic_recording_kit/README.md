# PHANTOM Arabic speech recording kit

This kit defines the controlled, scripted core dataset for PHANTOM's seven-class
spectrogram CNN/CRNN experiment. Use the prompt bank in
[`data/arabic_prompts_msa.csv`](data/arabic_prompts_msa.csv) and copy
[`data/recording_log_template.csv`](data/recording_log_template.csv) into the root of the
real audio dataset before filling it in.

## The recording decision / قرار التسجيل

**English:** Do not give each person random paragraphs. Every speaker records the **same
12 short, semantically neutral MSA sentences**, in **all seven labels**, with **two accepted
takes** of every combination. This controls the words so the model has a better chance of
learning acoustic expression rather than memorizing text or speaker identity. It produces
`12 prompts × 7 labels × 2 takes = 168 accepted clips per speaker`.

**العربية:** لا تعطوا كل متحدث فقرات عشوائية. يسجل كل متحدث **الجمل العربية الفصحى
المحايدة نفسها وعددها 12 جملة**، مع **كل التصنيفات السبعة**، وبواقع **محاولتين
مقبولتين** لكل تركيب. بهذه الطريقة تبقى الكلمات ثابتة ويكون هدف النموذج تعلم طريقة
الإلقاء الصوتية بدل حفظ النص أو هوية المتحدث. الناتج هو
`12 جملة × 7 تصنيفات × محاولتين = 168 مقطعًا مقبولًا لكل متحدث`.

Use these exact values in the trainer's `label` column:

| Trainer label | Arabic instruction |
|---|---|
| `neutral` | نبرة محايدة |
| `happy` | نبرة سعيدة |
| `sad` | نبرة حزينة |
| `angry` | نبرة غاضبة |
| `fearful` | نبرة خائفة |
| `surprised` | نبرة متفاجئة |
| `disgusted` | نبرة مشمئزة |

The label is the intended acted vocal expression, not a medical or psychological fact about
the speaker. Do not ask anyone to relive distress or trauma. A participant may pause or stop
at any time.

Perform every label at a natural, medium intensity:

| Label | أداء صوتي مطلوب |
|---|---|
| `neutral` | إلقاء يومي متوازن بلا عاطفة مقصودة |
| `happy` | نبرة دافئة وحيوية ومشرقة، من دون ضحك أو كلمات إضافية |
| `sad` | طاقة أخفض وسرعة أبطأ قليلًا، مع بقاء الكلام واضحًا |
| `angry` | نبرة حازمة ومتوترة وأقوى قليلًا، من دون صراخ أو تشبع |
| `fearful` | نبرة متوترة ومترددة قليلًا، من دون صراخ |
| `surprised` | دهشة عامة غير سعيدة أو خائفة، مع ارتفاع طبيعي في طبقة الصوت ومن دون صراخ |
| `disgusted` | نبرة نفور واضحة، من دون أصوات أو كلمات غير موجودة في الجملة |

## Audio format / صيغة الصوت

Set the recorder once and keep the same microphone, room, gain, and mouth-to-microphone
distance for the whole collection.

| Setting | Preferred master | Acceptable alternative |
|---|---:|---:|
| Container/codec | uncompressed integer-PCM WAV | uncompressed integer-PCM WAV |
| Channels | 1 (mono) | 1 (mono) |
| Sample rate | 48 kHz | 48 kHz |
| Bit depth | 24-bit | 16-bit |
| PCM bit rate | 1,152 kbps | 768 kbps |

The PCM bit rate is derived from `sample rate × bit depth × channels`; it is not a separate
quality setting. Do not use MP3, AAC, floating-point WAV, automatic noise removal, automatic
gain, compression, reverb, or background music. PHANTOM accepts integer-PCM WAV and
resamples recordings to 16 kHz internally for its 40-bin log-Mel features. Although it can
decode multiple channels, it averages them to mono, so recording mono avoids unnecessary
storage and channel-mixing differences.

If storage is genuinely limited, `16 kHz / 16-bit / mono integer-PCM WAV` is a compatible
space-saving capture format at `256 kbps`. Use one format consistently across the collection;
48 kHz/24-bit remains the preferred source master. Do not use 32-bit floating-point WAV.
At 48 kHz/24-bit mono, storage is about 8.64 MB per recorded minute; 168 clips averaging
four seconds require about 97 MB per speaker, before backups.

**العربية:** الصيغة المفضلة هي `WAV PCM` غير مضغوط، قناة واحدة `Mono`، بمعدل
`48 kHz` وعمق `24-bit`؛ ومعدل البيانات الناتج `1,152 kbps`. البديل المقبول هو
`48 kHz / 16-bit / Mono` ومعدل البيانات `768 kbps`. لا تستخدموا MP3 أو AAC ولا
تفعيل إزالة الضوضاء أو التحكم التلقائي في مستوى الصوت. وعند ضيق مساحة التخزين يمكن
استخدام `16 kHz / 16-bit / Mono PCM WAV` بمعدل `256 kbps`، مع تثبيت الصيغة نفسها
لكل التسجيلات.

## One recording session / جلسة تسجيل واحدة

1. Obtain informed consent and assign a pseudonymous ID such as `spk001`. Keep names and
   contact details out of filenames and this log.
2. Use a quiet, non-echoing room. Put the microphone about 15–20 cm from the mouth and keep
   its position fixed. Record only one person.
3. Make a short test. Normal speech must be clearly audible and the loudest angry or
   surprised delivery must not clip. Keep device processing disabled.
4. Read one prompt exactly as written; do not paraphrase it or add emotion words. Express
   only the requested vocal style. Keep one sentence per file, normally about 2–6 seconds
   and always below 8 seconds.
5. Leave roughly 0.25 seconds of quiet at the beginning and end. Redo a take if there is a
   misread word, interruption, clipping, handling noise, or another voice.
6. Record two accepted takes for every prompt-label pair. Use one emotion block at a time so
   the performance stays consistent, randomize prompts inside the block, and change the order
   of the seven emotion blocks between speakers. Take regular breaks.
7. Add one row to the copied recording log for each **accepted** file. Keep rejected takes
   outside the training manifest or replace them with a clean redo.

The 168 clips may be split across two shorter sessions. If so, include all seven labels in
each session, keep the equipment/settings fixed, and record `session_id`; never record one
label only on a particular day or device, because the model could learn the session or device
instead of the vocal expression.

**العربية:** احصلوا على الموافقة، واستخدموا رمزًا مستعارًا مثل `spk001`، وسجلوا في
غرفة هادئة وبالميكروفون نفسه وعلى مسافة ثابتة تقارب 15–20 سم. اقرأوا جملة واحدة كما
هي في كل ملف، وعبروا بالصوت عن التصنيف المطلوب دون إضافة كلمات عاطفية. اجعلوا المقطع
عادة بين ثانيتين وست ثوانٍ وأقل من ثماني ثوانٍ دائمًا. أعيدوا التسجيل عند الخطأ أو
الضوضاء أو تشبع الصوت، وسجلوا محاولتين سليمتين لكل جملة وتصنيف. غيّروا ترتيب الجمل
والتصنيفات بين المتحدثين وخذوا فترات راحة. يمكن تقسيم العمل على جلستين، لكن يجب أن
تحتوي كل جلسة على التصنيفات السبعة مع تثبيت الجهاز والإعدادات؛ لا تخصصوا يومًا أو جهازًا
لتصنيف واحد فقط.

## Files, IDs, and the trainer / الملفات والمعرّفات

Recommended dataset layout after copying the log template:

```text
audio_dataset/
  recording_manifest.csv
  wav/
    spk001/
      spk001_msa_01_angry_t01.wav
      spk001_msa_01_angry_t02.wav
    spk002/
      ...
```

For the first example above, use:

- `sample_id`: `spk001_msa_01_angry_t01`
- `path`: `wav/spk001/spk001_msa_01_angry_t01.wav`
- `label`: `angry`
- `speaker_id`: `spk001`
- `prompt_id`: `msa_01`
- `take`: `1`

Paths may be absolute, but relative paths are easier to move and are resolved relative to
the manifest CSV. Every `sample_id` and every audio path must be unique. The current trainer
requires all seven labels and at least four independent speakers, then creates a
speaker-independent train/validation/test split. Four speakers only satisfies this technical
minimum; it is not evidence that the resulting model will generalize.

When all accepted rows and files are ready, the copied log is compatible with:

```powershell
python scripts/train_audio.py C:\path\to\audio_dataset\recording_manifest.csv --architecture crnn
```

The trainer uses the required fields (`path`, `label`, `speaker_id`) and optional stable
`sample_id`; it ignores the additional audit fields. Do not put rejected rows in this CSV,
because an extra `quality_status` field would not make the trainer skip them.

## Dialect and later extensions / اللهجة والتوسعات اللاحقة

This first prompt bank is Modern Standard Arabic (`ar-MSA`). Speakers should not translate
or paraphrase it into their dialect. Use natural MSA pronunciation; do not force speakers to
pronounce full grammatical case endings. If a dialect such as Jordanian Arabic is part of
the target use case, build a second fixed prompt bank, use it consistently across speakers
and labels, record the dialect in the log, and evaluate it as a declared subset.

Free speech or spontaneous paragraphs can be useful later, but keep them in a separate
subset with their own prompt/session metadata. Do not mix person-specific random paragraphs
into this controlled core. Before calling intended labels ground truth, use a separate review
stage with independent Arabic-speaking listeners and exclude unclear/disputed clips from the
training manifest.

## Protocol basis

- [RAVDESS methodology](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391)
  used lexically matched statements, repeated performances, and uncompressed 48 kHz WAV.
- [CREMA-D's official project description](https://cheyneycomputerscience.github.io/CREMA-D/)
  describes a controlled bank of 12 sentences presented with multiple emotions.
- The [Library of Congress PCM format description](https://www.loc.gov/preservation/digital/formats/fdd/fdd000016.shtml)
  identifies uncompressed PCM as a preferred format and notes 48 kHz for audio origination;
  its sound-format guidance prefers 24-bit masters.
