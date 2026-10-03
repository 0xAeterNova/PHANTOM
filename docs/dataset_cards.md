# Candidate dataset cards

## How to read this catalog

These are preliminary research cards, checked against linked project/publisher sources on 2026-07-16. **Candidate does not mean approved.** A public download, code license, or academic citation does not by itself establish data-subject consent, all media rights, commercial permission, automated-download permission, or suitability for Project PHANTOM.

No dataset is bundled or downloaded by repository scripts. Any field marked **RESEARCH REQUIRED** blocks intake until an accountable reviewer records the answer and source version.

## Decision summary

| Source | Relevant research | Preliminary decision |
|---|---|---|
| RAVDESS | Speech and audiovisual portrayed expression | Candidate for limited non-commercial research; acted English limitations; explicit review/confirmation required |
| CREMA-D | Audio, video, audiovisual portrayed expression | Candidate; access form and ODbL/DbCL obligations; media-rights/legal review required |
| IEMOCAP | Multimodal dyadic expression/conversation | Restricted candidate; official access terms and label details require verification |
| MELD | Multimodal conversational emotion labels | Hold; audiovisual copyright and data-vs-code license unresolved |
| AffectNet | Facial expression/valence/arousal | Hold; academic-use agreement and web-face governance review required |
| UTKFace | Age-related research | Reject for default/commercial use; non-commercial web-face collection and construct concerns |
| FairFace | Age/binary attribute and bias research | Hold; labels do not match perceived-presentation construct; provenance/rights review required |
| Safety evaluation | Supportive/crisis policy | Use synthetic, invented test cases now; no external corpus approved |

## RAVDESS

- **Official name:** The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
- **Official source:** [Zenodo record 10.5281/zenodo.1188976](https://doi.org/10.5281/zenodo.1188976)
- **Intended task:** Research on perceived/portrayed emotional expression in speech, song, audio-video, and video—not internal-state recognition.
- **Modalities:** Audio-only WAV, audio-video MP4, and video-only.
- **Label set:** Speech: neutral, calm, happy, sad, angry, fearful, surprise, disgust; song excludes surprise and disgust. Normal/strong intensity is included, with neutral as an exception.
- **Language:** Two lexically matched English statements in a neutral North American accent.
- **Population limitations:** 24 professional actors (12 described by the source as female and 12 male); narrow acted setting, accent, vocabulary, and performer count. This is not representative of spontaneous, clinical, multilingual, disabled, child, or global populations.
- **License:** The record's rights field states [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Access restrictions:** Public Zenodo files; users remain responsible for the license and citation.
- **Commercial-use restrictions:** Non-commercial license; the record states commercial licenses may be purchased separately.
- **Citation:** Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English*. PLOS ONE, 13(5), e0196391. [DOI](https://doi.org/10.1371/journal.pone.0196391).
- **Known bias/representation issues:** Acted and lexically constrained expression, binary source demographic description, small performer population, North American English, studio conditions. Crowd perception labels are not a person's private emotion.
- **Automated download legally permitted:** **RESEARCH REQUIRED.** Zenodo exposes direct files and MD5 values, but this card does not establish permission for unattended automation. PHANTOM requires an explicit human license confirmation and recorded checksums.
- **Project decision:** Potential baseline only for a declared non-commercial portrayed-expression experiment; never enough alone for a real-world capability claim.

## CREMA-D

- **Official name:** Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D)
- **Official source:** [Maintainer repository](https://github.com/CheyneyComputerScience/CREMA-D)
- **Intended task:** Research on crowd perception of acted audio, visual, and combined audiovisual expression.
- **Modalities:** WAV/MP3 audio, video, audiovisual clips, annotations, and source-described demographic metadata.
- **Label set:** Anger, disgust, fear, happy, neutral, and sad; source also describes four intensity values (low, medium, high, unspecified).
- **Language:** English acted sentences; geographic/dialect details beyond the official card are **RESEARCH REQUIRED**.
- **Population limitations:** Official repository reports 7,442 clips from 91 actors, ages 20–74, with source categories of 48 male/43 female and several race/ethnicity categories. No children; acted fixed sentences; demographic categories are coarse and historically situated.
- **License:** Repository states the database is under [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) and individual contents under [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).
- **Access restrictions:** Maintainers ask users to complete an access form. Media uses Git LFS; a ZIP contains pointer files rather than full media.
- **Commercial-use restrictions:** No explicit non-commercial clause appears in the cited repository section, but ODbL share-alike/attribution, DbCL, performer/media rights, and the requested access process need legal review before any commercial use.
- **Citation:** Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset*. IEEE Transactions on Affective Computing, 5(4), 377–390. [DOI](https://doi.org/10.1109/TAFFC.2014.2336244).
- **Known bias/representation issues:** Acted expression, fixed English sentences, adult-only performers, source-imposed binary sex and broad race/ethnicity categories, crowd-perception labels, laboratory/studio artifacts.
- **Automated download legally permitted:** **Not approved.** Official clone instructions exist, but the requested form, LFS terms, and legal obligations must be completed first. The project script will not clone it.
- **Project decision:** Candidate for controlled research after access, rights, and governance approval; group splits must be actor-independent.

## IEMOCAP

- **Official name:** Interactive Emotional Dyadic Motion Capture (IEMOCAP) database
- **Official source:** [USC SAIL release page](https://sail.usc.edu/iemocap/)
- **Intended task:** Multimodal research on acted/improvised dyadic interaction and annotated expression.
- **Modalities:** **RESEARCH REQUIRED from the current official release documentation** before preparation. The published corpus is commonly described as audiovisual, speech, text, and motion-capture, but the exact downloadable release structure must be checked.
- **Label set:** **RESEARCH REQUIRED.** Record the exact categorical/dimensional annotations and the proposed mapping; do not collapse labels silently.
- **Language:** English; release-specific details require verification.
- **Population limitations:** Small performer population and acted/improvised laboratory conversations; exact release demographics and consent scope require official verification.
- **License:** **RESEARCH REQUIRED.**
- **Access restrictions:** Official distribution historically requires a release agreement; verify the current process directly.
- **Commercial-use restrictions:** **RESEARCH REQUIRED.**
- **Citation:** Busso, C., et al. (2008). *IEMOCAP: Interactive emotional dyadic motion capture database*. Language Resources and Evaluation, 42, 335–359. [DOI](https://doi.org/10.1007/s10579-008-9076-6).
- **Known bias/representation issues:** Small lab population, acted/improvised setting, English, annotation subjectivity, dialogue/session leakage risk.
- **Automated download legally permitted:** **No.** Do not automate a restricted/agreement-based acquisition.
- **Project decision:** Hold until the current official agreement and card are completed.

## MELD

- **Official name:** Multimodal EmotionLines Dataset (MELD)
- **Official source:** [DECLARE Lab repository](https://github.com/declare-lab/MELD)
- **Intended task:** Multimodal emotion/sentiment classification in multi-party conversational utterances.
- **Modalities:** Text, audio, and visual clips.
- **Label set:** Anger, disgust, sadness, joy, neutral, surprise, and fear; sentiment labels positive, neutral, negative.
- **Language:** English dialogue from the television series *Friends*.
- **Population limitations:** Scripted television performances, recurring professional characters/actors, edited production audio/video, US sitcom context. Splitting utterances can leak speakers, scenes, and production artifacts.
- **License:** Repository displays GPL-3.0 for repository content. Whether that license covers all dialogue and audiovisual data is **RESEARCH REQUIRED**.
- **Access restrictions:** Repository points to a hosted dataset distribution; verify its current card/terms and the rights to source clips.
- **Commercial-use restrictions:** **RESEARCH REQUIRED.** Do not infer television-media commercial rights from the code repository's GPL label.
- **Citation:** Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2019). *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation*. ACL 2019. [ACL Anthology](https://aclanthology.org/P19-1050/).
- **Known bias/representation issues:** Scripted/acted content, entertainment stereotypes, English/US cultural scope, recurring cast, laughter/editing/context cues, subjective labels, audiovisual copyright.
- **Automated download legally permitted:** **Not established; do not automate.**
- **Project decision:** Hold pending rights and release-term review. It is not approved for the preparation script.

## AffectNet

- **Official name:** AffectNet
- **Official source:** [AffectNet Academic Use License](https://mohammadmahoor.com/wp-content/uploads/2023/03/AffectNet-Agreement-v2-30Mar2023.pdf) and [project publication](https://doi.org/10.1109/TAFFC.2017.2740923)
- **Intended task:** Research on facial-expression categories and valence/arousal annotations in web-collected face images.
- **Modalities:** Still facial images and annotations.
- **Label set:** Exact current release labels/mappings are **RESEARCH REQUIRED** from the approved distribution documentation; do not rely on a third-party mirror.
- **Language:** Not a speech/text corpus; source query/language and geographic effects require review.
- **Population limitations:** Web-search sampling, unknown real-world contexts, annotation subjectivity, pose/occlusion variation, uncertain consent and demographic representation.
- **License:** Academic-use agreement; exact current terms must be reviewed and accepted by the institution/researcher.
- **Access restrictions:** Agreement-based academic access.
- **Commercial-use restrictions:** Academic-use framing indicates commercial use is not approved under this card; obtain explicit rights separately.
- **Citation:** Mollahosseini, A., Hasani, B., & Mahoor, M. H. (2019). *AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild*. IEEE Transactions on Affective Computing, 10(1), 18–31. [DOI](https://doi.org/10.1109/TAFFC.2017.2740923).
- **Known bias/representation issues:** Web collection and search-engine bias, face privacy/consent concerns, categorical ambiguity, annotator perception rather than internal state, subgroup/skin-tone performance risk.
- **Automated download legally permitted:** **No.** Do not bypass the academic agreement or authentication.
- **Project decision:** Hold; requires legal/ethics, provenance, and construct-validity review.

## UTKFace

- **Official name:** UTKFace
- **Official source:** [UTKFace project page](https://susanqq.github.io/UTKFace/)
- **Intended task:** Source lists face detection, age estimation/progression, landmark localization, and related face research.
- **Modalities:** Web-collected face images, aligned/cropped variants, landmarks, and source-provided age/gender/ethnicity labels.
- **Label set:** Source describes age from 0–116 and gender/ethnicity annotations; exact encodings must be verified before any parsing.
- **Language:** Not applicable to pixels; geographic/search-source coverage is undocumented in the card.
- **Population limitations:** Web images, unknown consent/context, source says age/gender/race ground truth was estimated algorithmically and double-checked, not necessarily self-report. Representation and label validity are unresolved.
- **License:** Official page states non-commercial research purposes only and notes that image copyright belongs to original owners.
- **Access restrictions:** Public links do not resolve underlying individual image rights or ethics approval.
- **Commercial-use restrictions:** Commercial use is prohibited by the stated dataset terms.
- **Citation:** Zhang, Z., Song, Y., & Qi, H. (2017). *Age Progression/Regression by Conditional Adversarial Autoencoder*. CVPR 2017. Verify final bibliographic details against the official proceedings before publication.
- **Known bias/representation issues:** Web-selection bias, estimated sensitive labels, binary gender framing, individual image rights, age error, face privacy, uncertain geographic/skin-tone balance.
- **Automated download legally permitted:** **Not established; do not automate.**
- **Project decision:** Not approved. A broad age-band feature must first pass necessity/proportionality review, and exact-age labels do not solve the project's governance problem.

## FairFace

- **Official name:** FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age
- **Official source:** [Author repository](https://github.com/joojs/fairface) and [WACV paper](https://openaccess.thecvf.com/content/WACV2021/html/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.html)
- **Intended task:** Bias measurement/mitigation research for face attribute classification.
- **Modalities:** Cropped face images with age-group, binary gender, and race-category annotations.
- **Label set:** Age bands and source-defined gender/race classes; verify exact current release mappings from the official files.
- **Language:** Not applicable to pixels; collection geography/culture requires review.
- **Population limitations:** Images derived from a large public-photo collection; balancing selected race categories does not establish representativeness, consent for this inference, or validity of imposed categories.
- **License:** Official repository states CC BY 4.0; verify that release and attribution chain before use.
- **Access restrictions:** Official image/label links are externally hosted; document version and hash.
- **Commercial-use restrictions:** CC BY 4.0 does not contain a non-commercial clause, but downstream image provenance/platform terms, privacy, biometric law, and the intended attribute use require legal review.
- **Citation:** Kärkkäinen, K., & Joo, J. (2021). *FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation*. WACV 2021, 1548–1558.
- **Known bias/representation issues:** Coarse imposed race categories, binary gender labels, appearance annotations that do not equal identity, web-photo selection, privacy/consent, annotation error, category imbalance within intersections.
- **Automated download legally permitted:** **Not established; do not automate.**
- **Project decision:** Not approved for perceived-gender-presentation training. Its binary label does not validate identity or PHANTOM's safeguarded presentation taxonomy. It may be considered only for a separately approved bias-audit study.

## Conversational safety evaluation

No external human-language safety corpus is approved. The repository uses invented, non-graphic synthetic fixtures for deterministic policy tests. Before adding a corpus, create a full card covering privacy, annotation harm taxonomy, license, access, demographic/language coverage, reviewer wellbeing, and whether examples may be redistributed.

Synthetic tests cannot estimate real-world prevalence or clinical risk. A qualified, ethically reviewed human evaluation is required before any stronger safety claim.

## Pretrained-model status

No pretrained speech-emotion, facial-expression, age-band, presentation, multimodal-affect, or conversational-safety weight is selected. Model hub popularity is not evidence of provenance, license compatibility, label validity, calibration, or safety. Complete the weight review in [RESEARCH_REQUIRED.md](../RESEARCH_REQUIRED.md) before naming a model as a dependency.
