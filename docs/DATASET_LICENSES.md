# Dataset Licensing and Provenance

Every dataset used to train a shipped DexteraAI model must be listed here with its
license and commercial-use status **before** the model ships. Model weights derived
from restrictively-licensed data inherit real legal risk, and "we only kept the
landmarks" is not a settled defence.

> This file records what was used and what needs checking. It is not legal advice.
> Have counsel confirm the terms of any dataset before a commercial release.

## Currently used

Licenses below were read at their source on **2026-09-04**, not inferred from
secondary summaries. Re-check on any retrain.

| Dataset | Used for | Source | License as stated at source | Commercial use |
|---|---|---|---|---|
| HaGRID (classification, 512px) | Track 1: 18 general gestures | `Jayabalambika/hagrid-classification-512p-dataset` on HuggingFace, re-hosting HaGRID | Mirror declares **no license**. Upstream (`hukenovs/hagrid`) ships a custom licence, `license/en_us.pdf` | **Permitted**, with conditions — see below |
| ASL Alphabet v03 | Track 2a: 26 ASL fingerspelling letters | `Marxulia/asl_sign_languages_alphabets_v03` on HuggingFace | **No license declared.** Dataset card is empty | **Unresolved** — see below |
| MediaPipe `hand_landmarker.task` | Hand landmark detection, train and inference | Google, `storage.googleapis.com/mediapipe-models` | Apache 2.0 | Yes ✅ |
| MediaPipe `face_landmarker.task` | Non-manual markers (blendshapes) | Google, same origin | Apache 2.0 | Yes ✅ |

### HaGRID: commercial use is allowed; two conditions attach

The upstream licence is **not** CC BY-SA 4.0, though it is widely cited as such.
It is a custom document titled *"Public license with attribution and conditions
reserved"*, and it says so itself in a footnote:

> This license is not a Creative Commons license. The text of this license is a
> reworking of a Creative Commons Corporation (Attribution-ShareAlike 4.0)
> license […] under the terms of the CC0.

Database rights in it run under the Civil Code of the Russian Federation rather
than an EU or US regime.

**There is no non-commercial restriction.** The word "commercial" does not occur
anywhere in the five-page document, and Section 2(a)(1) grants a "worldwide,
royalty-free, non-sublicensable, non-exclusive, irrevocable" licence to reproduce
the material and to create Adapted Material. This is the one point where the
earlier entry in this file was too pessimistic: the data can be used in a paid
product.

Two conditions do attach:

1. **Attribution** (Section 3(a)) — retain creator information, the copyright
   notice, notice of the licence, the warranty disclaimer, and a link to the
   material, and state that changes were made. This repository was not doing
   that; the Attribution section below now discharges it.
2. **ShareAlike** (Section 3(b)) — Adapted Material must be licensed under "a
   license with the same License Elements […] or a BY-SA-compatible license."

### The open question, stated precisely

Condition 2 collides with this repository's MIT licence **if trained weights are
Adapted Material**. MIT imposes no share-alike, so it is not BY-SA-compatible.

Whether a model trained on a dataset is a derivative work of that dataset is
genuinely unsettled law, not a question with a known answer this file is
withholding. Two facts bear on it here, both in the project's favour and neither
dispositive:

- **Only a subset was used.** Roughly 65,802 samples were extracted from a
  507,050-row corpus — about 13%. Section 4(b) attaches Adapted-Material status
  to a database incorporating "all or substantially all" of the content.
- **No source imagery survives.** The pipeline persists 21 landmark coordinates
  per hand and discards the images. Nothing in `models/` or `data/sequences/`
  reconstructs a HaGRID photograph.

That is an argument, not a clearance. **Have counsel settle it before charging
for anything built on `models/hagrid`.** If the answer comes back unfavourable,
the fix is bounded: retrain Track 1 on data with clear terms —
`extract_landmarks.py --image-dir` takes any folder-per-class image set, so no
model or app code changes.

### ASL Alphabet: the higher risk, and the one with no fallback

`Marxulia/asl_sign_languages_alphabets_v03` declares **no licence at all** and
its dataset card is empty. Absent a licence, copyright default is that no rights
are granted — permissive intent cannot be inferred from public availability, and
a re-uploader cannot grant rights they never held. Unlike HaGRID there is no
upstream document to fall back on and no identified rights holder to ask.

This makes `models/asl_alphabet` the **weaker** of the two positions
commercially, despite HaGRID being the one that carries visible conditions. A
licence with obligations is a stronger footing than no licence at all.

Fingerspelling is also the cheapest track to re-source: 26 static handshapes,
8,638 samples, no motion. It can be recorded first-hand in an afternoon, or
taken from a CC0/CC BY set, and retrained with one command.

## Attribution

Provided in satisfaction of Section 3(a) of the HaGRID licence.

`models/hagrid` is trained on hand landmarks extracted from the **HaGRID**
dataset by Alexander Kapitanov, Karina Kvanchiani, Alexander Nagaev, Roman
Kraynov and Andrei Makhliarchuk (SberDevices).

- Dataset and licence: https://github.com/hukenovs/hagrid
- Licence text: https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf
- Paper: *HaGRID — HAnd Gesture Recognition Image Dataset*, WACV 2024,
  https://arxiv.org/abs/2206.08219

**Changes made to the Licensed Material:** images were not redistributed. A
subset was read once to extract MediaPipe hand landmark coordinates, which were
normalised, converted to an 86-dimensional feature vector and used as training
input. The images themselves are not reproduced, stored or distributed by this
project.

The dataset is provided by its licensors without warranties, as set out in
Section 5 of that licence.

## Evaluated and deliberately not used

| Dataset | Why it was considered | Why it is not used |
|---|---|---|
| 20BN-Jester | Large dynamic/temporal gesture set (27 classes), the obvious source for swipes and waves | Distributed under **CC BY-NC-SA 4.0 — non-commercial**. Using it would make the dynamic-gesture model non-commercial too. |
| WLASL | Standard word-level ASL benchmark (~2000 signs) | Distributed under **research-only terms** in the common mirrors, and assembled from third-party videos whose own rights are unclear. Also needs body/face landmarks, not just hands. |

Dynamic gestures and word-level sign language therefore remain **not implemented**
rather than implemented on data that cannot ship. See "Known scope limits" in the
README.

Both exclusions are **settled, not provisional**: the project owner confirmed on
2026-09-04 that DexteraAI may be sold, which removes the research-and-education
route these two corpora are distributed under. Revisit only if a commercial
licence is bought from the rights holders, or the corpus is replaced.

## Rules for adding a dataset

1. Record it in the table above **before** training on it.
2. Capture the license as stated at the source, plus the date checked.
3. If the license is missing or non-commercial, mark the resulting weights clearly
   and keep them out of any commercial build.
4. Prefer datasets with an explicit license over popular ones without.

## Privacy note

Only landmark coordinates (21 points x 3 floats per hand) are persisted by the
extraction pipeline. Source images are read once and discarded, never copied into
`data/sequences/` or into any model artifact. That is a deliberate property of the
pipeline, and it should stay that way.
