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
| ASL-HG | Track 2a: 35 fingerspelling classes (A-Z, 1-9) | [Mendeley `j4y5w2c8w9`](https://data.mendeley.com/datasets/j4y5w2c8w9/1) | **CC BY 4.0**, confirmed from Mendeley's own licence field | **Yes** — attribution only, no share-alike ✅ |
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

### The share-alike condition, and how it is met

Condition 2 collides with an MIT-only repository: MIT imposes no share-alike, so
it is not BY-SA-compatible. The project owner's decision to **open source
DexteraAI** (2026-09-04) makes this straightforward to satisfy rather than
something to argue about.

The repository is therefore licensed in two parts:

| What | License | Where |
|---|---|---|
| All source code | MIT | `/LICENSE` |
| `models/hagrid` weights | CC BY-SA 4.0 | `models/LICENSE` |

CC BY-SA 4.0 carries the same License Elements the HaGRID licence names, so
distributing the weights under it discharges Section 3(b) whether or not trained
weights are legally Adapted Material. That question — genuinely unsettled, and
not one this file is pretending to answer — no longer needs answering for
Track 1, because the obligation is met either way.

Adopters should know this is a real constraint on them: anyone redistributing an
adaptation of `models/hagrid` inherits share-alike. It is not viral into your
application code, which stays MIT; it attaches to the weights. Retraining on
data without a share-alike obligation removes it, and the fallback is bounded —
`extract_landmarks.py --image-dir` takes any folder-per-class image set, so no
model or app code changes.

### ASL Alphabet: resolved, and the model is better for it

This was the project's one unlicensable bundle. `Marxulia/asl_sign_languages_alphabets_v03`
declared no licence and carried an empty dataset card, so no rights passed to
this project and there was nothing to sublicense. It has been **replaced, not
relicensed**, by ASL-HG — CC BY 4.0, attribution only, no share-alike — so
`models/asl_alphabet` now ships MIT alongside the code.

Replacing it improved the model rather than merely legalising it:

| | old (unlicensed) | ASL-HG |
|---|---|---|
| Classes | 26 letters | 35 (A-Z and 1-9) |
| Landmark samples | 8,638 | 27,984 |
| MediaPipe detection rate | 79% | **99.9%** |
| Subject IDs | none | 10 participants, named in every filename |

The last row mattered most. Every accuracy figure this project had published
came from a random split, which cannot say how a model does on a new person:
the same hands land on both sides of it. ASL-HG names the participant in each
filename, so the split is now subject-disjoint — trained on P1-P8, tested on
P9-P10, who appear in no training image.

That produced the first honest generalisation number here, and it sits 9 points
below what a random split claims. Both are recorded in the bundle manifest and
the model card.

The two-handed `0` class is excluded: ASL-HG uses the two-handed sign for zero
to keep it distinct from the letter O, and this pipeline encodes one hand in 86
dimensions, so that class could never fire correctly.

Reproduce with `make retrain-asl-clean`, after downloading the archive by hand —
Mendeley does not serve files to its API.

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
