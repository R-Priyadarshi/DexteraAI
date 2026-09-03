# Dataset Licensing and Provenance

Every dataset used to train a shipped DexteraAI model must be listed here with its
license and commercial-use status **before** the model ships. Model weights derived
from restrictively-licensed data inherit real legal risk, and "we only kept the
landmarks" is not a settled defence.

> This file records what was used and what needs checking. It is not legal advice.
> Have counsel confirm the terms of any dataset before a commercial release.

## Currently used

| Dataset | Used for | Source | Stated license | Commercial use | Status |
|---|---|---|---|---|---|
| HaGRID (classification, 512px) | Track 1: 18 general gestures | `Jayabalambika/hagrid-classification-512p-dataset` on HuggingFace, re-hosting the HaGRID dataset | Upstream HaGRID is published by SberDevices; commonly cited as CC BY-SA 4.0. The HuggingFace mirror declares **no license field**. | **Unconfirmed** | ⚠️ Verify before commercial release |
| ASL Alphabet v03 | Track 2a: 26 ASL fingerspelling letters | `Marxulia/asl_sign_languages_alphabets_v03` on HuggingFace | **No license declared** on the dataset card | **Unconfirmed** | ⚠️ Verify before commercial release |
| MediaPipe `hand_landmarker.task` | Hand landmark detection at train and inference time | Google, `storage.googleapis.com/mediapipe-models` | Apache 2.0 (MediaPipe) | Yes | ✅ OK |

### What "unconfirmed" means here

Both training datasets are community re-uploads that carry no explicit license
metadata. Re-hosting does not grant rights the uploader did not have. Before any
paid or commercial distribution of models trained on them, do one of:

1. Confirm the upstream license permits commercial use, and comply with its terms
   (attribution, and for ShareAlike, the downstream obligations it imposes).
2. Re-train on data you have clear rights to (own collection, or a dataset with an
   explicit permissive license).
3. Ship these as research/preview weights only, clearly labelled, and keep them out
   of the commercial artifact.

The pipeline makes option 2 cheap: `training/datasets/extract_landmarks.py` accepts
`--image-dir` for any folder-per-class image set, so swapping the data source does
not require touching the model code.

## Evaluated and deliberately not used

| Dataset | Why it was considered | Why it is not used |
|---|---|---|
| 20BN-Jester | Large dynamic/temporal gesture set (27 classes), the obvious source for swipes and waves | Distributed under **CC BY-NC-SA 4.0 — non-commercial**. Using it would make the dynamic-gesture model non-commercial too. |
| WLASL | Standard word-level ASL benchmark (~2000 signs) | Distributed under **research-only terms** in the common mirrors, and assembled from third-party videos whose own rights are unclear. Also needs body/face landmarks, not just hands. |

Dynamic gestures and word-level sign language therefore remain **not implemented**
rather than implemented on data that cannot ship. See "Known scope limits" in the
README.

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
