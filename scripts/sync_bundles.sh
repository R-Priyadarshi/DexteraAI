#!/usr/bin/env bash
# Publish trained model bundles into the web app's static assets.
#
# `models/<name>/` is where training writes; `apps/web/public/onnx/<name>/` is
# what the browser fetches. They are separate copies, so anything that rewrites
# a bundle — retraining, or `scripts/calibrate_bundle.py` fitting a temperature
# onto an existing checkpoint — updates `models/` and leaves the browser on the
# old file.
#
# That drift is silent and easy to miss, because `labels.json` still parses and
# the app still runs: it just falls back to a generic confidence threshold
# instead of the calibrated one, and reports accuracy the model no longer has.
#
# Run this after any training or calibration run.
#
#   scripts/sync_bundles.sh            # copy models/ -> public/onnx/
#   scripts/sync_bundles.sh --check    # report drift, change nothing (exit 1)
set -euo pipefail

cd "$(dirname "$0")/.."

SRC_ROOT="models"
DEST_ROOT="apps/web/public/onnx"
BUNDLES=(hagrid asl_alphabet motion)
FILES=(gesture.onnx gesture.onnx.data labels.json)

check_only=0
[ "${1:-}" = "--check" ] && check_only=1

drift=0
copied=0

for bundle in "${BUNDLES[@]}"; do
  src="$SRC_ROOT/$bundle"
  # Bundles are optional: `motion` only exists once someone has recorded and
  # trained dynamic gestures.
  [ -d "$src" ] || continue
  dest="$DEST_ROOT/$bundle"
  mkdir -p "$dest"

  for file in "${FILES[@]}"; do
    # gesture.onnx.data is absent for models small enough to need no external
    # data, so a missing file here is not an error.
    [ -f "$src/$file" ] || continue

    if [ -f "$dest/$file" ] && cmp -s "$src/$file" "$dest/$file"; then
      continue
    fi

    drift=1
    if [ "$check_only" -eq 1 ]; then
      echo "DRIFT  $bundle/$file"
    else
      cp "$src/$file" "$dest/$file"
      echo "synced $bundle/$file"
      copied=$((copied + 1))
    fi
  done
done

if [ "$check_only" -eq 1 ]; then
  if [ "$drift" -eq 1 ]; then
    echo
    echo "Served bundles are stale. Run: scripts/sync_bundles.sh" >&2
    exit 1
  fi
  echo "Served bundles are up to date."
  exit 0
fi

if [ "$copied" -eq 0 ]; then
  echo "Already up to date."
fi
