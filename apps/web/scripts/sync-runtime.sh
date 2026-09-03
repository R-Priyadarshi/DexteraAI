#!/usr/bin/env bash
# Copy the WASM runtimes the browser engine loads at runtime into public/onnx/.
#
# These are large binaries that ship inside npm packages, so they are not kept in
# git. Run this after `npm install` (it is wired to `postinstall`) or any time
# onnxruntime-web / @mediapipe/hands is upgraded.
#
# The gesture engine sets ort.env.wasm.wasmPaths = "/onnx/" and MediaPipe's
# locateFile points at "/onnx/mediapipe/", so both expect these paths.
set -euo pipefail

cd "$(dirname "$0")/.."

ORT_SRC="node_modules/onnxruntime-web/dist"
MP_SRC="node_modules/@mediapipe/hands"
TASKS_SRC="node_modules/@mediapipe/tasks-vision/wasm"
# Model bundles the dashboard offers; keep in step with MODEL_BUNDLES in
# src/app/dashboard/page.tsx.
BUNDLES=(hagrid asl_alphabet)
FACE_MODEL="../../models/mediapipe/face_landmarker.task"
DEST="public/onnx"

if [ ! -d "$ORT_SRC" ]; then
  echo "error: $ORT_SRC not found. Run 'npm install' first." >&2
  exit 1
fi

mkdir -p "$DEST/mediapipe"

echo "Syncing ONNX Runtime WASM -> $DEST"
cp -f "$ORT_SRC"/*.wasm "$DEST"/ 2>/dev/null || true
cp -f "$ORT_SRC"/ort-wasm*.mjs "$DEST"/ 2>/dev/null || true

if [ -d "$MP_SRC" ]; then
  echo "Syncing MediaPipe Hands assets -> $DEST/mediapipe"
  cp -f "$MP_SRC"/* "$DEST/mediapipe"/ 2>/dev/null || true
else
  echo "warning: $MP_SRC not found; skipping MediaPipe assets" >&2
fi

# The face landmarker runs the Tasks API, which loads its own WASM — a
# different runtime from the legacy hands solution above, so both ship.
if [ -d "$TASKS_SRC" ]; then
  echo "Syncing MediaPipe Tasks WASM -> $DEST/tasks"
  mkdir -p "$DEST/tasks"
  cp -f "$TASKS_SRC"/* "$DEST/tasks"/ 2>/dev/null || true
else
  echo "warning: $TASKS_SRC not found; face markers will be unavailable" >&2
fi

# Fetched by `make fetch-models`, not shipped in npm or git.
if [ -f "$FACE_MODEL" ]; then
  echo "Syncing face_landmarker.task -> $DEST/mediapipe"
  cp -f "$FACE_MODEL" "$DEST/mediapipe"/ 2>/dev/null || true
else
  echo "note: $FACE_MODEL missing — run 'make fetch-models' for face markers" >&2
fi

# Trained gesture bundles. These live in models/<name>/ under version control
# and are copied here because Next only serves what is under public/. Keeping
# one copy tracked instead of two keeps the weights from being stored twice.
for bundle in "${BUNDLES[@]}"; do
  src="../../models/$bundle"
  if [ -f "$src/gesture.onnx" ]; then
    echo "Syncing $bundle model bundle -> $DEST/$bundle"
    mkdir -p "$DEST/$bundle"
    cp -f "$src/gesture.onnx" "$src/labels.json" "$DEST/$bundle"/
    # Weights live beside the graph as external data whenever the exporter
    # split them out; a bundle exported below the threshold has none.
    [ -f "$src/gesture.onnx.data" ] && cp -f "$src/gesture.onnx.data" "$DEST/$bundle"/
  else
    echo "warning: $src/gesture.onnx missing; '$bundle' will not load" >&2
  fi
done

echo "Done. Runtime files in $DEST:"
ls -1 "$DEST"/*.wasm 2>/dev/null | wc -l | xargs echo "  wasm files:"
ls -1 "$DEST/mediapipe" 2>/dev/null | wc -l | xargs echo "  mediapipe files:"

ls -1 "$DEST"/*/gesture.onnx 2>/dev/null | wc -l | xargs echo "  model bundles:"

cat <<'NOTE'

To add a bundle: train and export it, then list it in BUNDLES above and in
MODEL_BUNDLES in src/app/dashboard/page.tsx.
    python dextera.py train --dataset data/sequences/<name> --export models/<name>
NOTE
