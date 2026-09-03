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

echo "Done. Runtime files in $DEST:"
ls -1 "$DEST"/*.wasm 2>/dev/null | wc -l | xargs echo "  wasm files:"
ls -1 "$DEST/mediapipe" 2>/dev/null | wc -l | xargs echo "  mediapipe files:"

cat <<'NOTE'

Model bundles (gesture.onnx + labels.json) are produced by training, not npm:
    python dextera.py train --dataset data/sequences/<name> --export models/<name>
    cp models/<name>/gesture.onnx* models/<name>/labels.json apps/web/public/onnx/<name>/
NOTE
