/**
 * GestureEngine — Client-side gesture recognition engine.
 *
 * Runs entirely in the browser using:
 *   - MediaPipe Hands (WASM) for landmark detection
 *   - ONNX Runtime Web (WebGPU/WASM) for gesture classification
 *
 * Zero cloud. Zero data leakage. All on-device.
 */

import * as ort from "onnxruntime-web";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface GestureResult {
  gestureName: string;
  gestureId: number;
  confidence: number;
  landmarks: Landmark[] | null;
  handedness: "left" | "right" | "unknown";
  inferenceTimeMs: number;
}

const GESTURE_LABELS = [
  "none",
  "open_palm",
  "closed_fist",
  "thumbs_up",
  "thumbs_down",
  "peace",
  "pointing_up",
  "ok_sign",
  "pinch",
  "wave",
];

const FEATURE_DIM = 86;
const SEQUENCE_LENGTH = 30;

// ---------------------------------------------------------------------------
// Feature Extraction (mirrors core/landmarks/features.py)
// ---------------------------------------------------------------------------

function extractFeatures(landmarks: Landmark[]): Float32Array {
  const features = new Float32Array(FEATURE_DIM);
  let idx = 0;

  // 1. Flattened coordinates (21 * 3 = 63)
  for (const lm of landmarks) {
    features[idx++] = lm.x;
    features[idx++] = lm.y;
    features[idx++] = lm.z;
  }

  // 2. Fingertip-to-wrist distances (5)
  const wrist = landmarks[0];
  const fingertips = [4, 8, 12, 16, 20];
  for (const tip of fingertips) {
    const dx = landmarks[tip].x - wrist.x;
    const dy = landmarks[tip].y - wrist.y;
    const dz = landmarks[tip].z - wrist.z;
    features[idx++] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  // 3. Fingertip pairwise distances (10)
  for (let i = 0; i < fingertips.length; i++) {
    for (let j = i + 1; j < fingertips.length; j++) {
      const a = landmarks[fingertips[i]];
      const b = landmarks[fingertips[j]];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      features[idx++] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
  }

  // 4. Finger curl ratios (5)
  const fingerDefs: [number, number, number][] = [
    [4, 3, 2],   // thumb
    [8, 6, 5],   // index
    [12, 10, 9], // middle
    [16, 14, 13],// ring
    [20, 18, 17],// pinky
  ];
  for (const [tip, pip, mcp] of fingerDefs) {
    const tipToMcp = dist3d(landmarks[tip], landmarks[mcp]);
    const mcpToPip = dist3d(landmarks[mcp], landmarks[pip]);
    const pipToTip = dist3d(landmarks[pip], landmarks[tip]);
    const totalLen = mcpToPip + pipToTip;
    if (totalLen < 1e-6) {
      features[idx++] = 0;
    } else {
      features[idx++] = Math.max(0, Math.min(1, 1 - tipToMcp / totalLen));
    }
  }

  // 5. Palm normal (3)
  const v1 = sub3d(landmarks[5], landmarks[0]);
  const v2 = sub3d(landmarks[17], landmarks[0]);
  const normal = cross3d(v1, v2);
  const normLen = Math.sqrt(
    normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2
  );
  if (normLen > 1e-6) {
    features[idx++] = normal[0] / normLen;
    features[idx++] = normal[1] / normLen;
    features[idx++] = normal[2] / normLen;
  } else {
    features[idx++] = 0;
    features[idx++] = 0;
    features[idx++] = 0;
  }

  return features;
}

function dist3d(a: Landmark, b: Landmark): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function sub3d(a: Landmark, b: Landmark): [number, number, number] {
  return [a.x - b.x, a.y - b.y, a.z - b.z];
}

function cross3d(
  a: [number, number, number],
  b: [number, number, number]
): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

// ---------------------------------------------------------------------------
// GestureEngine
// ---------------------------------------------------------------------------

export class GestureEngine {
  private hands: any | null = null;
  private session: ort.InferenceSession | null = null;
  private sequenceBuffer: Float32Array[] = [];
  private isInitialized = false;

  /**
   * Initialize MediaPipe Hands and ONNX Runtime session.
   */
  async initialize(modelUrl?: string): Promise<void> {
    // Initialize ONNX Runtime with WebGPU if available, fallback to WASM
    try {
      ort.env.wasm.wasmPaths = "/onnx/";

      if (modelUrl) {
        this.session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ["webgpu", "wasm"],
          graphOptimizationLevel: "all",
        });
      }
    } catch (err) {
      console.warn(
        "ONNX model not loaded (running in detection-only mode):",
        err
      );
    }

    this.isInitialized = true;
    console.log("GestureEngine initialized (detection-only mode)");
  }

  /**
   * Process a video frame and return gesture results.
   * All processing happens on-device.
   */
  async processFrame(
    video: HTMLVideoElement
  ): Promise<GestureResult | null> {
    if (!this.isInitialized) return null;

    const t0 = performance.now();

    // For now, return a placeholder — MediaPipe Hands JS integration
    // will be added once the npm package is installed
    // The architecture is ready for it

    const t1 = performance.now();

    return {
      gestureName: "detecting...",
      gestureId: -1,
      confidence: 0,
      landmarks: null,
      handedness: "unknown",
      inferenceTimeMs: t1 - t0,
    };
  }

  /**
   * Run ONNX inference on buffered features.
   */
  private async classifyGesture(): Promise<{
    gestureId: number;
    confidence: number;
  } | null> {
    if (!this.session || this.sequenceBuffer.length < SEQUENCE_LENGTH) {
      return null;
    }

    // Build input tensor: (1, seq_len, feature_dim)
    const inputData = new Float32Array(SEQUENCE_LENGTH * FEATURE_DIM);
    const recent = this.sequenceBuffer.slice(-SEQUENCE_LENGTH);
    for (let i = 0; i < SEQUENCE_LENGTH; i++) {
      inputData.set(recent[i], i * FEATURE_DIM);
    }

    const inputTensor = new ort.Tensor("float32", inputData, [
      1,
      SEQUENCE_LENGTH,
      FEATURE_DIM,
    ]);

    const maskData = new Uint8Array(SEQUENCE_LENGTH).fill(0);
    const maskTensor = new ort.Tensor("bool", maskData, [1, SEQUENCE_LENGTH]);

    const results = await this.session.run({
      input: inputTensor,
      mask: maskTensor,
    });

    const logits = results["logits"].data as Float32Array;

    // Softmax
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map((l: number) => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a: number, b: number) => a + b, 0);
    const probs = expLogits.map((e: number) => e / sumExp);

    let maxIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }

    return {
      gestureId: maxIdx,
      confidence: probs[maxIdx],
    };
  }

  /**
   * Add features to the temporal buffer.
   */
  private pushFeatures(landmarks: Landmark[]): void {
    const features = extractFeatures(landmarks);
    this.sequenceBuffer.push(features);
    if (this.sequenceBuffer.length > SEQUENCE_LENGTH * 2) {
      this.sequenceBuffer = this.sequenceBuffer.slice(-SEQUENCE_LENGTH);
    }
  }

  /**
   * Release all resources.
   */
  dispose(): void {
    this.session?.release();
    this.session = null;
    this.hands = null;
    this.sequenceBuffer = [];
    this.isInitialized = false;
  }
}
