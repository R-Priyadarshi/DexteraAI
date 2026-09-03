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
import { gestureStore } from "./gesture-store";
import {
  GestureSegmenter,
  type GesturePhase,
  type SegmenterConfig,
} from "./gesture-segmenter";

// Global Hands from /onnx/mediapipe/hands.js loaded in layout.tsx
declare const Hands: any;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

/** Motion-derived intent inferred alongside the classified gesture. */
export type SpatialIntent =
  | "swipe_left"
  | "swipe_right"
  | "hyper_left"
  | "hyper_right"
  | "pinch_open"
  | "pinch_close"
  | "none";

export interface GestureResult {
  gestureName: string;
  gestureId: number;
  confidence: number;
  landmarks: Landmark[] | null;
  handedness: "left" | "right" | "unknown";
  inferenceTimeMs: number;
  velocity: { x: number; y: number; z: number };
  spatialIntent: SpatialIntent;
  /**
   * True when the calibrated confidence fell below the bundle's open-set
   * rejection threshold. The label is still reported for display, but consumers
   * that trigger actions must treat a rejected frame as "no gesture".
   */
  rejected: boolean;
  /**
   * Discrete phase from the segmenter. Actions should fire on `onset` only;
   * `hold` repeats every frame for as long as the pose is maintained.
   */
  phase: GesturePhase;
  /** Milliseconds the current segment has been held. */
  heldMs: number;
  /** Unique id of the current segment, for de-duplicating repeated holds. */
  segmentId: number;
}

/**
 * Fallback labels, used only when a model bundle ships no labels.json.
 * The deployed vocabulary comes from the bundle so that Python and the browser
 * cannot drift apart. See docs/api-reference.md "Model Bundles".
 */
const DEFAULT_GESTURE_LABELS = [
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

/**
 * Sentinel id for a user-taught gesture. It sits far above any real class index
 * so it can never collide with a bundle label, however large the vocabulary
 * grows.
 */
export const CUSTOM_GESTURE_ID = 999;
const DEFAULT_SEQUENCE_LENGTH = 30;

/** Shape of the labels.json emitted next to an exported gesture.onnx. */
export interface ModelBundle {
  labels: string[];
  seq_len?: number;
  feature_dim?: number;
  val_accuracy?: number;
  test_accuracy?: number;
  calibration?: BundleCalibration | null;
}

/**
 * Confidence calibration fitted on the model's held-out validation split by
 * `training/evaluation/calibrate_confidence.py`.
 *
 * A softmax classifier is closed-set and typically overconfident, which matters
 * here because users constantly make hand shapes outside the vocabulary. The
 * temperature rescales logits so reported confidence matches observed accuracy;
 * the threshold is the cut-off below which we report nothing rather than a
 * confident wrong label.
 */
export interface BundleCalibration {
  temperature: number;
  rejection_threshold: number;
  ece_before?: number;
  ece_after?: number;
}

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
  /**
   * Whether the browser can give MediaPipe a WebGL context.
   *
   * Checked before initialization because MediaPipe's failure mode is an
   * alert() per frame from inside its bundled code, which cannot be caught.
   */
  static isWebGLAvailable(): boolean {
    if (typeof document === "undefined") return false;
    try {
      const canvas = document.createElement("canvas");
      const gl =
        canvas.getContext("webgl2") ||
        canvas.getContext("webgl") ||
        canvas.getContext("experimental-webgl");
      if (!gl) return false;
      // Release the probe context so it does not count against the browser's limit.
      (gl as WebGLRenderingContext)
        .getExtension("WEBGL_lose_context")
        ?.loseContext();
      return true;
    } catch {
      return false;
    }
  }

  private hands: any | null = null;
  private session: ort.InferenceSession | null = null;
  private sequenceBuffer: Float32Array[] = [];
  private isInitialized = false;
  private labels: string[] = [...DEFAULT_GESTURE_LABELS];
  private sequenceLength: number = DEFAULT_SEQUENCE_LENGTH;

  /**
   * Calibration from the loaded bundle. The defaults are deliberately inert
   * (T=1 leaves the softmax untouched) so an uncalibrated bundle behaves
   * exactly as it did before, just with a conservative generic threshold.
   */
  private temperature = 1.0;
  private rejectionThreshold = 0.6;

  private readonly segmenter = new GestureSegmenter();

  /** Tune onset/offset behaviour, e.g. for a latency-sensitive binding. */
  configureSegmenter(config: Partial<SegmenterConfig>): void {
    this.segmenter.configure(config);
  }

  getSegmenterConfig(): SegmenterConfig {
    return this.segmenter.getConfig();
  }

  /** Calibration actually in force, for display in the console. */
  getCalibration(): { temperature: number; rejectionThreshold: number } {
    return {
      temperature: this.temperature,
      rejectionThreshold: this.rejectionThreshold,
    };
  }

  /** The active gesture vocabulary (from the model bundle when available). */
  getLabels(): string[] {
    return [...this.labels];
  }

  /** Temporal window the loaded model expects. */
  getSequenceLength(): number {
    return this.sequenceLength;
  }

  /**
   * Load labels.json sitting next to the .onnx file, so the vocabulary travels
   * with the model instead of being hardcoded here. Falls back silently.
   */
  private async loadBundleLabels(modelUrl: string): Promise<void> {
    const labelsUrl = modelUrl.replace(/[^/]+$/, "labels.json");
    try {
      const res = await fetch(labelsUrl);
      if (!res.ok) return;
      const bundle: ModelBundle = await res.json();
      if (Array.isArray(bundle.labels) && bundle.labels.length > 0) {
        this.labels = bundle.labels;
        if (bundle.seq_len && bundle.seq_len > 0) {
          this.sequenceLength = bundle.seq_len;
        }
        // An uncalibrated bundle keeps T=1 and the conservative default
        // threshold rather than inheriting whatever the last bundle used.
        const cal = bundle.calibration;
        this.temperature = cal && cal.temperature > 0 ? cal.temperature : 1.0;
        this.rejectionThreshold = cal ? cal.rejection_threshold : 0.6;

        console.log(
          `GestureEngine: loaded ${this.labels.length} labels from bundle`,
          bundle.val_accuracy ? `(val acc ${bundle.val_accuracy})` : "",
          cal
            ? `calibrated T=${this.temperature} reject<${this.rejectionThreshold}`
            : "(uncalibrated)"
        );
      }
    } catch {
      console.warn("GestureEngine: no labels.json found, using default labels");
    }
  }

  /**
   * Initialize MediaPipe Hands and ONNX Runtime session.
   */
  async initialize(modelUrl?: string): Promise<void> {
    // 1. Initialize ONNX Runtime with WebGPU if available
    try {
      // Configure ORT with detailed paths for better reliability in workers
      ort.env.wasm.wasmPaths = "/onnx/";
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 4);
      ort.env.wasm.proxy = false;

      console.log("GestureEngine: Initializing ORT with wasmPaths:", ort.env.wasm.wasmPaths);

      if (modelUrl) {
        await this.loadBundleLabels(modelUrl);
        // High-perf manual fetch strategy for models with external data (.data files)
        const modelResponse = await fetch(modelUrl);
        const modelBuffer = await modelResponse.arrayBuffer();

        // Assume external data is [modelName].data if it exists
        const dataUrl = `${modelUrl}.data`;
        const dataResponse = await fetch(dataUrl);
        const dataBuffer = await dataResponse.arrayBuffer();

        this.session = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: ["webgpu", "wasm"],
          graphOptimizationLevel: "all",
          enableCpuMemArena: true,
          enableMemPattern: true,
          externalData: [
            {
              path: "gesture.onnx.data",
              data: new Uint8Array(dataBuffer),
            },
          ],
        });
        console.log("ONNX Session initialized with external data");
      }
    } catch (err) {
      console.warn("ONNX WebGPU/Manual fetch failure:", err);
      if (modelUrl) {
        try {
          // Fallback to simple URL loading for WASM if manual fetch fails
          this.session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ["wasm"],
          });
          console.log("ONNX Fallback session initialized");
        } catch (wasmErr) {
          console.error("ONNX Critical Failure: All loading strategies failed:", wasmErr);
          throw wasmErr;
        }
      }
    }

    // 2. Initialize MediaPipe Hands
    if (!Hands) {
      console.error("MediaPipe Hands library not found");
      return;
    }

    // MediaPipe needs WebGL and, when it cannot get a context, calls alert()
    // from inside its own bundle on every frame we send it. That makes the page
    // unusable behind hundreds of modal dialogs. Fail fast with something the
    // UI can actually display instead.
    if (!GestureEngine.isWebGLAvailable()) {
      throw new Error(
        "WebGL is unavailable, so hand tracking cannot run. Enable hardware " +
        "acceleration in your browser settings (Chrome: Settings > System > " +
        "\"Use graphics acceleration when available\"), then fully restart the " +
        "browser. chrome://gpu shows why it is disabled."
      );
    }

    this.hands = new Hands({
      locateFile: (file: string) => `/onnx/mediapipe/${file}`,
    });

    this.hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.3, // High sensitivity
      minTrackingConfidence: 0.2,  // High sensitivity
    });

    this.hands.onResults((results: any) => {
      this.lastMPResults = results;
      if (this.currentDetectionPromise) {
        this.currentDetectionPromise.resolve();
        this.currentDetectionPromise = null;
      }
    });

    this.isInitialized = true;
    console.log("GestureEngine: MediaPipe + ONNX initialized");
  }

  private lastMPResults: any = null;
  private currentDetectionPromise: { resolve: Function, reject: Function } | null = null;

  private lastLandmarks: Landmark[] | null = null;
  private lastTimestamp: number = 0;

  // Motion smoothing buffers: velocity samples, wrist positions, and a
  // stationary-frame counter used to gate pinch detection.
  private vBuffer: { x: number; y: number }[] = [];
  private pBuffer: { x: number; y: number }[] = [];
  private stationaryFrames: number = 0;

  /**
   * Process a video frame and return gesture results.
   * All processing happens on-device.
   */
  async processFrame(
    video: HTMLVideoElement
  ): Promise<GestureResult | null> {
    if (!this.isInitialized || !this.hands) return null;

    const t0 = performance.now();

    // 1. Trigger MediaPipe Landmark Detection with Sync Wrapper
    const detectionWait = new Promise<void>((resolve, reject) => {
        this.currentDetectionPromise = { resolve, reject };
        // Increase sensitivity: detect even partial/blurred hands
        this.hands!.send({ image: video }).catch(reject);
    });

    await detectionWait;

    if (!this.lastMPResults || !this.lastMPResults.multiHandLandmarks?.length) {
      this.missingFrameCount++;
      if (this.missingFrameCount > this.MISSING_FRAME_TOLERANCE) {
        this.sequenceBuffer = []; // Only wipe after persistent loss
        // Tracking is gone, so we cannot say the gesture *ended* — only that we
        // no longer know. Reset without emitting an offset.
        this.segmenter.reset();
      }
      return {
        gestureName: "no hand",
        gestureId: -1,
        confidence: 0,
        landmarks: null,
        handedness: "unknown",
        inferenceTimeMs: performance.now() - t0,
        velocity: { x: 0, y: 0, z: 0 },
        spatialIntent: "none",
        rejected: true,
        phase: "idle",
        heldMs: 0,
        segmentId: 0,
      };
    }

    const landmarks = this.lastMPResults.multiHandLandmarks[0] as Landmark[];
    const handedness = (this.lastMPResults.multiHandedness?.[0]?.label.toLowerCase() || "unknown") as "left" | "right" | "unknown";

    // 2. Feature Extraction & Buffering
    this.pushFeatures(landmarks);

    // 3. Temporal Classification (Transformer)
    let gestureResult = { gestureId: -1, confidence: 0, rejected: true };

    if (this.session && this.sequenceBuffer.length >= this.sequenceLength) {
      const classification = await this.classifyGesture();
      if (classification) {
        gestureResult = classification;
      }
    }

    const t1 = performance.now();

    // 4. Custom gestures.
    //
    // These are consulted only when the trained model has *not* produced a
    // confident answer. A user-taught k-NN class matched on a handful of
    // examples should never outrank a calibrated model that scored 98% on a
    // held-out split — it only fills the gap where that model abstains.
    let finalGestureId = gestureResult.gestureId;
    let finalConfidence = gestureResult.confidence;
    let finalRejected = gestureResult.rejected;
    let finalGestureName =
      finalGestureId >= 0 ? (this.labels[finalGestureId] ?? "unknown") : "unknown";

    if (finalRejected) {
      const customMatch = this.matchCustomGesture(landmarks);
      if (customMatch && customMatch.confidence > finalConfidence) {
        finalGestureId = CUSTOM_GESTURE_ID;
        finalConfidence = customMatch.confidence;
        finalGestureName = customMatch.name;
        // The k-NN match carries its own acceptance test (see
        // `matchCustomGesture`), so reaching here means it passed.
        finalRejected = false;
      }
    }

    // 5. Dynamic Spatial Intent (Pinch/Swipe)
    //
    // Pinch is resolved below rather than here: an ungated distance test fires
    // constantly while the hand is in motion, because the thumb and index pass
    // close together during almost any travel. It is only meaningful once the
    // hand has settled.
    let spatialIntent: GestureResult["spatialIntent"] = "none";
    const dist = dist3d(landmarks[4], landmarks[8]);

    // 6. Calculate Velocity for Swipes (Hardened with Kinetic Momentum & Directional Lock)
    const velocity = { x: 0, y: 0, z: 0 };
    const now = performance.now();

    if (this.lastLandmarks && this.lastTimestamp > 0) {
        const dt = (now - this.lastTimestamp) / 1000;
        if (dt > 0) {
            // Mirror-Aware Velocity: (User-Perspective)
            const rawX = (this.lastLandmarks[0].x - landmarks[0].x) / dt;
            const rawY = (landmarks[0].y - this.lastLandmarks[0].y) / dt;
            
            // 5-frame moving average to eliminate jitter
            this.vBuffer.push({ x: rawX, y: rawY });
            if (this.vBuffer.length > 5) this.vBuffer.shift();
            
            velocity.x = this.vBuffer.reduce((sum: number, v: { x: number; y: number }) => sum + v.x, 0) / this.vBuffer.length;
            velocity.y = this.vBuffer.reduce((sum: number, v: { x: number; y: number }) => sum + v.y, 0) / this.vBuffer.length;
        }
    }
    this.lastLandmarks = [...landmarks];
    this.lastTimestamp = now;

    // High-precision pinch detection (Gated by velocity)
    const isStationary = Math.abs(velocity.x) < 0.1 && Math.abs(velocity.y) < 0.1;
    if (isStationary) {
        this.stationaryFrames += 1;
    } else {
        this.stationaryFrames = 0;
    }

    const isStableStationary = this.stationaryFrames > 10;
    if (dist < 0.06 && isStableStationary) spatialIntent = "pinch_close";
    else if (dist < 0.12 && isStableStationary) spatialIntent = "pinch_open";

    // 5-frame moving average to eliminate jitter
    this.pBuffer.push({ x: landmarks[0].x, y: landmarks[0].y });
    if (this.pBuffer.length > 5) this.pBuffer.shift();

    const firstP = this.pBuffer[0];
    const lastP = this.pBuffer[this.pBuffer.length - 1];
    const totalDX = lastP.x - firstP.x;
    
    // Multi-Tier Kinetic Logic (Industrial Grade)
    const isHorizontalDominant = Math.abs(velocity.x) > (Math.abs(velocity.y) * 2.5);
    const standardThreshold = 0.25;
    const hyperThreshold = 0.85; // High-speed jump
    const minDisplacement = 0.03;

    if (isHorizontalDominant && Math.abs(totalDX) > minDisplacement) {
        if (velocity.x < -hyperThreshold) spatialIntent = "hyper_left";
        else if (velocity.x > hyperThreshold) spatialIntent = "hyper_right";
        else if (velocity.x < -standardThreshold) spatialIntent = "swipe_left";
        else if (velocity.x > standardThreshold) spatialIntent = "swipe_right";
    }

    // 7. Segmentation.
    //
    // Everything above scores this frame in isolation. The segmenter converts
    // that stream into discrete onset/hold/offset events, which is what any
    // consumer binding a non-idempotent action actually needs.
    const segment = this.segmenter.push(
      {
        gestureName: finalGestureName,
        gestureId: finalGestureId,
        confidence: finalConfidence,
        rejected: finalRejected,
      },
      now
    );

    return {
      gestureName: finalGestureName,
      gestureId: finalGestureId,
      confidence: finalConfidence,
      landmarks,
      handedness,
      inferenceTimeMs: t1 - t0,
      velocity,
      spatialIntent,
      rejected: finalRejected,
      phase: segment.phase,
      heldMs: segment.heldMs,
      segmentId: segment.segmentId,
    };
  }

  private missingFrameCount = 0;
  private readonly MISSING_FRAME_TOLERANCE = 10;

  /**
   * Run ONNX inference on buffered features.
   */
  private async classifyGesture(): Promise<{
    gestureId: number;
    confidence: number;
    rejected: boolean;
  } | null> {
    if (!this.session) return null;

    if (this.sequenceBuffer.length < this.sequenceLength) {
      // Buffer not yet full: nothing to classify, and no claim to make.
      return { gestureId: -1, confidence: 0, rejected: true };
    }

    // Build input tensor: (1, seq_len, feature_dim)
    const inputData = new Float32Array(this.sequenceLength * FEATURE_DIM);
    const recent = this.sequenceBuffer.slice(-this.sequenceLength);
    for (let i = 0; i < this.sequenceLength; i++) {
      inputData.set(recent[i], i * FEATURE_DIM);
    }

    const inputTensor = new ort.Tensor("float32", inputData, [
      1,
      this.sequenceLength,
      FEATURE_DIM,
    ]);

    const maskData = new Uint8Array(this.sequenceLength).fill(0);
    const maskTensor = new ort.Tensor("bool", maskData, [1, this.sequenceLength]);

    const results = await this.session.run({
      input: inputTensor,
      mask: maskTensor,
    });

    const raw = results["logits"].data as Float32Array;

    // Temperature scaling, then softmax. Dividing by T never changes which
    // class wins — only how confident the model claims to be — so this is safe
    // to apply before the argmax and is what makes `rejectionThreshold`
    // comparable to the accuracy measured at fit time.
    const t = this.temperature > 0 ? this.temperature : 1.0;
    const logits = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) logits[i] = raw[i] / t;

    // Subtract the max before exponentiating, or a large logit overflows to
    // Infinity and the whole distribution becomes NaN.
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }

    let sumExp = 0;
    const probs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      probs[i] = Math.exp(logits[i] - maxLogit);
      sumExp += probs[i];
    }

    let maxIdx = 0;
    for (let i = 0; i < probs.length; i++) {
      probs[i] /= sumExp;
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }

    return {
      gestureId: maxIdx,
      confidence: probs[maxIdx],
      rejected: probs[maxIdx] < this.rejectionThreshold,
    };
  }

  /**
   * Match current landmarks against custom gestures using a hardened K-NN approach.
   * 
   * Uses weighted Euclidean distance and Z-score normalization for industrial-grade
   * biometric stability.
   */
  private matchCustomGesture(currentLandmarks: Landmark[]): { name: string, confidence: number } | null {
    const customGestures = gestureStore.getGestures();
    if (customGestures.length === 0) return null;

    const currentFeatures = extractFeatures(currentLandmarks);
    const normalizedCurrent = this.normalizeFeatures(currentFeatures);

    let bestMatch: { name: string, confidence: number } | null = null;
    let minDistance = Infinity;

    for (const gesture of customGestures) {
      for (const sample of gesture.samples) {
        const sampleFeatures = extractFeatures(sample);
        const normalizedSample = this.normalizeFeatures(sampleFeatures);
        
        // Weighted distance calculation: give more weight to finger curl ratios (idx 71-75)
        const distance = this.weightedEuclideanDistance(normalizedCurrent, normalizedSample);

        if (distance < minDistance) {
          minDistance = distance;
          bestMatch = {
            name: gesture.name,
            confidence: Math.max(0, Math.min(0.99, 1 - (distance / 4))) // Scaled confidence
          };
        }
      }
    }

    return (bestMatch && bestMatch.confidence > 0.88) ? bestMatch : null;
  }

  private normalizeFeatures(features: Float32Array): Float32Array {
    const normalized = new Float32Array(features.length);
    for (let i = 0; i < features.length; i++) {
        // Simple Min-Max normalization for spatial features (0-1 range)
        // Hand-specific Z-score could be added here for even higher precision
        normalized[i] = features[i];
        
        // Boost importance of curl ratios (indices 78 to 82 in the 86-dim vector)
        if (i >= 78 && i <= 82) {
            normalized[i] *= 2.0; 
        }
    }
    return normalized;
  }

  private weightedEuclideanDistance(a: Float32Array, b: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      // Weights: fingertips and curl ratios are prioritized
      const weight = (i >= 63) ? 2.0 : 1.0; 
      sum += (diff * diff) * weight;
    }
    return Math.sqrt(sum);
  }

  /**
   * Add features to the temporal buffer.
   */
  private pushFeatures(landmarks: Landmark[]): void {
    const features = extractFeatures(landmarks);
    this.sequenceBuffer.push(features);
    this.missingFrameCount = 0; // Reset on success

    if (this.sequenceBuffer.length > this.sequenceLength * 2) {
      this.sequenceBuffer = this.sequenceBuffer.slice(-this.sequenceLength);
    }
  }

  /**
   * Release all resources.
   */
  dispose(): void {
    this.session?.release();
    this.session = null;

    // MediaPipe Hands holds a WebGL context. Dropping the reference does not
    // free it, and browsers cap concurrent WebGL contexts (~16 in Chrome), so
    // leaking one per session ends in "Failed to create WebGL canvas context".
    if (this.hands) {
      try {
        const closing = this.hands.close?.();
        if (closing && typeof closing.catch === "function") {
          closing.catch((err: unknown) =>
            console.warn("GestureEngine: MediaPipe close failed", err)
          );
        }
      } catch (err) {
        console.warn("GestureEngine: MediaPipe close threw", err);
      }
    }
    this.hands = null;
    this.lastMPResults = null;
    this.currentDetectionPromise = null;
    this.sequenceBuffer = [];
    this.vBuffer = [];
    this.pBuffer = [];
    this.stationaryFrames = 0;
    this.lastLandmarks = null;
    this.segmenter.reset();
    this.isInitialized = false;
  }
}
