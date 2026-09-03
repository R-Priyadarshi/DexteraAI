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
import { type GesturePhase, type SegmenterConfig } from "./gesture-segmenter";
import { buildPrototype, matchPrototypes, type Prototype } from "./few-shot";
import { HandTrack, type Handedness, type HandResult } from "./hand-track";

export type { HandResult, Handedness } from "./hand-track";

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
  /**
   * Every hand detected this frame, each with independent recognition state.
   * The top-level fields above mirror `hands[0]`, the primary hand, so existing
   * single-hand consumers keep working unchanged.
   */
  hands: HandResult[];
  /**
   * Set when two hands are detected and both produced an accepted label, in a
   * stable left/right order. Null whenever fewer than two hands are present.
   */
  combo: TwoHandedCombo | null;
}

/**
 * A two-handed pose, expressed as an ordered pair of single-hand labels.
 *
 * The shipped models are trained on one hand, so a genuine two-hand classifier
 * would need two-hand training data that neither HaGRID nor the ASL alphabet
 * set provides. Composing the pair from two independent single-hand
 * predictions is the honest alternative: it is not a two-handed *model*, but it
 * does give a real two-handed command surface — and squares the vocabulary,
 * since 18 labels yield 324 ordered pairs.
 */
export interface TwoHandedCombo {
  left: string;
  right: string;
  /** `left+right`, for use as a binding key. */
  id: string;
  /** The weaker of the two hands' confidences — the pair is only as good as that. */
  confidence: number;
  /** Distance between the two wrists, normalised. Drives scale gestures. */
  separation: number;
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

/** Minimum prototype-match confidence for a taught gesture to be reported. */
const CUSTOM_MATCH_THRESHOLD = 0.8;

/** Per-finger curl ratios occupy indices 78-82 of the 86-dim feature vector. */
const CURL_FEATURE_START = 78;
const CURL_FEATURE_END = 82;
const CURL_WEIGHT = 2.0;

/**
 * Weight on derived features (angles, distances, curls) over raw coordinates.
 *
 * Square root of two, because the previous matcher applied a factor of 2 to
 * *squared* differences inside the distance computation. Folding the weighting
 * into the vectors themselves means it must be applied to the values, and
 * sqrt(2) squared is 2 — so the metric is unchanged while prototypes can now
 * be built in the same space they are matched in.
 */
const DERIVED_FEATURE_WEIGHT = Math.SQRT2;
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
  private isInitialized = false;

  /** Recognition state per hand, keyed by handedness. */
  private tracks = new Map<Handedness, HandTrack>();

  /** Hands MediaPipe is configured to detect. */
  private maxHands = 1;
  private labels: string[] = [...DEFAULT_GESTURE_LABELS];
  private sequenceLength: number = DEFAULT_SEQUENCE_LENGTH;

  /**
   * Calibration from the loaded bundle. The defaults are deliberately inert
   * (T=1 leaves the softmax untouched) so an uncalibrated bundle behaves
   * exactly as it did before, just with a conservative generic threshold.
   */
  private temperature = 1.0;
  private rejectionThreshold = 0.6;

  private segmenterConfig: Partial<SegmenterConfig> = {};

  /** Tune onset/offset behaviour, e.g. for a latency-sensitive binding. */
  configureSegmenter(config: Partial<SegmenterConfig>): void {
    this.segmenterConfig = { ...this.segmenterConfig, ...config };
    for (const track of this.tracks.values()) track.segmenter.configure(config);
  }

  getSegmenterConfig(): SegmenterConfig {
    return this.track("unknown").segmenter.getConfig();
  }

  /**
   * Detect one hand or two.
   *
   * Two-handed tracking roughly doubles landmark-detection cost, so it stays
   * opt-in rather than being the default on every device.
   */
  async setMaxHands(count: 1 | 2): Promise<void> {
    if (count === this.maxHands) return;
    this.maxHands = count;
    if (this.hands) {
      this.hands.setOptions({
        maxNumHands: count,
        modelComplexity: 1,
        minDetectionConfidence: 0.3,
        minTrackingConfidence: 0.2,
      });
    }
    // State for a hand that is no longer tracked would otherwise persist and
    // resurface stale on the next switch back.
    this.tracks.clear();
  }

  getMaxHands(): number {
    return this.maxHands;
  }

  /** Track for a hand, created on first sight. */
  private track(handedness: Handedness): HandTrack {
    let t = this.tracks.get(handedness);
    if (!t) {
      t = new HandTrack(handedness, this.segmenterConfig);
      this.tracks.set(handedness, t);
    }
    return t;
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

      // Threaded WASM needs SharedArrayBuffer, which needs the page to be
      // cross-origin isolated (COOP + COEP). Those headers are set in
      // `next.config.mjs` for the dev server, but `output: "export"` drops
      // them — a static host has to send them itself, and many do not. Asking
      // for threads that cannot be created just produces a runtime warning and
      // a silent fallback, so the request is matched to what the page can
      // actually support.
      const isolated =
        typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
      ort.env.wasm.numThreads = isolated
        ? Math.min(4, navigator.hardwareConcurrency || 4)
        : 1;
      ort.env.wasm.proxy = false;

      if (!isolated) {
        console.info(
          "GestureEngine: page is not cross-origin isolated, so ONNX runs " +
            "single-threaded. Serve with Cross-Origin-Opener-Policy: same-origin " +
            "and Cross-Origin-Embedder-Policy: require-corp for multithreaded WASM."
        );
      }

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
      maxNumHands: this.maxHands,
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

  /**
   * Process a video frame and return gesture results.
   * All processing happens on-device.
   */
  async processFrame(
    video: HTMLVideoElement
  ): Promise<GestureResult | null> {
    if (!this.isInitialized || !this.hands) return null;

    const t0 = performance.now();

    // 1. Landmark detection. MediaPipe's callback API is wrapped in a promise
    //    so the caller can await one frame at a time.
    const detectionWait = new Promise<void>((resolve, reject) => {
      this.currentDetectionPromise = { resolve, reject };
      this.hands!.send({ image: video }).catch(reject);
    });
    await detectionWait;

    const mp = this.lastMPResults;
    const detected: Landmark[][] = mp?.multiHandLandmarks ?? [];

    if (detected.length === 0) {
      for (const track of this.tracks.values()) track.markMissing();
      return this.emptyResult(performance.now() - t0);
    }

    const now = performance.now();
    const results: HandResult[] = [];
    const seen = new Set<Handedness>();

    for (let i = 0; i < detected.length; i++) {
      const landmarks = detected[i];
      if (!landmarks || landmarks.length < 21) continue;

      // MediaPipe reports handedness from the camera's point of view, which is
      // mirrored relative to the user. Keeping its label as-is would mean the
      // console's "left hand" is the user's right.
      const raw = mp.multiHandedness?.[i]?.label?.toLowerCase();
      const handedness: Handedness =
        raw === "left" ? "right" : raw === "right" ? "left" : "unknown";

      seen.add(handedness);
      const track = this.track(handedness);

      track.push(extractFeatures(landmarks), this.sequenceLength);

      let gestureId = -1;
      let confidence = 0;
      let rejected = true;

      if (this.session && track.isReady(this.sequenceLength)) {
        const classification = await this.classify(track);
        if (classification) {
          gestureId = classification.gestureId;
          confidence = classification.confidence;
          rejected = classification.rejected;
        }
      }

      let gestureName =
        gestureId >= 0 ? (this.labels[gestureId] ?? "unknown") : "unknown";

      // Custom gestures fill the gap where the trained model abstains; they
      // never outrank a confident prediction from it.
      if (rejected) {
        const custom = this.matchCustomGesture(landmarks);
        if (custom && custom.confidence > confidence) {
          gestureId = CUSTOM_GESTURE_ID;
          gestureName = custom.name;
          confidence = custom.confidence;
          rejected = false;
        }
      }

      const { velocity, spatialIntent } = track.motion(landmarks, now);
      const segment = track.segmenter.push(
        { gestureName, gestureId, confidence, rejected },
        now
      );

      results.push({
        handedness,
        gestureName,
        gestureId,
        confidence,
        rejected,
        landmarks,
        velocity,
        spatialIntent,
        phase: segment.phase,
        heldMs: segment.heldMs,
        segmentId: segment.segmentId,
      });
    }

    // Hands that exist as tracks but were not detected this frame.
    for (const [handedness, track] of this.tracks) {
      if (!seen.has(handedness)) track.markMissing();
    }

    if (results.length === 0) {
      return this.emptyResult(performance.now() - t0);
    }

    // The primary hand is the most confident accepted one, falling back to the
    // first detected. Using detection order alone would let the top-level
    // fields flicker between hands as MediaPipe reorders them.
    const primary =
      results.find((r) => !r.rejected) ??
      results.reduce((a, b) => (b.confidence > a.confidence ? b : a));

    return {
      gestureName: primary.gestureName,
      gestureId: primary.gestureId,
      confidence: primary.confidence,
      landmarks: primary.landmarks,
      handedness: primary.handedness,
      inferenceTimeMs: performance.now() - t0,
      velocity: primary.velocity,
      spatialIntent: primary.spatialIntent,
      rejected: primary.rejected,
      phase: primary.phase,
      heldMs: primary.heldMs,
      segmentId: primary.segmentId,
      hands: results,
      combo: this.buildCombo(results),
    };
  }

  /**
   * Compose a two-handed pose from two accepted single-hand predictions.
   *
   * Returns null unless both hands are present and both were accepted — a pair
   * built on a rejected half is not a two-handed gesture, it is one gesture and
   * some noise.
   */
  private buildCombo(results: HandResult[]): TwoHandedCombo | null {
    if (results.length < 2) return null;

    const left = results.find((r) => r.handedness === "left");
    const right = results.find((r) => r.handedness === "right");
    if (!left || !right || left.rejected || right.rejected) return null;

    const dx = left.landmarks[0].x - right.landmarks[0].x;
    const dy = left.landmarks[0].y - right.landmarks[0].y;

    return {
      left: left.gestureName,
      right: right.gestureName,
      id: `${left.gestureName}+${right.gestureName}`,
      confidence: Math.min(left.confidence, right.confidence),
      separation: Math.sqrt(dx * dx + dy * dy),
    };
  }

  /** Result for a frame in which no hand was recognised. */
  private emptyResult(elapsedMs: number): GestureResult {
    return {
      gestureName: "no hand",
      gestureId: -1,
      confidence: 0,
      landmarks: null,
      handedness: "unknown",
      inferenceTimeMs: elapsedMs,
      velocity: { x: 0, y: 0, z: 0 },
      spatialIntent: "none",
      rejected: true,
      phase: "idle",
      heldMs: 0,
      segmentId: 0,
      hands: [],
      combo: null,
    };
  }

  /**
   * Run ONNX inference on buffered features.
   */
  private async classify(track: HandTrack): Promise<{
    gestureId: number;
    confidence: number;
    rejected: boolean;
  } | null> {
    if (!this.session) return null;

    if (!track.isReady(this.sequenceLength)) {
      // Window not yet full: nothing to classify, and no claim to make.
      return { gestureId: -1, confidence: 0, rejected: true };
    }

    const inputData = track.window(this.sequenceLength, FEATURE_DIM);
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
   * Match against user-taught gestures using class prototypes.
   *
   * Each taught gesture is reduced to a mean vector plus the spread of its own
   * samples, and a query is scored in units of that spread. That makes scores
   * comparable across gestures with genuinely different tightness — a fist
   * whose samples cluster hard and a wave whose samples spread wide can no
   * longer share one absolute distance threshold, which previously
   * over-rejected the first and over-accepted the second.
   */
  private matchCustomGesture(
    currentLandmarks: Landmark[]
  ): { name: string; confidence: number } | null {
    const prototypes = this.prototypes();
    if (prototypes.length === 0) return null;

    const query = this.weightFeatures(extractFeatures(currentLandmarks));
    const match = matchPrototypes(query, prototypes);

    return match && match.confidence > CUSTOM_MATCH_THRESHOLD
      ? { name: match.name, confidence: match.confidence }
      : null;
  }

  /**
   * Prototypes for the taught gestures, rebuilt only when the store changes.
   *
   * Deriving these per frame meant a user with ten taught gestures paid for
   * ~400 feature extractions every frame to recompute values that never
   * change. The signature covers gesture ids and sample counts, so teaching,
   * deleting or re-recording a gesture invalidates the cache.
   */
  private prototypes(): Prototype[] {
    const gestures = gestureStore.getGestures();
    const signature = gestures.map((g) => `${g.id}:${g.samples.length}`).join("|");

    if (signature === this.prototypeSignature) return this.prototypeCache;

    this.prototypeCache = gestures
      .map((g) =>
        buildPrototype(
          g.id,
          g.name,
          g.samples.map((sample) => this.weightFeatures(extractFeatures(sample)))
        )
      )
      .filter((p): p is Prototype => p !== null);
    this.prototypeSignature = signature;

    return this.prototypeCache;
  }

  private prototypeCache: Prototype[] = [];
  private prototypeSignature = "";

  /**
   * Emphasise the per-finger curl ratios before comparing.
   *
   * Curl carries most of what distinguishes one hand shape from another, while
   * raw coordinates carry a lot of position and scale that a match should not
   * depend on. Scaling those five dimensions up biases the distance toward
   * shape.
   */
  private weightFeatures(features: Float32Array): Float32Array {
    const out = new Float32Array(features.length);
    // Derived features (angles, distances, curls) discriminate shape better
    // than the raw coordinates that precede them, so they count for more. The
    // weighting is folded in here rather than applied at comparison time, so
    // prototypes are built in the same space they are matched in.
    for (let i = 0; i < features.length; i++) {
      out[i] = i >= 63 ? features[i] * DERIVED_FEATURE_WEIGHT : features[i];
    }
    for (let i = CURL_FEATURE_START; i <= CURL_FEATURE_END; i++) {
      out[i] *= CURL_WEIGHT;
    }
    return out;
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
    this.tracks.clear();
    this.isInitialized = false;
  }
}
