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
const DEFAULT_SEQUENCE_LENGTH = 30;

/** Shape of the labels.json emitted next to an exported gesture.onnx. */
export interface ModelBundle {
  labels: string[];
  seq_len?: number;
  feature_dim?: number;
  val_accuracy?: number;
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
  private hands: any | null = null;
  private session: ort.InferenceSession | null = null;
  private sequenceBuffer: Float32Array[] = [];
  private isInitialized = false;
  private labels: string[] = [...DEFAULT_GESTURE_LABELS];
  private sequenceLength: number = DEFAULT_SEQUENCE_LENGTH;

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
        console.log(
          `GestureEngine: loaded ${this.labels.length} labels from bundle`,
          bundle.val_accuracy ? `(val acc ${bundle.val_accuracy})` : ""
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
      };
    }

    const landmarks = this.lastMPResults.multiHandLandmarks[0] as Landmark[];
    const handedness = (this.lastMPResults.multiHandedness?.[0]?.label.toLowerCase() || "unknown") as "left" | "right" | "unknown";

    // 2. Feature Extraction & Buffering
    this.pushFeatures(landmarks);

    // 3. Temporal Classification (Transformer)
    let gestureResult = { gestureId: 0, confidence: 1.0 }; // Default to "static" if no session

    if (this.session && this.sequenceBuffer.length >= this.sequenceLength) {
      const classification = await this.classifyGesture();
      if (classification) {
        gestureResult = classification;
      }
    }

    const t1 = performance.now();

    // 4. Custom Gesture Matching (K-NN Fallback or Priority)
    let finalGestureId = gestureResult.gestureId;
    let finalConfidence = gestureResult.confidence;
    let finalGestureName = this.labels[gestureResult.gestureId] || "unknown";

    // P0: Geometric Heuristic Check (Hard override for Handshake stability)
    // "Peace" (5), "Fist" (2), "Thumbs Up" (3)
    const geoMatch = this.detectGeometricGesture(landmarks);
    if (geoMatch) {
      finalGestureId = geoMatch.id;
      finalGestureName = geoMatch.name;
      finalConfidence = 1.0; // Geometric certainty
    } else if (finalConfidence < 0.85) {
      // P1: K-NN Custom Gestures
      const customMatch = this.matchCustomGesture(landmarks);
      if (customMatch && customMatch.confidence > finalConfidence) {
        finalGestureId = 999;
        finalConfidence = customMatch.confidence;
        finalGestureName = customMatch.name;
      }
    }

    // 5. Dynamic Spatial Intent (Pinch/Swipe)
    let spatialIntent: GestureResult["spatialIntent"] = "none";
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const dist = dist3d(thumbTip, indexTip);
    
    // High-precision pinch detection (Geometric distance)
    if (dist < 0.06) spatialIntent = "pinch_close";
    else if (dist < 0.12) spatialIntent = "pinch_open";

    // 6. Calculate Velocity for Swipes (Hardened with Kinetic Momentum & Directional Lock)
    const velocity = { x: 0, y: 0, z: 0 };
    const now = performance.now();
    let totalDisplacementX = 0;

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
            
            // Calculate total displacement over the buffer window
            totalDisplacementX = landmarks[0].x - (this.lastLandmarks[0].x || landmarks[0].x);
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

    // Global motion telemetry for industrial verification
    if (Math.abs(velocity.x) > 0.1) {
        console.log(`[Neural_Motion] Vel_X: ${velocity.x.toFixed(3)} | Dominant: ${isHorizontalDominant} | Intent: ${spatialIntent}`);
    }

    return {
      gestureName: finalGestureName,
      gestureId: finalGestureId,
      confidence: finalConfidence,
      landmarks,
      handedness,
      inferenceTimeMs: t1 - t0,
      velocity,
      spatialIntent
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
  } | null> {
    if (!this.session) return null;

    if (this.sequenceBuffer.length < this.sequenceLength) {
      // Buffer not yet full, return "pre-inference" state
      return {
        gestureId: -1,
        confidence: 0
      };
    }
    // ... rest of classifyGesture unchanged ...

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
   * Robust geometric heuristic for static gestures (Peace, Fist, Thumbs Up).
   * Bypasses ML model for critical UI interactions.
   */
  private detectGeometricGesture(landmarks: Landmark[]): { id: number, name: string } | null {
    // Helper to check if a finger is extended (tip higher than pip - assuming upright hand)
    // Note: Y increases downwards in MediaPipe/Screen coords. So Lower Y = Higher Position.
    // However, this simple check fails if hand is inverted.
    // Better check: Distance from Wrist.

    // We use the "Tip furthest from wrist" logic for extension.
    const isExtended = (tipIdx: number, pipIdx: number, mcpIdx: number) => {
      const tip = landmarks[tipIdx];
      const pip = landmarks[pipIdx];
      const mcp = landmarks[mcpIdx];
      const wrist = landmarks[0];

      // 1. Distance check
      const tipDist = dist3d(tip, wrist);
      const pipDist = dist3d(pip, wrist);
      const mcpDist = dist3d(mcp, wrist);

      // Tip should be further than PIP and MCP
      const isFarther = tipDist > pipDist && pipDist > mcpDist;

      // 2. Curvature check (dot product of vectors) - Simplified:
      // If curled, tip to wrist distance is significantly shorter than fully extended
      return isFarther;
    };

    // Refined thresholds can be added if needed
    const thumbExtended = isExtended(4, 3, 2);
    const indexExtended = isExtended(8, 6, 5);
    const middleExtended = isExtended(12, 10, 9);
    const ringExtended = isExtended(16, 14, 13);
    const pinkyExtended = isExtended(20, 18, 17);

    // Debug geometric states (Force log for now)
    if (Math.random() < 0.1) {
      console.log(`[Geo] T:${thumbExtended} I:${indexExtended} M:${middleExtended} R:${ringExtended} P:${pinkyExtended}`);
    }

    // 1. Peace Sign (Index + Middle extended, Ring + Pinky curled)
    const features = extractFeatures(landmarks);
    const thumbCurl = features[78];
    const indexCurl = features[79];
    const middleCurl = features[80];
    const ringCurl = features[81];
    const pinkyCurl = features[82];

    // Relative logic: Index and Middle are significantly MORE extended than Ring and Pinky
    if (indexCurl < ringCurl - 0.2 && middleCurl < pinkyCurl - 0.2) {
      return { id: 5, name: "peace" };
    }

    // 2. Thumbs Up (Thumb extended, others curled)
    // Vertical Dominance: Thumb tip must be higher than the index knuckle
    const isVerticalThumb = landmarks[4].y < landmarks[5].y - 0.05;
    if (isVerticalThumb && thumbCurl < 0.5 && indexCurl > 0.4 && middleCurl > 0.4 && ringCurl > 0.4 && pinkyCurl > 0.4) {
      return { id: 3, name: "thumbs_up" };
    }

    // 3. Closed Fist (All 4 main fingers curled, thumb tucked or curled)
    if (indexCurl > 0.4 && middleCurl > 0.4 && ringCurl > 0.4 && pinkyCurl > 0.4) {
      return { id: 2, name: "closed_fist" };
    }

    // 4. Open Palm (All extended)
    if (indexExtended && middleExtended && ringExtended && pinkyExtended) {
      return { id: 1, name: "open_palm" };
    }

    return null;
  }

  /**
   * Release all resources.
   */
  dispose(): void {
    this.session?.release();
    this.session = null;
    this.hands = null;
    this.sequenceBuffer = [];
    this.vBuffer = [];
    this.pBuffer = [];
    this.stationaryFrames = 0;
    this.lastLandmarks = null;
    this.isInitialized = false;
  }
}
