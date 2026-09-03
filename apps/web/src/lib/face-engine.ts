"use client";

/**
 * FaceEngine — non-manual markers from facial blendshapes.
 *
 * The second modality in `intent-refinery.ts` was speech, and speech excludes
 * people: anyone non-verbal, anyone in a shared or noisy room, anyone who would
 * rather not talk to their computer. For a product whose stated purpose is
 * accessible input, making the confirming signal available on the face as well
 * as the voice is the point, not a flourish.
 *
 * It also matches how the language actually works. ASL marks yes/no questions
 * with raised brows, wh-questions with lowered ones, and negation with the head
 * and mouth — grammar carried above the hands. `core/vision/holistic_detector.py`
 * already tracks the same brow, eye and mouth points on the Python side.
 *
 * No model is trained here. MediaPipe's FaceLandmarker emits 52 blendshape
 * coefficients directly, so a marker is a threshold over named coefficients —
 * the same geometric approach as the finger-curl ratios, and it needs no
 * training data, which matters because the corpus for this (WLASL) is
 * research-licensed and cannot ship.
 */

export type FacialMarker = "brow_raise" | "brow_furrow" | "mouth_open";

export interface FaceReading {
    marker: FacialMarker | null;
    /** Strength of the winning marker, 0-1. */
    intensity: number;
    /** Milliseconds spent in face detection for this frame. */
    detectMs: number;
}

/**
 * Blendshape coefficients that define each marker, and the level one has to
 * reach to count.
 *
 * The thresholds are deliberately high. A marker competes with speech to
 * authorise an irreversible action, so a resting face that drifts over the line
 * is worse than one that occasionally misses — a false positive here fires
 * something the user did not ask for, while a false negative just asks them to
 * do it again.
 */
import { asset } from "./base-path";

const MARKERS: Record<FacialMarker, { shapes: string[]; threshold: number }> = {
    // Yes/no question, and the natural "yes, do it" face.
    brow_raise: { shapes: ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"], threshold: 0.45 },
    // Wh-question, and the natural "no / wait" face.
    brow_furrow: { shapes: ["browDownLeft", "browDownRight"], threshold: 0.4 },
    // A deliberate, unambiguous third signal.
    mouth_open: { shapes: ["jawOpen"], threshold: 0.45 },
};

/** Where `sync-runtime.sh` puts the Tasks WASM and the face model. */
const WASM_PATH = asset("/onnx/tasks");
const MODEL_PATH = asset("/onnx/mediapipe/face_landmarker.task");

export class FaceEngine {
    private static instance: FaceEngine;
    private landmarker: unknown = null;
    private initializing: Promise<boolean> | null = null;
    private lastDetectMs = 0;
    /** Monotonically increasing, as the Tasks video API requires. */
    private frameIndex = 0;

    public static getInstance(): FaceEngine {
        if (!FaceEngine.instance) FaceEngine.instance = new FaceEngine();
        return FaceEngine.instance;
    }

    public get isReady(): boolean {
        return this.landmarker !== null;
    }

    public get detectMs(): number {
        return this.lastDetectMs;
    }

    /**
     * Load the landmarker. Safe to call repeatedly; returns whether it is usable.
     *
     * Failure is a supported outcome, not an exception: this runs alongside hand
     * detection, and a browser that cannot load a second model must keep the
     * gesture path working rather than take the page down with it.
     */
    public async init(): Promise<boolean> {
        if (this.landmarker) return true;
        if (this.initializing) return this.initializing;

        this.initializing = (async () => {
            try {
                const vision = await import("@mediapipe/tasks-vision");
                const fileset = await vision.FilesetResolver.forVisionTasks(WASM_PATH);
                this.landmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
                    baseOptions: { modelAssetPath: MODEL_PATH },
                    runningMode: "VIDEO",
                    numFaces: 1,
                    // The coefficients are the whole point; landmarks alone would
                    // mean re-deriving brow and mouth geometry by hand.
                    outputFaceBlendshapes: true,
                });
                return true;
            } catch (err) {
                console.warn("FaceEngine: unavailable, continuing without it", err);
                this.landmarker = null;
                return false;
            } finally {
                this.initializing = null;
            }
        })();

        return this.initializing;
    }

    /**
     * Read the strongest non-manual marker in this frame, if any.
     *
     * Returns null when the engine is not loaded or no face is visible, which
     * the caller must treat as "no second modality", never as a rejection.
     */
    public detect(video: HTMLVideoElement): FaceReading | null {
        if (!this.landmarker) return null;

        const started = performance.now();
        let result: { faceBlendshapes?: { categories: { categoryName: string; score: number }[] }[] };
        try {
            const lm = this.landmarker as {
                detectForVideo: (v: HTMLVideoElement, t: number) => typeof result;
            };
            // The Tasks video API rejects a timestamp it has already seen, and
            // two frames can land inside the same millisecond.
            result = lm.detectForVideo(video, performance.now() + this.frameIndex++);
        } catch {
            return null;
        }
        this.lastDetectMs = performance.now() - started;

        const shapes = result?.faceBlendshapes?.[0]?.categories;
        if (!shapes || shapes.length === 0) return null;

        const score = new Map(shapes.map((c) => [c.categoryName, c.score]));

        let best: FacialMarker | null = null;
        let bestIntensity = 0;
        for (const [marker, spec] of Object.entries(MARKERS) as [
            FacialMarker,
            (typeof MARKERS)[FacialMarker],
        ][]) {
            // Mean across the contributing shapes: a brow raise means both
            // brows, and averaging keeps one twitching eyebrow from passing.
            const mean =
                spec.shapes.reduce((sum, name) => sum + (score.get(name) ?? 0), 0) /
                spec.shapes.length;
            if (mean >= spec.threshold && mean > bestIntensity) {
                best = marker;
                bestIntensity = mean;
            }
        }

        return { marker: best, intensity: bestIntensity, detectMs: this.lastDetectMs };
    }

    /** Release the model. The page keeps working without it. */
    public close(): void {
        const lm = this.landmarker as { close?: () => void } | null;
        try {
            lm?.close?.();
        } catch {
            /* a failed teardown must not surface to the user */
        }
        this.landmarker = null;
        this.frameIndex = 0;
    }
}

export const faceEngine = FaceEngine.getInstance();

/** Exported for tests and for the settings UI that explains each marker. */
export const FACIAL_MARKERS = Object.keys(MARKERS) as FacialMarker[];
export const MARKER_SPECS = MARKERS;
