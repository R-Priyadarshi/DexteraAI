/**
 * Per-hand recognition state.
 *
 * Recognising two hands is not a matter of running the model twice: each hand
 * needs its own temporal window, its own motion history, and its own
 * segmenter. Sharing any of those across hands interleaves two different
 * gestures into one buffer, and the model then classifies a sequence that
 * neither hand actually performed.
 *
 * Tracks are keyed by handedness rather than by detection order, because
 * MediaPipe's array order is not stable between frames — indexing by position
 * swaps the two hands' histories whenever the order changes.
 */

import { type Landmark, type SpatialIntent } from "./gesture-engine";
import { GestureSegmenter, type SegmenterConfig } from "./gesture-segmenter";

export type Handedness = "left" | "right" | "unknown";

/** Recognition output for a single hand. */
export interface HandResult {
    handedness: Handedness;
    gestureName: string;
    gestureId: number;
    confidence: number;
    rejected: boolean;
    landmarks: Landmark[];
    velocity: { x: number; y: number; z: number };
    spatialIntent: SpatialIntent;
    phase: import("./gesture-segmenter").GesturePhase;
    heldMs: number;
    segmentId: number;
}

function dist3d(a: Landmark, b: Landmark): number {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = a.z - b.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** Frames of smoothing applied to velocity and wrist position. */
const SMOOTHING_FRAMES = 5;

/** Frames a hand may go missing before its buffers are discarded. */
const MISSING_TOLERANCE = 10;

export class HandTrack {
    readonly handedness: Handedness;
    readonly segmenter: GestureSegmenter;

    /** Feature vectors for the temporal window, oldest first. */
    private buffer: Float32Array[] = [];

    private lastLandmarks: Landmark[] | null = null;
    private lastTimestamp = 0;
    private velocitySamples: { x: number; y: number }[] = [];
    private wristSamples: { x: number; y: number }[] = [];
    private stationaryFrames = 0;
    private missingFrames = 0;

    constructor(handedness: Handedness, segmenterConfig?: Partial<SegmenterConfig>) {
        this.handedness = handedness;
        this.segmenter = new GestureSegmenter(segmenterConfig);
    }

    /** Append a feature vector, trimming the buffer to twice the window. */
    push(features: Float32Array, sequenceLength: number): void {
        this.buffer.push(features);
        this.missingFrames = 0;
        if (this.buffer.length > sequenceLength * 2) {
            this.buffer = this.buffer.slice(-sequenceLength);
        }
    }

    isReady(sequenceLength: number): boolean {
        return this.buffer.length >= sequenceLength;
    }

    /** Flatten the most recent `sequenceLength` frames into one tensor buffer. */
    window(sequenceLength: number, featureDim: number): Float32Array {
        const out = new Float32Array(sequenceLength * featureDim);
        const recent = this.buffer.slice(-sequenceLength);
        for (let i = 0; i < recent.length; i++) {
            out.set(recent[i], i * featureDim);
        }
        return out;
    }

    /**
     * Note that this hand was not detected this frame.
     *
     * Returns true once the loss is persistent enough that the track should be
     * considered gone. Brief losses are tolerated: MediaPipe drops a hand for a
     * frame or two on rotation and occlusion, and discarding the window each
     * time would mean the temporal buffer never fills.
     */
    markMissing(): boolean {
        this.missingFrames++;
        if (this.missingFrames > MISSING_TOLERANCE) {
            this.buffer = [];
            // Tracking is lost, so we cannot claim the gesture ended — only
            // that we no longer know what it is.
            this.segmenter.reset();
            this.lastLandmarks = null;
            this.velocitySamples = [];
            this.wristSamples = [];
            this.stationaryFrames = 0;
            return true;
        }
        return false;
    }

    /**
     * Update motion state and derive a spatial intent.
     *
     * Velocity is smoothed over several frames: raw frame-to-frame deltas are
     * dominated by landmark jitter, which produces phantom swipes from a
     * stationary hand.
     */
    motion(
        landmarks: Landmark[],
        now: number
    ): { velocity: { x: number; y: number; z: number }; spatialIntent: SpatialIntent } {
        const velocity = { x: 0, y: 0, z: 0 };

        if (this.lastLandmarks && this.lastTimestamp > 0) {
            const dt = (now - this.lastTimestamp) / 1000;
            if (dt > 0) {
                // X is negated so motion is expressed in the user's frame of
                // reference rather than the mirrored camera's.
                const rawX = (this.lastLandmarks[0].x - landmarks[0].x) / dt;
                const rawY = (landmarks[0].y - this.lastLandmarks[0].y) / dt;

                this.velocitySamples.push({ x: rawX, y: rawY });
                if (this.velocitySamples.length > SMOOTHING_FRAMES) {
                    this.velocitySamples.shift();
                }
                const n = this.velocitySamples.length;
                velocity.x = this.velocitySamples.reduce((s, v) => s + v.x, 0) / n;
                velocity.y = this.velocitySamples.reduce((s, v) => s + v.y, 0) / n;
            }
        }
        this.lastLandmarks = landmarks;
        this.lastTimestamp = now;

        this.wristSamples.push({ x: landmarks[0].x, y: landmarks[0].y });
        if (this.wristSamples.length > SMOOTHING_FRAMES) this.wristSamples.shift();

        const stationary = Math.abs(velocity.x) < 0.1 && Math.abs(velocity.y) < 0.1;
        this.stationaryFrames = stationary ? this.stationaryFrames + 1 : 0;

        let spatialIntent: SpatialIntent = "none";

        // Pinch is only meaningful once the hand has settled: thumb and index
        // pass close together during almost any travel, so an ungated distance
        // test fires constantly while the hand is moving.
        if (this.stationaryFrames > 10) {
            const pinch = dist3d(landmarks[4], landmarks[8]);
            if (pinch < 0.06) spatialIntent = "pinch_close";
            else if (pinch < 0.12) spatialIntent = "pinch_open";
        }

        // A swipe needs both speed and net displacement. Speed alone fires on a
        // hand that shakes in place; displacement alone fires on a slow drift.
        const first = this.wristSamples[0];
        const last = this.wristSamples[this.wristSamples.length - 1];
        const displacement = last.x - first.x;
        const horizontal = Math.abs(velocity.x) > Math.abs(velocity.y) * 2.5;

        if (horizontal && Math.abs(displacement) > 0.03) {
            if (velocity.x < -0.85) spatialIntent = "hyper_left";
            else if (velocity.x > 0.85) spatialIntent = "hyper_right";
            else if (velocity.x < -0.25) spatialIntent = "swipe_left";
            else if (velocity.x > 0.25) spatialIntent = "swipe_right";
        }

        return { velocity, spatialIntent };
    }

    reset(): void {
        this.buffer = [];
        this.segmenter.reset();
        this.lastLandmarks = null;
        this.lastTimestamp = 0;
        this.velocitySamples = [];
        this.wristSamples = [];
        this.stationaryFrames = 0;
        this.missingFrames = 0;
    }
}
