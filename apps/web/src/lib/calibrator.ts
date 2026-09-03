/**
 * Calibrator — Neural Calibration Engine.
 * 
 * Measures environmental "noise" (shakiness, lighting quality)
 * and dynamically calculates the optimal confidence thresholds 
 * for the GestureEngine.
 */

import { Landmark } from "./gesture-engine";

export interface CalibrationMetrics {
    stability: number;     // 0-1 (1 = perfectly still)
    lighting: number;      // 0-1 (1 = perfect landmarks)
    latencyJitter: number; // ms variance
}

/** Monotonic clock, falling back to Date.now outside the browser (tests). */
function now(): number {
    return typeof performance !== "undefined" ? performance.now() : Date.now();
}

export class Calibrator {
    private static instance: Calibrator;
    private history: Landmark[][] = [];
    private frameTimes: number[] = [];
    private MAX_HISTORY = 50;
    
    private constructor() {}

    public static getInstance(): Calibrator {
        if (!Calibrator.instance) {
            Calibrator.instance = new Calibrator();
        }
        return Calibrator.instance;
    }

    /**
     * Discard everything recorded so far.
     *
     * Calibration describes one continuous session — a camera, a room, a
     * person. Carrying it across a camera restart or a new session means
     * reporting stability for a hand that is no longer there, and it is what
     * made this singleton untestable.
     */
    public reset(): void {
        this.history = [];
        this.frameTimes = [];
    }

    /**
     * Record a frame for calibration.
     */
    public record(landmarks: Landmark[] | null) {
        if (!landmarks) {
            // Signal a "missing frame" drop in quality
            this.history.push([]); 
        } else {
            this.history.push(landmarks);
        }
        this.frameTimes.push(now());

        if (this.history.length > this.MAX_HISTORY) {
            this.history.shift();
        }
        if (this.frameTimes.length > this.MAX_HISTORY) {
            this.frameTimes.shift();
        }
    }

    /**
     * Calculate dynamic system metrics.
     */
    public getMetrics(): CalibrationMetrics {
        if (this.history.length < 10) {
            return { stability: 1, lighting: 1, latencyJitter: 0 };
        }

        // 1. Lighting / Detection Rate (Ratio of valid frames)
        const validFrames = this.history.filter(h => h.length > 0).length;
        const lighting = validFrames / this.history.length;

        // 2. Stability (Average variance of wrist/index landmarks)
        const stability = this.calculateStability();

        // 3. Latency jitter: how unevenly frames are arriving.
        return { stability, lighting, latencyJitter: this.calculateJitter() };
    }

    /**
     * Standard deviation of the interval between frames, in milliseconds.
     *
     * A steady 30fps loop gives intervals clustered around 33ms and a jitter
     * near zero. Contention — a busy tab, thermal throttling, a detector
     * occasionally taking two frames' worth of time — spreads them out, and
     * that spread is what makes recognition feel unreliable even while the
     * mean frame rate still looks fine. The mean alone hides it, which is why
     * this reports the deviation rather than the average.
     */
    private calculateJitter(): number {
        if (this.frameTimes.length < 3) return 0;

        const intervals: number[] = [];
        for (let i = 1; i < this.frameTimes.length; i++) {
            intervals.push(this.frameTimes[i] - this.frameTimes[i - 1]);
        }

        const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        const variance =
            intervals.reduce((acc, v) => acc + (v - mean) ** 2, 0) / intervals.length;
        return Math.sqrt(variance);
    }

    private calculateStability(): number {
        const validSequences = this.history.filter(h => h.length > 0);
        if (validSequences.length < 2) return 1.0;

        let totalVariance = 0;
        const landmarkIndices = [0, 5, 17]; // Wrist, Index MCP, Pinky MCP

        for (const idx of landmarkIndices) {
            const coords = validSequences.map(h => h[idx]);
            const avgX = coords.reduce((a, b) => a + b.x, 0) / coords.length;
            const avgY = coords.reduce((a, b) => a + b.y, 0) / coords.length;
            
            const variance = coords.reduce((acc, curr) => {
                return acc + Math.pow(curr.x - avgX, 2) + Math.pow(curr.y - avgY, 2);
            }, 0) / coords.length;
            
            totalVariance += variance;
        }

        // Normalize: higher variance = lower stability
        // Heuristic: 0.005 variance is "very shaky"
        const normalized = Math.max(0, 1 - (totalVariance / 0.005));
        return normalized;
    }

    /**
     * Suggest a confidence threshold based on metrics.
     * Lower quality environment -> Higher confidence requirement.
     */
    public getSuggestedThreshold(): number {
        const { stability, lighting } = this.getMetrics();
        
        // Base threshold
        let threshold = 0.85;

        // If unstable, increase threshold to avoid false positives
        if (stability < 0.8) threshold += 0.05;
        
        // If lighting is poor, increase threshold as model may be hallucinating
        if (lighting < 0.7) threshold += 0.05;

        return Math.min(0.98, threshold);
    }
}

export const calibrator = Calibrator.getInstance();
