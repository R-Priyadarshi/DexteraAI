import { beforeEach, describe, expect, it, vi } from "vitest";

import { calibrator } from "../calibrator";
import type { Landmark } from "../gesture-engine";

/** 21 landmarks at a fixed position — a perfectly still hand. */
function stillHand(): Landmark[] {
    return Array.from({ length: 21 }, (_, i) => ({ x: 0.5, y: 0.5, z: i * 0.001 }));
}

/** Drive the calibrator's clock so the intervals under test are exact. */
function recordFrames(intervalsMs: number[], hand: () => Landmark[] | null): void {
    let t = 1000;
    const spy = vi.spyOn(performance, "now");
    spy.mockImplementation(() => t);
    calibrator.record(hand());
    for (const dt of intervalsMs) {
        t += dt;
        calibrator.record(hand());
    }
    spy.mockRestore();
}

describe("Calibrator latency jitter", () => {
    beforeEach(() => {
        calibrator.reset();
    });

    it("reports no jitter before it has enough frames", () => {
        // Two frames is one interval; a deviation needs more than that.
        recordFrames([33], stillHand);
        expect(calibrator.getMetrics().latencyJitter).toBe(0);
    });

    it("reports ~0 for a metronomic frame loop", () => {
        // A steady 30fps camera: every interval identical, so no spread.
        recordFrames(Array(20).fill(33), stillHand);
        expect(calibrator.getMetrics().latencyJitter).toBeCloseTo(0, 6);
    });

    it("reports the standard deviation of the intervals, not their mean", () => {
        // Alternating 20ms / 60ms: mean 40, deviation 20. A mean-based metric
        // would call this a healthy 25fps and miss the stutter entirely, which
        // is the case this field exists to catch.
        const intervals: number[] = [];
        for (let i = 0; i < 20; i++) intervals.push(i % 2 === 0 ? 20 : 60);
        recordFrames(intervals, stillHand);
        expect(calibrator.getMetrics().latencyJitter).toBeCloseTo(20, 5);
    });

    it("keeps counting frames where the hand was lost", () => {
        // A dropped detection is still a frame; excluding it would make a
        // stuttering feed look smooth.
        recordFrames(Array(20).fill(33), () => null);
        const m = calibrator.getMetrics();
        expect(m.latencyJitter).toBeCloseTo(0, 6);
        expect(m.lighting).toBe(0);
    });
});
