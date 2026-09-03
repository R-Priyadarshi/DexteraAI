import { beforeEach, describe, expect, it, vi } from "vitest";

import { type GestureResult } from "../gesture-engine";
import { intentRefinery } from "../intent-refinery";
import { type VoiceIntent } from "../voice-engine";

vi.mock("../haptic-engine", () => ({ hapticEngine: { pulse: vi.fn() } }));

/** A gesture onset, which is the only phase the refinery buffers. */
function onset(gestureName: string): GestureResult {
    return { gestureName, phase: "onset", rejected: false } as unknown as GestureResult;
}

function held(gestureName: string): GestureResult {
    return { gestureName, phase: "hold", rejected: false } as unknown as GestureResult;
}

/** Register a fusion whose only job is to count how often it ran. */
function countingFusion(gestureName: string, voiceIntent: VoiceIntent) {
    const calls = { n: 0 };
    intentRefinery.register({
        id: `test_${gestureName}_${voiceIntent}`,
        name: "test",
        gestureName,
        voiceIntent,
        feedbackType: "light",
        execute: () => {
            calls.n += 1;
        },
    });
    return calls;
}

describe("IntentRefinery", () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
        intentRefinery.reset();
    });

    it("does not fire on a gesture alone", () => {
        const calls = countingFusion("palm", "confirm");
        for (let i = 0; i < 5; i++) {
            expect(intentRefinery.process(onset("palm"), null)).toBeNull();
            vi.advanceTimersByTime(33);
        }
        expect(calls.n).toBe(0);
    });

    it("does not fire on a spoken intent alone", () => {
        const calls = countingFusion("palm", "confirm");
        for (let i = 0; i < 5; i++) {
            expect(intentRefinery.process(onset("fist"), "confirm")).toBeNull();
            vi.advanceTimersByTime(33);
        }
        expect(calls.n).toBe(0);
    });

    it("fires once when both agree, in either order", () => {
        const calls = countingFusion("palm", "confirm");

        // Voice first, gesture second.
        intentRefinery.process(onset("fist"), "confirm");
        vi.advanceTimersByTime(300);
        expect(intentRefinery.process(onset("palm"), null)).not.toBeNull();
        expect(calls.n).toBe(1);

        // Gesture first, voice second — past the cooldown.
        vi.advanceTimersByTime(2000);
        intentRefinery.reset();
        const calls2 = countingFusion("palm", "confirm");
        intentRefinery.process(onset("palm"), null);
        vi.advanceTimersByTime(300);
        expect(intentRefinery.process(onset("fist"), "confirm")).not.toBeNull();
        expect(calls2.n).toBe(1);
    });

    it("executes the action exactly once per fusion", () => {
        // `process` runs the action itself. The dashboard used to run it a
        // second time, so an emergency halt locked twice and anything
        // toggle-shaped cancelled itself.
        const calls = countingFusion("palm", "confirm");
        intentRefinery.process(onset("palm"), "confirm");
        expect(calls.n).toBe(1);
    });

    it("will not fuse across the window", () => {
        const calls = countingFusion("palm", "confirm");
        intentRefinery.process(onset("fist"), "confirm");
        // Past FUSION_WINDOW_MS, so the spoken intent is stale.
        vi.advanceTimersByTime(2100);
        expect(intentRefinery.process(onset("palm"), null)).toBeNull();
        expect(calls.n).toBe(0);
    });

    it("holds off during the cooldown", () => {
        const calls = countingFusion("palm", "confirm");
        intentRefinery.process(onset("palm"), "confirm");
        expect(calls.n).toBe(1);

        vi.advanceTimersByTime(500); // still inside COOLDOWN_MS
        intentRefinery.process(onset("palm"), "confirm");
        expect(calls.n).toBe(1);
    });

    it("does not re-fire from a word held across many frames", () => {
        // The regression this file exists for. The caller keeps a recognised
        // intent set for about two seconds so it can be displayed, and
        // `process` runs every frame — so a single spoken word used to enter
        // the buffer ~60 times. One was consumed on the match and the rest sat
        // there, ready to fuse again the moment the cooldown lapsed.
        const calls = countingFusion("palm", "confirm");

        intentRefinery.process(onset("palm"), "confirm");
        expect(calls.n).toBe(1);

        // The same utterance is still being reported, frame after frame.
        for (let i = 0; i < 60; i++) {
            intentRefinery.process(held("palm"), "confirm");
            vi.advanceTimersByTime(33);
        }

        // Cooldown has long since lapsed; a fresh gesture must not find a
        // leftover copy of that one word.
        const before = calls.n;
        intentRefinery.process(onset("palm"), null);
        expect(calls.n).toBe(before);
    });

    it("ignores held frames, counting only gesture onsets", () => {
        const calls = countingFusion("palm", "confirm");
        for (let i = 0; i < 30; i++) {
            intentRefinery.process(held("palm"), null);
            vi.advanceTimersByTime(10);
        }
        intentRefinery.process(held("fist"), "confirm");
        expect(calls.n).toBe(0);
    });

    it("does not fuse a rejected gesture", () => {
        // Below the calibrated rejection threshold means "not recognised", and
        // an unrecognised pose must not help authorise an irreversible action.
        const calls = countingFusion("palm", "confirm");
        const rejected = { gestureName: "palm", phase: "onset", rejected: true } as unknown as GestureResult;
        intentRefinery.process(rejected, "confirm");
        expect(calls.n).toBe(0);
    });
});
