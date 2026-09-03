import { describe, expect, it } from "vitest";

import { AirCanvas, DEFAULT_AIR_CANVAS_CONFIG, type Landmark } from "../air-canvas";

/**
 * Build a hand where only the four landmarks that matter are controlled.
 *
 * Wrist and middle knuckle sit 0.2 apart, so that span is the unit every
 * threshold is expressed in: `pinchGap` and the tip position below are given in
 * those units and converted here.
 */
function hand({ pinchGap, tip = { x: 0.5, y: 0.5 } }: { pinchGap: number; tip?: { x: number; y: number } }): Landmark[] {
    const HAND_SCALE = 0.2;
    const points: Landmark[] = Array.from({ length: 21 }, () => ({ x: 0, y: 0 }));
    points[0] = { x: 0.5, y: 0.9 }; // wrist
    points[9] = { x: 0.5, y: 0.9 - HAND_SCALE }; // middle MCP
    points[8] = { ...tip }; // index tip
    points[4] = { x: tip.x + pinchGap * HAND_SCALE, y: tip.y }; // thumb tip
    return points;
}

/** Draw a straight run of points long enough to clear the minimum-length gate. */
function drawLine(canvas: AirCanvas, steps = 20, pinchGap = 0.2): void {
    for (let i = 0; i < steps; i++) {
        canvas.feed(hand({ pinchGap, tip: { x: 0.2 + (i / steps) * 0.6, y: 0.5 } }));
    }
}

describe("AirCanvas", () => {
    it("refuses thresholds that would make the pen chatter", () => {
        // Without a gap between them the pen flickers on the exact value a
        // hovering hand sits at, and one stroke arrives as a dozen fragments.
        expect(() => new AirCanvas({ pinchDown: 0.6, pinchUp: 0.5 })).toThrow(/must exceed/);
        expect(() => new AirCanvas({ pinchDown: 0.5, pinchUp: 0.5 })).toThrow(/must exceed/);
    });

    it("starts a stroke when the fingers close", () => {
        const canvas = new AirCanvas();
        expect(canvas.feed(hand({ pinchGap: 1.2 }))).toBeNull();
        expect(canvas.isDrawing).toBe(false);

        expect(canvas.feed(hand({ pinchGap: 0.2 }))?.type).toBe("start");
        expect(canvas.isDrawing).toBe(true);
    });

    it("keeps drawing through the gap between the two thresholds", () => {
        // The point of the hysteresis: a pinch that drifts to 0.6 — above
        // pinchDown, below pinchUp — is still one continuous stroke.
        const canvas = new AirCanvas();
        canvas.feed(hand({ pinchGap: 0.2 }));

        const between = (DEFAULT_AIR_CANVAS_CONFIG.pinchDown + DEFAULT_AIR_CANVAS_CONFIG.pinchUp) / 2;
        canvas.feed(hand({ pinchGap: between, tip: { x: 0.6, y: 0.5 } }));
        expect(canvas.isDrawing).toBe(true);
    });

    it("ends the stroke when the fingers open, and hands back the path", () => {
        const canvas = new AirCanvas();
        drawLine(canvas);

        const event = canvas.feed(hand({ pinchGap: 1.5 }));
        expect(event?.type).toBe("end");
        if (event?.type !== "end") throw new Error("expected an end event");
        expect(event.points.length).toBeGreaterThan(5);
        expect(canvas.isDrawing).toBe(false);
    });

    it("discards a stroke too short to have been meant", () => {
        // A pinch to grab something, not to draw.
        const canvas = new AirCanvas();
        canvas.feed(hand({ pinchGap: 0.2 }));
        canvas.feed(hand({ pinchGap: 0.2, tip: { x: 0.51, y: 0.5 } }));

        const event = canvas.feed(hand({ pinchGap: 1.5 }));
        expect(event?.type).toBe("discard");
    });

    it("mirrors x, because the preview the person is drawing into is mirrored", () => {
        // Without this every asymmetric shape is captured as its reflection.
        // Mirroring is not a rotation, so normalisation cannot undo it and the
        // stroke simply never matches.
        const canvas = new AirCanvas();
        canvas.feed(hand({ pinchGap: 0.2, tip: { x: 0.2, y: 0.5 } }));
        expect(canvas.trail[0].x).toBeCloseTo(0.8, 6);

        const unmirrored = new AirCanvas({ mirrorX: false });
        unmirrored.feed(hand({ pinchGap: 0.2, tip: { x: 0.2, y: 0.5 } }));
        expect(unmirrored.trail[0].x).toBeCloseTo(0.2, 6);
    });

    it("ends rather than discards when the hand leaves the frame mid-stroke", () => {
        // Reaching past the camera's edge is a normal way to finish a large
        // shape. Throwing the stroke away there reads as the app dropping it.
        const canvas = new AirCanvas();
        drawLine(canvas);

        const event = canvas.feed(null);
        expect(event?.type).toBe("end");
    });

    it("ignores a frame with no hand when not drawing", () => {
        expect(new AirCanvas().feed(null)).toBeNull();
    });

    it("does not record points the hand has not actually moved between", () => {
        // MediaPipe reports a still finger with sub-pixel jitter every frame;
        // recording all of it buries the shape in duplicates.
        const canvas = new AirCanvas();
        canvas.feed(hand({ pinchGap: 0.2, tip: { x: 0.5, y: 0.5 } }));
        for (let i = 0; i < 30; i++) {
            canvas.feed(hand({ pinchGap: 0.2, tip: { x: 0.5, y: 0.5 } }));
        }
        expect(canvas.trail).toHaveLength(1);
    });

    it("caps a runaway stroke instead of growing without bound", () => {
        const canvas = new AirCanvas({ maxPoints: 10 });
        for (let i = 0; i < 200; i++) {
            canvas.feed(hand({ pinchGap: 0.2, tip: { x: (i % 50) / 50, y: 0.5 } }));
        }
        expect(canvas.trail.length).toBeLessThanOrEqual(10);
    });

    it("survives a degenerate hand without emitting NaN", () => {
        // Every landmark at the same point makes the hand scale zero, and every
        // ratio computed from it Infinity or NaN.
        const canvas = new AirCanvas();
        const collapsed: Landmark[] = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5 }));
        expect(canvas.feed(collapsed)).toBeNull();
        expect(canvas.isDrawing).toBe(false);
    });

    it("forgets a stroke in progress on reset", () => {
        const canvas = new AirCanvas();
        drawLine(canvas);
        expect(canvas.isDrawing).toBe(true);

        canvas.reset();
        expect(canvas.isDrawing).toBe(false);
        expect(canvas.trail).toHaveLength(0);
    });
});
