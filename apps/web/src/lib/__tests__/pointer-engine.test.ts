import { describe, expect, it } from "vitest";
import { PointerEngine } from "../pointer-engine";

const VIEWPORT = { width: 1000, height: 800 };

/** 21 landmarks with the index fingertip placed at (x, y). */
const hand = (x: number, y: number) =>
  Array.from({ length: 21 }, (_, i) => (i === 8 ? { x, y } : { x: 0.5, y: 0.5 }));

describe("PointerEngine", () => {
  it("maps the first frame directly, without smoothing lag", () => {
    // Smoothing toward an initial position of (0,0) would start every session
    // with the cursor sliding in from the corner.
    const p = new PointerEngine({ gain: 1 });
    const s = p.update(hand(0.5, 0.5), VIEWPORT, 0);
    expect(s.x).toBeCloseTo(500);
    expect(s.y).toBeCloseTo(400);
  });

  it("mirrors the x axis so hand-right is cursor-right", () => {
    const p = new PointerEngine({ gain: 1 });
    // A fingertip near x=0 in the (mirrored) camera frame is the user's right.
    expect(p.update(hand(0.0, 0.5), VIEWPORT, 0).x).toBeCloseTo(1000);
    p.reset();
    expect(p.update(hand(1.0, 0.5), VIEWPORT, 0).x).toBeCloseTo(0);
  });

  it("stretches the usable centre of the frame across the whole screen", () => {
    // With gain 0.5 the centre half of the frame covers the full width, so the
    // user reaches the screen edge without reaching the camera's edge.
    const p = new PointerEngine({ gain: 0.5 });
    expect(p.update(hand(0.75, 0.5), VIEWPORT, 0).x).toBeCloseTo(0);
    p.reset();
    expect(p.update(hand(0.25, 0.5), VIEWPORT, 0).x).toBeCloseTo(1000);
  });

  it("clamps rather than overshooting past the screen edge", () => {
    const p = new PointerEngine({ gain: 0.5 });
    expect(p.update(hand(0.95, 0.5), VIEWPORT, 0).x).toBe(0);
    p.reset();
    expect(p.update(hand(0.05, 0.95), VIEWPORT, 0).y).toBe(800);
  });

  it("clicks after the dwell time on a steady hand", () => {
    const p = new PointerEngine({ dwellMs: 500, gain: 1 });
    p.update(hand(0.5, 0.5), VIEWPORT, 0);

    let clicked = false;
    for (let t = 33; t <= 600; t += 33) {
      const s = p.update(hand(0.5, 0.5), VIEWPORT, t);
      if (s.clicked) {
        clicked = true;
        expect(s.dwellProgress).toBe(1);
        break;
      }
    }
    expect(clicked).toBe(true);
  });

  it("reports dwell progress monotonically while held", () => {
    const p = new PointerEngine({ dwellMs: 1000, gain: 1 });
    p.update(hand(0.5, 0.5), VIEWPORT, 0);
    const a = p.update(hand(0.5, 0.5), VIEWPORT, 200).dwellProgress;
    const b = p.update(hand(0.5, 0.5), VIEWPORT, 600).dwellProgress;
    expect(b).toBeGreaterThan(a);
  });

  it("does not click while the cursor is still travelling", () => {
    const p = new PointerEngine({ dwellMs: 300, dwellRadiusPx: 20, gain: 1 });
    let clicked = false;
    for (let i = 0; i < 40; i++) {
      // Sweeping steadily across the screen: never dwelling anywhere.
      const s = p.update(hand(0.1 + i * 0.02, 0.5), VIEWPORT, i * 33);
      if (s.clicked) clicked = true;
    }
    expect(clicked).toBe(false);
  });

  it("tolerates tremor within the dwell radius", () => {
    // Someone with a natural tremor must still be able to click; a strict
    // equality test on position would make that impossible.
    const p = new PointerEngine({ dwellMs: 400, dwellRadiusPx: 60, gain: 1 });
    p.update(hand(0.5, 0.5), VIEWPORT, 0);

    let clicked = false;
    for (let t = 33; t <= 800; t += 33) {
      const jitter = (t % 66 === 0 ? 1 : -1) * 0.004;
      if (p.update(hand(0.5 + jitter, 0.5), VIEWPORT, t).clicked) clicked = true;
    }
    expect(clicked).toBe(true);
  });

  it("does not fire a second click during the refractory period", () => {
    const p = new PointerEngine({ dwellMs: 200, refractoryMs: 1000, gain: 1 });
    let clicks = 0;
    for (let t = 0; t <= 900; t += 33) {
      if (p.update(hand(0.5, 0.5), VIEWPORT, t).clicked) clicks++;
    }
    expect(clicks).toBe(1);
  });

  it("holds position and cancels the dwell when tracking is lost", () => {
    const p = new PointerEngine({ dwellMs: 300, gain: 1 });
    p.update(hand(0.3, 0.7), VIEWPORT, 0);
    const held = p.update(hand(0.3, 0.7), VIEWPORT, 100);

    const lost = p.update(null, VIEWPORT, 133);
    expect(lost.active).toBe(false);
    expect(lost.clicked).toBe(false);
    expect(lost.dwellProgress).toBe(0);
    // Position is retained rather than snapping to a corner.
    expect(lost.x).toBeCloseTo(held.x);
    expect(lost.y).toBeCloseTo(held.y);
  });

  it("smooths toward the target instead of jumping", () => {
    const p = new PointerEngine({ smoothing: 0.25, gain: 1 });
    p.update(hand(0.5, 0.5), VIEWPORT, 0);
    const s = p.update(hand(0.9, 0.5), VIEWPORT, 33);
    // Target is x=100; a single step must cover only part of the distance.
    expect(s.x).toBeGreaterThan(100);
    expect(s.x).toBeLessThan(500);
  });
});
