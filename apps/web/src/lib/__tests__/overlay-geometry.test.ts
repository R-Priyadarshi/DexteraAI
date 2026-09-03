import { describe, expect, it } from "vitest";
import { coverGeometry, projectLandmark } from "../overlay-geometry";

describe("coverGeometry", () => {
  it("fills the box exactly when aspects match", () => {
    const g = coverGeometry(640, 480, 1280, 960);
    expect(g.drawnWidth).toBeCloseTo(1280);
    expect(g.drawnHeight).toBeCloseTo(960);
    expect(g.offsetX).toBeCloseTo(0);
    expect(g.offsetY).toBeCloseTo(0);
  });

  it("crops top and bottom for a 4:3 stream in a 16:9 box", () => {
    // The common case: most webcams are 4:3, most video panels are 16:9.
    const g = coverGeometry(640, 480, 1600, 900);
    expect(g.drawnWidth).toBeCloseTo(1600);
    expect(g.drawnHeight).toBeCloseTo(1200);
    expect(g.offsetX).toBeCloseTo(0);
    expect(g.offsetY).toBeCloseTo(-150);
  });

  it("crops left and right for a wide stream in a narrow box", () => {
    const g = coverGeometry(1920, 1080, 600, 800);
    expect(g.drawnHeight).toBeCloseTo(800);
    expect(g.drawnWidth).toBeCloseTo(1422.22, 1);
    expect(g.offsetY).toBeCloseTo(0);
    expect(g.offsetX).toBeCloseTo(-411.11, 1);
  });

  it("never leaves the box unfilled", () => {
    // The defining property of `cover`: the drawn video covers the box on both
    // axes, so no background shows through.
    for (const [vw, vh, bw, bh] of [
      [640, 480, 1600, 900],
      [1920, 1080, 600, 800],
      [1280, 720, 1280, 720],
      [480, 640, 1000, 500],
    ]) {
      const g = coverGeometry(vw, vh, bw, bh);
      expect(g.drawnWidth).toBeGreaterThanOrEqual(bw - 1e-6);
      expect(g.drawnHeight).toBeGreaterThanOrEqual(bh - 1e-6);
    }
  });

  it("degrades safely before stream metadata arrives", () => {
    // videoWidth is 0 until metadata loads; dividing by it would put NaN into
    // every coordinate for the first frames.
    const g = coverGeometry(0, 0, 800, 600);
    expect(g.drawnWidth).toBe(800);
    expect(g.drawnHeight).toBe(600);
    expect(Number.isFinite(g.offsetX)).toBe(true);
    expect(Number.isFinite(g.offsetY)).toBe(true);
  });
});

describe("projectLandmark", () => {
  const box = { w: 1600, h: 900 };
  const g = coverGeometry(640, 480, box.w, box.h);

  it("maps the frame centre to the box centre", () => {
    const p = projectLandmark({ x: 0.5, y: 0.5 }, g);
    expect(p.x).toBeCloseTo(box.w / 2);
    expect(p.y).toBeCloseTo(box.h / 2);
  });

  it("mirrors x so moving the hand right moves the overlay right", () => {
    const left = projectLandmark({ x: 0.1, y: 0.5 }, g);
    const right = projectLandmark({ x: 0.9, y: 0.5 }, g);
    expect(left.x).toBeGreaterThan(right.x);
  });

  it("can be told not to mirror", () => {
    const p = projectLandmark({ x: 0.25, y: 0.5 }, g, false);
    expect(p.x).toBeCloseTo(0.25 * g.drawnWidth + g.offsetX);
  });

  it("places cropped-away content outside the box", () => {
    // A landmark near the top of a 4:3 frame is cropped out of a 16:9 box, so
    // its projection must fall above the visible area rather than being
    // squashed into it.
    const p = projectLandmark({ x: 0.5, y: 0.02 }, g);
    expect(p.y).toBeLessThan(0);
  });

  it("reproduces a measured on-screen position", () => {
    // Regression lock against a real observed frame: a 640x480 stream in a
    // 1027x575 panel put the wrist at (0.5359, 0.7550), and the dot rendered
    // at (477, 484) within the panel.
    const measured = coverGeometry(640, 480, 1027, 575);
    const p = projectLandmark({ x: 0.5359, y: 0.755 }, measured);
    expect(p.x).toBeCloseTo(476.6, 0);
    expect(p.y).toBeCloseTo(483.9, 0);
  });
});
