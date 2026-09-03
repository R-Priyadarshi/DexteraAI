import { describe, expect, it } from "vitest";
import { GestureSegmenter, type GesturePhase } from "../gesture-segmenter";

/** Feed one label repeatedly and collect the phases it produced. */
function run(
  seg: GestureSegmenter,
  frames: Array<{ name: string; conf?: number; rejected?: boolean }>,
  startMs = 1000,
  stepMs = 33
): GesturePhase[] {
  return frames.map((f, i) =>
    seg.push(
      {
        gestureName: f.name,
        gestureId: f.name === "none" ? -1 : 1,
        confidence: f.conf ?? 0.95,
        rejected: f.rejected ?? false,
      },
      startMs + i * stepMs
    ).phase
  );
}

const repeat = (name: string, n: number, conf?: number) =>
  Array.from({ length: n }, () => ({ name, conf }));

describe("GestureSegmenter", () => {
  it("requires consecutive frames before opening a segment", () => {
    const seg = new GestureSegmenter({ onsetFrames: 4 });
    const phases = run(seg, repeat("palm", 6));
    // Frames 0-2 accumulate; the 4th agreeing frame opens the segment.
    expect(phases.slice(0, 3)).toEqual(["idle", "idle", "idle"]);
    expect(phases[3]).toBe("onset");
    expect(phases.slice(4)).toEqual(["hold", "hold"]);
  });

  it("emits exactly one onset for a sustained pose", () => {
    const seg = new GestureSegmenter();
    const phases = run(seg, repeat("palm", 60));
    expect(phases.filter((p) => p === "onset")).toHaveLength(1);
  });

  it("survives a single dropped frame without ending the segment", () => {
    // The whole point of exit hysteresis: brief tracking loss mid-gesture must
    // not produce an offset/onset pair from a hand that never moved.
    const seg = new GestureSegmenter({ onsetFrames: 3, offsetFrames: 6 });
    const phases = run(seg, [
      ...repeat("palm", 5),
      { name: "fist" },
      ...repeat("palm", 5),
    ]);
    expect(phases).not.toContain("offset");
    expect(phases.filter((p) => p === "onset")).toHaveLength(1);
  });

  it("closes the segment once disagreement is sustained", () => {
    const seg = new GestureSegmenter({ onsetFrames: 3, offsetFrames: 4 });
    const phases = run(seg, [...repeat("palm", 6), ...repeat("fist", 6)]);
    expect(phases).toContain("offset");
    expect(phases.indexOf("offset")).toBe(6 + 3);
  });

  it("does not open a new segment during the refractory period", () => {
    const seg = new GestureSegmenter({
      onsetFrames: 2,
      offsetFrames: 2,
      refractoryMs: 400,
    });
    // 33ms/frame: the tail is well inside the 400ms dead time.
    const phases = run(seg, [
      ...repeat("palm", 4),
      ...repeat("none", 3),
      ...repeat("fist", 6),
    ]);
    expect(phases.filter((p) => p === "onset")).toHaveLength(1);
  });

  it("ignores rejected frames entirely", () => {
    const seg = new GestureSegmenter({ onsetFrames: 2 });
    const phases = run(
      seg,
      Array.from({ length: 10 }, () => ({ name: "palm", rejected: true }))
    );
    expect(phases.every((p) => p === "idle")).toBe(true);
  });

  it("ignores background labels", () => {
    const seg = new GestureSegmenter({ onsetFrames: 2 });
    expect(run(seg, repeat("none", 10)).every((p) => p === "idle")).toBe(true);
  });

  it("uses a lower confidence bar to stay active than to start", () => {
    // Asymmetric thresholds: 0.55 cannot open a segment, but it can sustain
    // one. A symmetric threshold would sit on the boundary the signal is
    // dithering across, which is what produces chatter.
    const seg = new GestureSegmenter({
      onsetFrames: 2,
      enterConfidence: 0.7,
      exitConfidence: 0.45,
    });
    expect(run(seg, repeat("palm", 4, 0.55)).every((p) => p === "idle")).toBe(true);

    const seg2 = new GestureSegmenter({
      onsetFrames: 2,
      enterConfidence: 0.7,
      exitConfidence: 0.45,
    });
    const phases = run(seg2, [...repeat("palm", 3, 0.9), ...repeat("palm", 4, 0.55)]);
    expect(phases).not.toContain("offset");
  });

  it("reports how long a segment has been held", () => {
    const seg = new GestureSegmenter({ onsetFrames: 2 });
    let last = seg.push(
      { gestureName: "palm", gestureId: 1, confidence: 0.9 },
      0
    );
    for (let i = 1; i < 12; i++) {
      last = seg.push(
        { gestureName: "palm", gestureId: 1, confidence: 0.9 },
        i * 100
      );
    }
    expect(last.phase).toBe("hold");
    // Onset landed on frame 1 (t=100ms); the last frame is t=1100ms.
    expect(last.heldMs).toBe(1000);
  });

  it("gives each segment a distinct id", () => {
    const seg = new GestureSegmenter({
      onsetFrames: 2,
      offsetFrames: 2,
      refractoryMs: 0,
    });
    const ids = new Set<number>();
    let t = 0;
    for (const name of ["palm", "palm", "palm", "none", "none", "none",
                        "fist", "fist", "fist"]) {
      const e = seg.push(
        { gestureName: name, gestureId: name === "none" ? -1 : 1, confidence: 0.9 },
        (t += 33)
      );
      if (e.phase === "onset") ids.add(e.segmentId);
    }
    expect(ids.size).toBe(2);
  });

  it("abandons an open segment on reset without emitting an offset", () => {
    const seg = new GestureSegmenter({ onsetFrames: 2 });
    run(seg, repeat("palm", 5));
    expect(seg.isActive()).toBe(true);
    seg.reset();
    expect(seg.isActive()).toBe(false);
  });
});
