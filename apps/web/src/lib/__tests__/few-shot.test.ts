import { describe, expect, it } from "vitest";
import { buildPrototype, matchPrototypes, type Prototype } from "../few-shot";

const vec = (...v: number[]) => Float32Array.from(v);

/** `n` samples scattered around `centre` by `radius`, deterministically. */
function cluster(centre: number[], radius: number, n: number): Float32Array[] {
  return Array.from({ length: n }, (_, i) => {
    const phase = (i / n) * Math.PI * 2;
    return Float32Array.from(
      centre.map((c, d) => c + radius * Math.sin(phase + d))
    );
  });
}

describe("buildPrototype", () => {
  it("returns null for a gesture with no samples", () => {
    expect(buildPrototype("a", "a", [])).toBeNull();
  });

  it("places the centroid at the mean of the samples", () => {
    const p = buildPrototype("a", "a", [vec(0, 0), vec(2, 4)])!;
    expect(Array.from(p.centroid)).toEqual([1, 2]);
  });

  it("measures a loose gesture as having more spread than a tight one", () => {
    const tight = buildPrototype("t", "t", cluster([0, 0, 0], 0.05, 12))!;
    const loose = buildPrototype("l", "l", cluster([0, 0, 0], 2.0, 12))!;
    expect(loose.spread).toBeGreaterThan(tight.spread);
  });

  it("floors the spread of near-identical samples", () => {
    // Teaching by holding perfectly still yields ~zero spread, which would
    // make every z-score divide by almost nothing.
    const p = buildPrototype("a", "a", [vec(1, 1), vec(1, 1), vec(1, 1)])!;
    expect(p.spread).toBeGreaterThan(0);
    expect(Number.isFinite(p.spread)).toBe(true);
  });

  it("handles a single sample without producing zero spread", () => {
    const p = buildPrototype("a", "a", [vec(3, 3)])!;
    expect(p.spread).toBeGreaterThan(0);
    expect(p.sampleCount).toBe(1);
  });
});

describe("matchPrototypes", () => {
  const fist = buildPrototype("f", "fist", cluster([0, 0, 0], 0.2, 10))!;
  const wave = buildPrototype("w", "wave", cluster([10, 10, 10], 0.2, 10))!;
  const all: Prototype[] = [fist, wave];

  it("returns null when there are no prototypes", () => {
    expect(matchPrototypes(vec(0, 0, 0), [])).toBeNull();
  });

  it("matches a query at the centre of a cluster", () => {
    const m = matchPrototypes(fist.centroid, all)!;
    expect(m.name).toBe("fist");
    expect(m.confidence).toBeGreaterThan(0.9);
  });

  it("rejects a query far from every prototype", () => {
    // The open-set case, and the common one: most of what a camera sees is not
    // a taught gesture, and returning the least-bad match would fire actions
    // constantly.
    expect(matchPrototypes(vec(500, 500, 500), all)).toBeNull();
  });

  it("is not dominated by a single outlier sample", () => {
    // Nearest-neighbour matches the closest example, so one badly-framed
    // recording during teaching becomes a permanent false-positive magnet.
    // A centroid absorbs it instead.
    const withOutlier = buildPrototype("f", "fist", [
      ...cluster([0, 0, 0], 0.2, 10),
      vec(40, 40, 40),
    ])!;
    const nearOutlier = vec(39, 39, 39);
    const m = matchPrototypes(nearOutlier, [withOutlier]);
    expect(m).toBeNull();
  });

  it("scores relative to each class's own spread", () => {
    // The same absolute distance should be unremarkable for a loose gesture
    // and unusual for a tight one.
    const tight = buildPrototype("t", "tight", cluster([0, 0], 0.1, 10))!;
    const loose = buildPrototype("l", "loose", cluster([0, 0], 3.0, 10))!;

    const query = vec(1.2, 0);
    const zTight = matchPrototypes(query, [tight])?.zDistance ?? Infinity;
    const zLoose = matchPrototypes(query, [loose])!.zDistance;

    expect(zLoose).toBeLessThan(zTight);
  });

  it("is less confident when two prototypes are equally close", () => {
    // Clusters loose enough that the midpoint is still within the acceptance
    // envelope of both — otherwise the query is simply rejected and there is
    // no confidence to compare.
    const a = buildPrototype("a", "a", cluster([0, 0], 1.0, 8))!;
    const b = buildPrototype("b", "b", cluster([2, 0], 1.0, 8))!;

    const decisive = matchPrototypes(vec(0, 0), [a, b])!.confidence;
    const ambiguous = matchPrototypes(vec(1, 0), [a, b])!.confidence;

    expect(ambiguous).toBeLessThan(decisive);
    expect(ambiguous).toBeLessThan(0.75);
  });

  it("reports confidence in [0, 1]", () => {
    for (const q of [vec(0, 0, 0), vec(1, 1, 1), vec(9, 9, 9), vec(10, 10, 10)]) {
      const m = matchPrototypes(q, all);
      if (m) {
        expect(m.confidence).toBeGreaterThanOrEqual(0);
        expect(m.confidence).toBeLessThanOrEqual(1);
      }
    }
  });

  it("picks the nearer of two prototypes", () => {
    // Queries sit just inside each cluster; a point midway between them is
    // correctly rejected rather than assigned, which the open-set test covers.
    expect(matchPrototypes(vec(10.1, 10.1, 10.1), all)!.name).toBe("wave");
    expect(matchPrototypes(vec(0.1, 0.1, 0.1), all)!.name).toBe("fist");
  });

  it("rejects a point midway between two gestures", () => {
    // Equidistant from both and close to neither: the honest answer is that
    // this is not either gesture, not a coin flip between them.
    expect(matchPrototypes(vec(5, 5, 5), all)).toBeNull();
  });
});
