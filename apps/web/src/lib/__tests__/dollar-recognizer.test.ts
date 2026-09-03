import { describe, expect, it } from "vitest";

import { BUILT_IN_SHAPES, BUILT_IN_SHAPE_NAMES } from "../air-shapes";
import { DollarRecognizer, normalize, type Point } from "../dollar-recognizer";

/**
 * Feeding a template back to itself proves almost nothing — the interesting
 * question is whether a *hand-drawn* version of a shape still matches, so these
 * synthesise one: rotated, scaled, moved, jittered, and sampled unevenly to
 * stand in for a hand that speeds up and slows down mid-stroke.
 *
 * Noise is seeded. A flaky recogniser test is worse than none, because the
 * failure gets blamed on the randomness rather than on the code.
 */

function mulberry32(seed: number): () => number {
    return () => {
        seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

interface DrawOptions {
    rotate?: number;
    scale?: number;
    offset?: Point;
    jitter?: number;
    seed?: number;
    /** Simulate variable drawing speed by resampling unevenly. */
    unevenSpeed?: boolean;
}

/**
 * Interpolate along a polyline to `n` points.
 *
 * The built-in shapes are sparse — `caret` is three vertices — and a hand emits
 * hundreds. Densifying first is what makes the rest of this simulation
 * meaningful: without it, "drop two thirds of the points" deletes the corners
 * that define the shape rather than thinning a dense trail, and the test fails
 * for a reason that has nothing to do with the recogniser.
 */
function densify(shape: Point[], n: number): Point[] {
    const segments = shape.length - 1;
    const out: Point[] = [];
    for (let i = 0; i < n; i++) {
        const t = (i / (n - 1)) * segments;
        const seg = Math.min(Math.floor(t), segments - 1);
        const local = t - seg;
        out.push({
            x: shape[seg].x + local * (shape[seg + 1].x - shape[seg].x),
            y: shape[seg].y + local * (shape[seg + 1].y - shape[seg].y),
        });
    }
    return out;
}

/** Turn an ideal shape into something a hand plausibly produced. */
function drawnByHand(shape: Point[], opts: DrawOptions = {}): Point[] {
    const { rotate = 0, scale = 1, offset = { x: 0, y: 0 }, jitter = 0, seed = 1 } = opts;
    const rand = mulberry32(seed);
    const cos = Math.cos(rotate);
    const sin = Math.sin(rotate);

    let points = densify(shape, 120).map((p) => ({
        x: (p.x * cos - p.y * sin) * scale + offset.x + (rand() - 0.5) * 2 * jitter,
        y: (p.x * sin + p.y * cos) * scale + offset.y + (rand() - 0.5) * 2 * jitter,
    }));

    if (opts.unevenSpeed) {
        // Keep every point in the first half, every third in the second — a
        // stroke that started carefully and finished in a rush.
        const mid = Math.floor(points.length / 2);
        points = [...points.slice(0, mid), ...points.slice(mid).filter((_, i) => i % 3 === 0)];
    }
    return points;
}

function recognizerWithBuiltIns(): DollarRecognizer {
    const r = new DollarRecognizer();
    for (const [name, points] of Object.entries(BUILT_IN_SHAPES)) {
        r.learn(name, points);
    }
    return r;
}

describe("normalize", () => {
    it("puts a stroke in a canonical position, size and orientation", () => {
        // Both sides start from the same point density. Comparing a sparse
        // polyline against a densified one would measure resampling
        // discretisation (~0.2% here) rather than the invariance being claimed.
        const a = normalize(densify(BUILT_IN_SHAPES.triangle, 120));
        const b = normalize(drawnByHand(BUILT_IN_SHAPES.triangle, { scale: 7, offset: { x: 400, y: -90 } }));

        for (let i = 0; i < a.length; i++) {
            expect(a[i].x).toBeCloseTo(b[i].x, 4);
            expect(a[i].y).toBeCloseTo(b[i].y, 4);
        }
    });

    it("produces the same point count regardless of input length", () => {
        // Speed determines how many samples a stroke has; meaning does not.
        const sparse = normalize([
            { x: 0, y: 0 },
            { x: 1, y: 1 },
        ]);
        const dense = normalize(BUILT_IN_SHAPES.spiral);
        expect(sparse).toHaveLength(dense.length);
    });
});

describe("DollarRecognizer", () => {
    it("returns null when it knows nothing", () => {
        expect(new DollarRecognizer().recognize(BUILT_IN_SHAPES.circle)).toBeNull();
    });

    it("returns null for a stroke too short to be a shape", () => {
        expect(recognizerWithBuiltIns().recognize([{ x: 0, y: 0 }])).toBeNull();
    });

    it("rejects a template that is a single point", () => {
        expect(() => new DollarRecognizer().learn("dot", [{ x: 0, y: 0 }])).toThrow(/at least 2 points/);
    });

    it.each(BUILT_IN_SHAPE_NAMES)("recognises a hand-drawn %s", (name) => {
        const drawn = drawnByHand(BUILT_IN_SHAPES[name], {
            rotate: 0.25,
            scale: 3.5,
            offset: { x: 120, y: -60 },
            jitter: 0.06,
            seed: 42,
            unevenSpeed: true,
        });

        const result = recognizerWithBuiltIns().recognize(drawn);
        expect(result?.name).toBe(name);
        expect(result!.score).toBeGreaterThan(0.75);
    });

    it("tells the built-in shapes apart from one another", () => {
        // Each shape must beat all seven others, not merely score well against
        // itself — a recogniser where everything matches everything is useless.
        const recognizer = recognizerWithBuiltIns();
        for (const name of BUILT_IN_SHAPE_NAMES) {
            const drawn = drawnByHand(BUILT_IN_SHAPES[name], { jitter: 0.04, seed: 7 });
            expect(recognizer.recognize(drawn)?.name).toBe(name);
        }
    });

    it("survives a stroke drawn at a different size and place entirely", () => {
        const recognizer = recognizerWithBuiltIns();
        const tiny = drawnByHand(BUILT_IN_SHAPES.check, { scale: 0.05, offset: { x: -900, y: 400 } });
        expect(recognizer.recognize(tiny)?.name).toBe("check");
    });

    it("learns a symbol from one example", () => {
        // The whole reason for this algorithm: no corpus, no training step.
        const recognizer = new DollarRecognizer();
        const lightning: Point[] = [
            { x: 0.6, y: -1 },
            { x: -0.2, y: 0.05 },
            { x: 0.3, y: 0.05 },
            { x: -0.5, y: 1 },
        ];
        recognizer.learn("lightning", lightning);

        const result = recognizer.recognize(drawnByHand(lightning, { jitter: 0.05, seed: 3 }));
        expect(result?.name).toBe("lightning");
        expect(result!.score).toBeGreaterThan(0.8);
    });

    it("forgets a symbol on request", () => {
        const recognizer = recognizerWithBuiltIns();
        expect(recognizer.names).toContain("circle");

        expect(recognizer.forget("circle")).toBe(1);
        expect(recognizer.names).not.toContain("circle");
        expect(recognizer.forget("circle")).toBe(0);
        expect(recognizer.recognize(BUILT_IN_SHAPES.circle)?.name).not.toBe("circle");
    });

    it("keeps several examples of one symbol and matches the closest", () => {
        // A user teaching the same sign twice should improve it, not shadow it.
        const recognizer = new DollarRecognizer();
        recognizer.learn("wave", BUILT_IN_SHAPES.zigzag);
        recognizer.learn("wave", BUILT_IN_SHAPES.spiral);

        expect(recognizer.size).toBe(2);
        expect(recognizer.names).toEqual(["wave"]);
        expect(recognizer.recognize(BUILT_IN_SHAPES.spiral)?.name).toBe("wave");
    });

    it("scores an unrelated scribble lower than a real shape", () => {
        // There is no rejection threshold inside the recogniser; the caller
        // sets one. That only works if the score is actually discriminative.
        const recognizer = recognizerWithBuiltIns();
        const rand = mulberry32(99);
        const scribble = Array.from({ length: 40 }, () => ({ x: rand() * 2 - 1, y: rand() * 2 - 1 }));

        const noise = recognizer.recognize(scribble)!.score;
        const real = recognizer.recognize(drawnByHand(BUILT_IN_SHAPES.triangle, { jitter: 0.03 }))!.score;
        expect(noise).toBeLessThan(real);
    });
});
