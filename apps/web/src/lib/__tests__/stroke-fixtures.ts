/**
 * Synthesising a stroke a hand would actually have produced.
 *
 * Shared by the recognizer and alphabet tests, and worth getting right: twice
 * now a bad simulation has failed tests for reasons that had nothing to do with
 * the code under test.
 *
 * The first was density. The built-in forms are sparse — a caret is three
 * vertices — so thinning them removed the corners that define the shape instead
 * of thinning a trail. Hence `densify` before anything else.
 *
 * The second was the *character* of the noise. Independent noise per point is
 * not what a hand does. A fingertip drifts: consecutive samples are strongly
 * correlated, and the wobble is slow relative to the sample rate. Adding white
 * noise instead inflates the measured path length several-fold — each step
 * being mostly jitter rather than travel — and since resampling distributes
 * points by arc length, the canonical form ends up describing the noise. Glyphs
 * then fail to match *themselves*, which is the tell that the fixture is wrong
 * rather than the recogniser. `tremor` interpolates between control points so
 * the wobble is smooth, which is both realistic and non-destructive.
 */

import type { Point } from "../dollar-recognizer";

export function mulberry32(seed: number): () => number {
    return () => {
        seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** Interpolate along a polyline to `n` evenly indexed points. */
export function densify(shape: readonly Point[], n: number): Point[] {
    const segments = shape.length - 1;
    return Array.from({ length: n }, (_, i) => {
        const t = (i / (n - 1)) * segments;
        const seg = Math.min(Math.floor(t), segments - 1);
        const local = t - seg;
        return {
            x: shape[seg].x + local * (shape[seg + 1].x - shape[seg].x),
            y: shape[seg].y + local * (shape[seg + 1].y - shape[seg].y),
        };
    });
}

/**
 * Smooth low-frequency wobble: `n` values in [-amplitude, amplitude], drawn at
 * one control point per `period` samples and linearly interpolated between.
 */
function tremor(n: number, amplitude: number, rand: () => number, period = 20): number[] {
    const controls = Array.from({ length: Math.ceil(n / period) + 2 }, () => (rand() - 0.5) * 2 * amplitude);
    return Array.from({ length: n }, (_, i) => {
        const t = i / period;
        const k = Math.floor(t);
        const local = t - k;
        return controls[k] + local * (controls[k + 1] - controls[k]);
    });
}

export interface StrokeOptions {
    /** Rotation applied to the whole stroke, radians. */
    rotate?: number;
    scale?: number;
    offset?: Point;
    /** Peak wobble, in the same units as the shape's coordinates. */
    jitter?: number;
    seed?: number;
    /** Thin the second half, standing in for a hand that speeds up. */
    unevenSpeed?: boolean;
    /** Points before transformation. */
    density?: number;
}

/** Turn an ideal form into a plausible hand-drawn stroke. */
export function handDrawn(shape: readonly Point[], opts: StrokeOptions = {}): Point[] {
    const {
        rotate = 0,
        scale = 1,
        offset = { x: 0, y: 0 },
        jitter = 0,
        seed = 1,
        density = 140,
    } = opts;

    const rand = mulberry32(seed);
    const cos = Math.cos(rotate);
    const sin = Math.sin(rotate);
    const wobbleX = tremor(density, jitter, rand);
    const wobbleY = tremor(density, jitter, rand);

    let points = densify(shape, density).map((p, i) => ({
        x: (p.x * cos - p.y * sin) * scale + offset.x + wobbleX[i],
        y: (p.x * sin + p.y * cos) * scale + offset.y + wobbleY[i],
    }));

    if (opts.unevenSpeed) {
        const mid = Math.floor(points.length / 2);
        points = [...points.slice(0, mid), ...points.slice(mid).filter((_, i) => i % 3 === 0)];
    }
    return points;
}
