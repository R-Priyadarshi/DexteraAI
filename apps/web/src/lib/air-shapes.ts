/**
 * The built-in air-drawing vocabulary.
 *
 * Shapes are generated rather than transcribed from recorded strokes, which
 * keeps them exact and reviewable — a template is a few lines of geometry
 * instead of 64 opaque coordinate pairs. The recognizer resamples along the
 * path anyway, so a polygon needs only its vertices; curves are sampled here.
 *
 * What is *not* in this set is the more interesting half. `dollar-recognizer`
 * normalises rotation away, so any two shapes that differ only by orientation
 * collapse into one. That rules out arrows and directional lines — and they are
 * not missed, because directional intent is already served by the velocity
 * swipe path. It also rules out keeping both `caret` and `v`, or both `line`
 * drawn left-to-right and right-to-left. Where a pair collided, one was kept.
 *
 * The set is deliberately small. Every shape added makes every other shape
 * slightly easier to confuse, and a user teaching their own symbol is competing
 * against these. Eight distinct shapes recognise far better than twenty
 * overlapping ones.
 */

import type { Point } from "./dollar-recognizer";

/** Sample a parametric curve at `n` points over t in [0, 1]. */
function sample(n: number, fn: (t: number) => Point): Point[] {
    return Array.from({ length: n }, (_, i) => fn(i / (n - 1)));
}

const TAU = Math.PI * 2;

export const BUILT_IN_SHAPES: Record<string, Point[]> = {
    circle: sample(32, (t) => ({ x: Math.cos(t * TAU), y: Math.sin(t * TAU) })),

    // Closed, returning to the start corner — an unclosed square reads as a
    // "C" rotated, which is exactly the collision this set avoids.
    square: [
        { x: -1, y: -1 },
        { x: 1, y: -1 },
        { x: 1, y: 1 },
        { x: -1, y: 1 },
        { x: -1, y: -1 },
    ],

    triangle: [
        { x: 0, y: -1 },
        { x: 1, y: 1 },
        { x: -1, y: 1 },
        { x: 0, y: -1 },
    ],

    // Unequal arms, which is what separates it from `caret` after rotation
    // normalisation. A symmetric tick would be the same shape.
    check: [
        { x: -1, y: 0 },
        { x: -0.3, y: 1 },
        { x: 1, y: -1 },
    ],

    caret: [
        { x: -1, y: 1 },
        { x: 0, y: -1 },
        { x: 1, y: 1 },
    ],

    zigzag: [
        { x: -1, y: 0.6 },
        { x: -0.35, y: -0.6 },
        { x: 0.1, y: 0.6 },
        { x: 0.55, y: -0.6 },
        { x: 1, y: 0.6 },
    ],

    // Two and a bit turns. One turn is a circle; three starts to look like a
    // scribble at the resolution a hand can manage in the air.
    spiral: sample(48, (t) => ({
        x: t * Math.cos(t * TAU * 2.25),
        y: t * Math.sin(t * TAU * 2.25),
    })),

    // Open enough not to be a circle: three-quarters of a turn.
    c: sample(24, (t) => {
        const a = Math.PI * 0.25 + t * Math.PI * 1.5;
        return { x: Math.cos(a), y: Math.sin(a) };
    }),
};

export const BUILT_IN_SHAPE_NAMES = Object.keys(BUILT_IN_SHAPES);
