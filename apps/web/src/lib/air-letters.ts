/**
 * A single-stroke alphabet for writing in the air.
 *
 * Two constraints shape every form here, and both come from the medium rather
 * than from taste.
 *
 * **One stroke.** There is no pen to lift. A pinch opens and closes a stroke,
 * so a letter that needs the pen picked up — the crossbar of an A, the dot of
 * an i — would need the person to pinch twice and the app to guess whether the
 * second stroke belongs to the first letter or starts the next one. Palm's
 * Graffiti solved this in 1996 by simplifying the letters instead, and these
 * follow it: A is a caret, T is drawn without lifting, X is a bowtie.
 *
 * **Orientation matters.** Unlike the shape set, these are matched with
 * rotation invariance switched off, because half the alphabet is the other half
 * turned around: M is W, N is Z, 6 is 9, C is U. `ORIENTED_OPTIONS` keeps a ±12°
 * tolerance so a tilted hand still reads, which is far short of the 90° that
 * would start merging those pairs.
 *
 * Coordinates are x-right, y-DOWN, matching image space — so -1 is the top of
 * the letter. Forms are drawn the way a right-handed person writes them, and
 * direction counts: a circle drawn clockwise and one drawn anticlockwise are
 * different strokes to the recognizer. Where a form retraces its own path (H,
 * T, Y), that is deliberate and harmless — resampling spreads points evenly
 * along the path however many times it doubles back.
 */

import type { Point } from "./dollar-recognizer";

/** Points along a circular arc, angles in degrees, y-down. */
function arc(fromDeg: number, toDeg: number, steps = 16, rx = 1, ry = 1): Point[] {
    return Array.from({ length: steps }, (_, i) => {
        const a = ((fromDeg + ((toDeg - fromDeg) * i) / (steps - 1)) * Math.PI) / 180;
        return { x: Math.cos(a) * rx, y: Math.sin(a) * ry };
    });
}

function shift(points: Point[], dx: number, dy: number): Point[] {
    return points.map((p) => ({ x: p.x + dx, y: p.y + dy }));
}

export const AIR_LETTERS: Record<string, Point[]> = {
    // Graffiti's A: the crossbar is dropped, leaving an unambiguous caret.
    A: [{ x: -0.9, y: 1 }, { x: 0, y: -1 }, { x: 0.9, y: 1 }],
    B: [
        { x: -0.6, y: 1 }, { x: -0.6, y: -1 }, { x: 0.25, y: -0.75 },
        { x: 0.35, y: -0.35 }, { x: -0.6, y: -0.05 }, { x: 0.4, y: 0.3 },
        { x: 0.35, y: 0.75 }, { x: -0.6, y: 1 },
    ],
    // Opens to the right, drawn top-to-bottom anticlockwise.
    C: arc(-60, 60, 18).map((p) => ({ x: -p.x, y: p.y })),
    D: [
        { x: -0.6, y: 1 }, { x: -0.6, y: -1 },
        ...arc(-90, 90, 14, 0.95, 1).map((p) => ({ x: p.x - 0.6, y: p.y })),
    ],
    // An epsilon, which is how E is written without lifting.
    E: [
        { x: 0.55, y: -0.85 }, { x: -0.2, y: -1 }, { x: -0.6, y: -0.55 },
        { x: 0.05, y: -0.05 }, { x: -0.6, y: 0.55 }, { x: -0.2, y: 1 },
        { x: 0.55, y: 0.85 },
    ],
    F: [{ x: 0.7, y: -1 }, { x: -0.45, y: -1 }, { x: -0.45, y: 1 }],
    // The bar has to be a real bar, turning the corner twice. A short nub was
    // read as L at small sizes, the arc having been flattened by jitter.
    G: [
        ...arc(-65, 65, 16).map((p) => ({ x: -p.x, y: p.y })),
        { x: 0.6, y: 0.75 }, { x: 0.6, y: 0.15 }, { x: 0.05, y: 0.15 },
    ],
    // Retraces the left stem to reach the crossbar, as a hand naturally would.
    H: [
        { x: -0.55, y: -1 }, { x: -0.55, y: 1 }, { x: -0.55, y: 0 },
        { x: 0.55, y: 0 }, { x: 0.55, y: -1 }, { x: 0.55, y: 1 },
    ],
    I: [{ x: 0, y: -1 }, { x: 0, y: 1 }],
    J: [{ x: 0.4, y: -1 }, { x: 0.4, y: 0.5 }, { x: 0, y: 1 }, { x: -0.5, y: 0.75 }],
    K: [
        { x: -0.55, y: -1 }, { x: -0.55, y: 1 }, { x: -0.55, y: 0.1 },
        { x: 0.55, y: -1 }, { x: -0.55, y: 0.1 }, { x: 0.55, y: 1 },
    ],
    L: [{ x: -0.5, y: -1 }, { x: -0.5, y: 1 }, { x: 0.65, y: 1 }],
    M: [
        { x: -0.85, y: 1 }, { x: -0.85, y: -1 }, { x: 0, y: 0.25 },
        { x: 0.85, y: -1 }, { x: 0.85, y: 1 },
    ],
    N: [{ x: -0.7, y: 1 }, { x: -0.7, y: -1 }, { x: 0.7, y: 1 }, { x: 0.7, y: -1 }],
    O: arc(-90, 270, 24),
    P: [
        { x: -0.55, y: 1 }, { x: -0.55, y: -1 }, { x: 0.35, y: -0.8 },
        { x: 0.4, y: -0.3 }, { x: -0.55, y: -0.05 },
    ],
    Q: [...arc(-90, 270, 22), { x: 0.45, y: 0.55 }, { x: 0.95, y: 1 }],
    R: [
        { x: -0.55, y: 1 }, { x: -0.55, y: -1 }, { x: 0.35, y: -0.8 },
        { x: 0.4, y: -0.3 }, { x: -0.55, y: -0.05 }, { x: 0.55, y: 1 },
    ],
    // Curved throughout, with no straight run anywhere. That is the whole
    // difference from 5, which opens with a hard horizontal bar and a corner;
    // without it the two are the same stroke and the classic OCR confusion
    // shows up here too.
    S: [
        { x: 0.5, y: -0.72 }, { x: 0.15, y: -1 }, { x: -0.3, y: -0.92 },
        { x: -0.52, y: -0.5 }, { x: -0.2, y: -0.16 }, { x: 0.25, y: 0.06 },
        { x: 0.52, y: 0.45 }, { x: 0.3, y: 0.88 }, { x: -0.18, y: 1 },
        { x: -0.55, y: 0.72 },
    ],
    // Across the top, back to the middle, then down.
    T: [{ x: -0.7, y: -1 }, { x: 0.7, y: -1 }, { x: 0, y: -1 }, { x: 0, y: 1 }],
    U: [
        { x: -0.6, y: -1 }, { x: -0.6, y: 0.45 }, { x: -0.25, y: 0.95 },
        { x: 0.25, y: 0.95 }, { x: 0.6, y: 0.45 }, { x: 0.6, y: -1 },
    ],
    V: [{ x: -0.75, y: -1 }, { x: 0, y: 1 }, { x: 0.75, y: -1 }],
    W: [
        { x: -0.9, y: -1 }, { x: -0.45, y: 1 }, { x: 0, y: -0.25 },
        { x: 0.45, y: 1 }, { x: 0.9, y: -1 },
    ],
    // A bowtie: down-right, across, down-left. One stroke, no lift.
    X: [{ x: -0.7, y: -1 }, { x: 0.7, y: 1 }, { x: -0.7, y: 1 }, { x: 0.7, y: -1 }],
    Y: [
        { x: -0.6, y: -1 }, { x: 0, y: 0 }, { x: 0.6, y: -1 },
        { x: 0, y: 0 }, { x: 0, y: 1 },
    ],
    Z: [{ x: -0.7, y: -1 }, { x: 0.7, y: -1 }, { x: -0.7, y: 1 }, { x: 0.7, y: 1 }],
};

export const AIR_DIGITS: Record<string, Point[]> = {
    // Slashed, because an unslashed zero and the letter O are the same stroke
    // and no amount of tuning separates them.
    // The slash carries a quarter of the path length on purpose. Shorter, and
    // resampling gives it too few points to outweigh the ring it shares with
    // the letter O — which is otherwise the same stroke at any orientation.
    "0": [...arc(-90, 270, 16, 0.62, 1), { x: 0.62, y: -1 }, { x: -0.62, y: 1 }],
    "1": [{ x: -0.35, y: -0.7 }, { x: 0, y: -1 }, { x: 0, y: 1 }],
    // A pronounced hook over the top, which is what separates it from 7 — that
    // is a straight bar into a straight diagonal and nothing else.
    "2": [
        { x: -0.55, y: -0.45 }, { x: -0.3, y: -0.88 }, { x: 0.15, y: -1 },
        { x: 0.52, y: -0.68 }, { x: 0.35, y: -0.12 }, { x: -0.5, y: 0.95 },
        { x: 0.6, y: 1 },
    ],
    "3": [
        { x: -0.5, y: -0.85 }, { x: 0.15, y: -1 }, { x: 0.45, y: -0.55 },
        { x: -0.15, y: -0.05 }, { x: 0.5, y: 0.4 }, { x: 0.15, y: 1 },
        { x: -0.5, y: 0.85 },
    ],
    "4": [{ x: 0.3, y: -1 }, { x: -0.6, y: 0.3 }, { x: 0.65, y: 0.3 }, { x: 0.3, y: 0.3 }, { x: 0.3, y: 1 }],
    // Deliberately angular where S is round: a flat top bar, a square corner,
    // and a straight stem down to the belly.
    "5": [
        { x: 0.55, y: -1 }, { x: -0.5, y: -1 }, { x: -0.5, y: -0.12 },
        { x: 0.05, y: -0.22 }, { x: 0.5, y: 0.25 }, { x: 0.35, y: 0.82 },
        { x: -0.2, y: 1 }, { x: -0.58, y: 0.75 },
    ],
    "6": [
        { x: 0.45, y: -1 }, { x: -0.3, y: -0.35 }, { x: -0.5, y: 0.45 },
        { x: 0, y: 1 }, { x: 0.5, y: 0.6 }, { x: 0.2, y: 0.1 },
        { x: -0.45, y: 0.3 },
    ],
    "7": [{ x: -0.6, y: -1 }, { x: 0.6, y: -1 }, { x: -0.2, y: 1 }],
    "8": [
        ...shift(arc(90, -270, 14, 0.45, 0.5), 0, -0.5),
        ...shift(arc(-90, 270, 14, 0.55, 0.5), 0, 0.5),
    ],
    "9": [
        { x: -0.45, y: 1 }, { x: 0.3, y: 0.35 }, { x: 0.5, y: -0.45 },
        { x: 0, y: -1 }, { x: -0.5, y: -0.6 }, { x: -0.2, y: -0.1 },
        { x: 0.45, y: -0.3 },
    ],
};

/**
 * Letters and digits are separate sets, and must be recognised separately.
 *
 * The near-twins are across the boundary, not within it: 0 and O, 1 and I, 5
 * and S, 2 and Z. Merged into one 36-symbol set they are still mostly readable
 * — measured at 100% for a steady hand — but they are the first things to go
 * as tremor rises, and at heavy tremor merging roughly triples the error rate
 * (see the measurements in air-letters.test.ts).
 *
 * Splitting them costs the writer nothing, because mode is the one thing they
 * always know and the recogniser would otherwise have to guess. Palm reached
 * the same answer in 1996 with separate areas of the Graffiti pad for letters
 * and numbers. The accuracy it buys arrives exactly when the person is already
 * struggling to draw steadily, which is when they can least afford to be
 * second-guessed.
 */
export const AIR_SETS = {
    letters: AIR_LETTERS,
    digits: AIR_DIGITS,
} as const;

export type AirSetName = keyof typeof AIR_SETS;

export const AIR_LETTER_NAMES = Object.keys(AIR_LETTERS);
export const AIR_DIGIT_NAMES = Object.keys(AIR_DIGITS);
