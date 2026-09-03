"use client";

/**
 * Prototype matching for user-taught gestures.
 *
 * The previous approach was nearest-neighbour over raw weighted features with
 * confidence read off a fixed linear ramp (`1 - distance / 4`). Three problems
 * with that, all of which show up in use:
 *
 * 1. **A single outlier sample decides the label.** Nearest-neighbour matches
 *    the closest *example*, so one badly-framed recording during teaching
 *    becomes a permanent false-positive magnet — a hand that resembles that one
 *    bad frame matches the whole gesture.
 * 2. **The confidence had no relationship to the data.** Distance 1.0 always
 *    meant 0.75, whether the gesture was tight and well-separated or diffuse
 *    and overlapping with another. It could not be compared across gestures,
 *    which is exactly what a match has to do.
 * 3. **Tight and loose gestures were treated identically.** A fist, whose
 *    samples cluster hard, and a wave, whose samples spread wide, got the same
 *    absolute distance threshold. One was over-rejected and the other
 *    over-accepted.
 *
 * This replaces it with class prototypes and per-class spread. Each gesture
 * becomes a mean vector plus the standard deviation of its samples' distances
 * from it. A query is scored in units of that spread — how unusual it would be
 * *for this gesture* — which makes scores comparable across gestures with
 * genuinely different tightness. Confidence is then a softmax over those
 * scores, so it reflects how much better the winner is than the alternatives
 * rather than a fixed function of one number.
 *
 * This is still not a calibrated probability, and it is not claimed as one; it
 * is a comparable score. The trained model's calibrated confidence always wins
 * where it is willing to commit — see `gesture-engine.ts`.
 */

export interface Prototype {
    id: string;
    name: string;
    /** Mean feature vector across the gesture's samples. */
    centroid: Float32Array;
    /**
     * Standard deviation of sample-to-centroid distances. Floored, because a
     * gesture taught from near-identical frames would otherwise have ~zero
     * spread and reject everything that is not a pixel-perfect repeat.
     */
    spread: number;
    sampleCount: number;
}

export interface PrototypeMatch {
    id: string;
    name: string;
    confidence: number;
    /** Distance to the winning prototype, in units of that class's spread. */
    zDistance: number;
}

/**
 * Floor on class spread.
 *
 * Teaching a gesture by holding perfectly still yields samples that are nearly
 * identical, and a spread near zero turns the z-score into a division by
 * almost nothing — every query then looks infinitely far away.
 */
const MIN_SPREAD = 0.15;

/**
 * How many spreads away a query may be and still be considered a member.
 *
 * At 3 spreads a genuine repeat of the gesture is almost always inside, while
 * an unrelated hand shape is almost always outside — the usual reason to pick
 * three standard deviations.
 */
const MAX_Z_DISTANCE = 3.0;

/**
 * Softmax temperature over negative z-distances.
 *
 * Lower is more decisive. This is set so that a clear winner one full spread
 * closer than its nearest rival lands around 0.9 confidence, which is what the
 * caller's acceptance threshold is tuned against.
 */
const SOFTMAX_TEMPERATURE = 0.55;

function euclidean(a: Float32Array, b: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sum += d * d;
    }
    return Math.sqrt(sum);
}

/** Build a prototype from a gesture's weighted sample vectors. */
export function buildPrototype(
    id: string,
    name: string,
    samples: Float32Array[]
): Prototype | null {
    if (samples.length === 0) return null;

    const dim = samples[0].length;
    const centroid = new Float32Array(dim);
    for (const sample of samples) {
        for (let i = 0; i < dim; i++) centroid[i] += sample[i];
    }
    for (let i = 0; i < dim; i++) centroid[i] /= samples.length;

    // Spread as the RMS distance from the centroid. A single sample has no
    // measurable spread, so it falls back to the floor rather than to zero.
    let sumSquares = 0;
    for (const sample of samples) {
        const d = euclidean(sample, centroid);
        sumSquares += d * d;
    }
    const spread = Math.sqrt(sumSquares / samples.length);

    return {
        id,
        name,
        centroid,
        spread: Math.max(MIN_SPREAD, spread),
        sampleCount: samples.length,
    };
}

/**
 * Match a query against a set of prototypes.
 *
 * Returns null when nothing is within `MAX_Z_DISTANCE` of any prototype — the
 * open-set case, which is the common one: most of what a camera sees is not any
 * taught gesture, and reporting the least-bad match there would fire actions
 * constantly.
 */
export function matchPrototypes(
    query: Float32Array,
    prototypes: Prototype[]
): PrototypeMatch | null {
    if (prototypes.length === 0) return null;

    const scored = prototypes.map((p) => ({
        prototype: p,
        z: euclidean(query, p.centroid) / p.spread,
    }));

    scored.sort((a, b) => a.z - b.z);
    const best = scored[0];

    if (best.z > MAX_Z_DISTANCE) return null;

    // Softmax over negative z-distances. With one prototype this is 1.0 by
    // definition, so the z-distance gate above is the only thing standing
    // between a single taught gesture and matching everything.
    let total = 0;
    const weights = scored.map(({ z }) => {
        const w = Math.exp(-z / SOFTMAX_TEMPERATURE);
        total += w;
        return w;
    });

    const confidence = total > 0 ? weights[0] / total : 0;

    return {
        id: best.prototype.id,
        name: best.prototype.name,
        confidence,
        zDistance: best.z,
    };
}
