/**
 * $1 Unistroke Recognizer — shape recognition from a drawn path.
 *
 * Wobbrock, Wilson & Li, UIST 2007. Chosen here for a specific reason: it
 * learns a symbol from **one** example. Every other capability in this project
 * needed a corpus, and corpora are what has been blocking it — the fingerprint
 * model is trained on data nobody licensed, and word-level signing has no data
 * that can ship at all. This needs none. A user draws a shape once and it is
 * recognised from then on, which is also the honest answer to "recognise any
 * sign": the vocabulary is whatever its user puts in it.
 *
 * It is geometry, not learning. A candidate path and a template are both
 * reduced to a canonical form — resampled to a fixed point count, rotated to a
 * common angle, scaled to a common box, centred — and then compared by mean
 * point-to-point distance. Nothing is fitted and there is no model to ship.
 *
 * The one thing to know before adding templates: normalisation discards
 * absolute rotation, so shapes that differ only by orientation collapse
 * together. An arrow pointing left and one pointing right are the same template
 * here. That is deliberate — a triangle drawn askew should still read as a
 * triangle — and it is why the built-in set carries no directional shapes.
 * Direction is already handled by the velocity-based swipe path, which is the
 * right place for it.
 */

export interface Point {
    x: number;
    y: number;
}

export interface Template {
    name: string;
    /** Canonical form, produced by `normalize`. */
    points: Point[];
}

export interface Recognition {
    name: string;
    /** 0..1. 1 is an exact match against the canonical form. */
    score: number;
}

/** Points each stroke is resampled to. 64 is the value the paper settles on. */
const NUM_POINTS = 64;

/** Side of the square every stroke is scaled into, in arbitrary units. */
const SQUARE_SIZE = 250;

/** Half-width of the rotation search, in radians (±45°). */
const ANGLE_RANGE = (45 * Math.PI) / 180;

/** Search resolution, in radians (2°). */
const ANGLE_PRECISION = (2 * Math.PI) / 180;

const PHI = 0.5 * (-1 + Math.sqrt(5));

/**
 * Below this width-to-height ratio a stroke counts as a line and is scaled
 * uniformly. 0.25 keeps "I", "1" and "-" intact while leaving genuinely
 * two-dimensional letters on the non-uniform path.
 */
const LINEAR_ASPECT = 0.25;

/** Longest possible distance inside the square, used to turn distance into a score. */
const HALF_DIAGONAL = 0.5 * Math.sqrt(SQUARE_SIZE ** 2 + SQUARE_SIZE ** 2);

function pathLength(points: readonly Point[]): number {
    let total = 0;
    for (let i = 1; i < points.length; i++) {
        total += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
    }
    return total;
}

function centroid(points: readonly Point[]): Point {
    let x = 0;
    let y = 0;
    for (const p of points) {
        x += p.x;
        y += p.y;
    }
    return { x: x / points.length, y: y / points.length };
}

/**
 * Resample to `n` evenly spaced points.
 *
 * This is what makes the comparison independent of drawing speed: a slow hand
 * emits many frames per centimetre and a fast one few, and after resampling
 * both become the same path.
 */
function resample(points: readonly Point[], n: number): Point[] {
    const interval = pathLength(points) / (n - 1);
    if (!Number.isFinite(interval) || interval === 0) {
        return Array.from({ length: n }, () => ({ ...points[0] }));
    }

    let accumulated = 0;
    const out: Point[] = [{ ...points[0] }];
    const work = [...points];

    for (let i = 1; i < work.length; i++) {
        const prev = work[i - 1];
        const curr = work[i];
        const d = Math.hypot(curr.x - prev.x, curr.y - prev.y);

        if (accumulated + d >= interval) {
            const t = (interval - accumulated) / d;
            const next = { x: prev.x + t * (curr.x - prev.x), y: prev.y + t * (curr.y - prev.y) };
            out.push(next);
            // Reconsider the same segment from the new point, so a long segment
            // yields as many samples as it should rather than one.
            work.splice(i, 0, next);
            accumulated = 0;
        } else {
            accumulated += d;
        }
    }

    // Floating-point drift can leave the final point one short.
    while (out.length < n) {
        out.push({ ...work[work.length - 1] });
    }
    return out.slice(0, n);
}

/** Angle from the centroid to the first point — the stroke's own reference. */
function indicativeAngle(points: readonly Point[]): number {
    const c = centroid(points);
    return Math.atan2(c.y - points[0].y, c.x - points[0].x);
}

function rotateBy(points: readonly Point[], radians: number): Point[] {
    const c = centroid(points);
    const cos = Math.cos(radians);
    const sin = Math.sin(radians);
    return points.map((p) => ({
        x: (p.x - c.x) * cos - (p.y - c.y) * sin + c.x,
        y: (p.x - c.x) * sin + (p.y - c.y) * cos + c.y,
    }));
}

/**
 * Scale into a square of `SQUARE_SIZE`.
 *
 * Non-uniformly for most strokes, which is the paper's choice and worth
 * knowing: it makes a squashed circle match a round one. That suits drawing in
 * the air, where the aspect ratio of a shape says more about how far the arm
 * reached than about what the person meant.
 *
 * Except for strokes that are nearly a line, where it is actively harmful. A
 * hand-drawn "I" is 250 units tall and perhaps 8 wide, all of that width being
 * tremor; scaling x by 250/8 magnifies the tremor thirtyfold and turns the
 * letter into a wandering scribble that matches anything. So a stroke thinner
 * than `LINEAR_ASPECT` is scaled uniformly, which keeps a line a line. The
 * paper notes this case; several $1 variants handle it the same way.
 */
function scaleToSquare(points: readonly Point[]): Point[] {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const width = Math.max(...xs) - Math.min(...xs);
    const height = Math.max(...ys) - Math.min(...ys);
    const longest = Math.max(width, height);
    if (longest === 0) {
        return points.map((p) => ({ ...p }));
    }

    if (Math.min(width, height) / longest < LINEAR_ASPECT) {
        const factor = SQUARE_SIZE / longest;
        return points.map((p) => ({ x: p.x * factor, y: p.y * factor }));
    }
    return points.map((p) => ({
        x: width === 0 ? p.x : p.x * (SQUARE_SIZE / width),
        y: height === 0 ? p.y : p.y * (SQUARE_SIZE / height),
    }));
}

function translateToOrigin(points: readonly Point[]): Point[] {
    const c = centroid(points);
    return points.map((p) => ({ x: p.x - c.x, y: p.y - c.y }));
}

/**
 * Reduce a raw stroke to the form templates and candidates are compared in.
 *
 * `rotationInvariant` is the single most consequential switch in this file.
 * With it on, a shape drawn at any angle reads the same — right for a triangle.
 * With it off, orientation is preserved, which is the only way an alphabet can
 * work: M and W, N and Z, 6 and 9, C and U are each the same stroke turned
 * around, and normalising rotation away makes them literally indistinguishable.
 */
export function normalize(points: readonly Point[], rotationInvariant = true): Point[] {
    const resampled = resample(points, NUM_POINTS);
    const oriented = rotationInvariant
        ? rotateBy(resampled, -indicativeAngle(resampled))
        : resampled;
    return translateToOrigin(scaleToSquare(oriented));
}

/** Mean point-to-point distance between two canonical paths. */
function pathDistance(a: readonly Point[], b: readonly Point[]): number {
    let total = 0;
    for (let i = 0; i < a.length; i++) {
        total += Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y);
    }
    return total / a.length;
}

function distanceAtAngle(points: readonly Point[], template: Template, radians: number): number {
    return pathDistance(rotateBy(points, radians), template.points);
}

/**
 * Best distance over a small rotation window, by golden-section search.
 *
 * Aligning indicative angles gets the two paths close but not flush, so this
 * recovers the last few degrees. Golden section rather than a sweep because the
 * distance function is unimodal here, and this needs about a tenth of the
 * evaluations.
 */
function distanceAtBestAngle(
    points: readonly Point[],
    template: Template,
    angleRange: number,
): number {
    if (angleRange === 0) {
        return distanceAtAngle(points, template, 0);
    }
    let a = -angleRange;
    let b = angleRange;
    let x1 = PHI * a + (1 - PHI) * b;
    let f1 = distanceAtAngle(points, template, x1);
    let x2 = (1 - PHI) * a + PHI * b;
    let f2 = distanceAtAngle(points, template, x2);

    while (Math.abs(b - a) > ANGLE_PRECISION) {
        if (f1 < f2) {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = PHI * a + (1 - PHI) * b;
            f1 = distanceAtAngle(points, template, x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = (1 - PHI) * a + PHI * b;
            f2 = distanceAtAngle(points, template, x2);
        }
    }
    return Math.min(f1, f2);
}

export interface RecognizerOptions {
    /**
     * Whether a symbol keeps its meaning when turned. True for shapes, false
     * for anything where up is part of the identity — see `normalize`.
     */
    rotationInvariant: boolean;
    /**
     * Half-width of the rotation search, in radians. Even an orientation-
     * sensitive set wants a little, because a hand in the air is never level;
     * too much and the pairs that rotation-invariance would have merged start
     * merging anyway.
     */
    angleRange: number;
}

export const SHAPE_OPTIONS: RecognizerOptions = {
    rotationInvariant: true,
    angleRange: ANGLE_RANGE,
};

/** ±12° — enough for a tilted hand, far short of the 90° that turns N into Z. */
export const ORIENTED_OPTIONS: RecognizerOptions = {
    rotationInvariant: false,
    angleRange: (12 * Math.PI) / 180,
};

export class DollarRecognizer {
    private templates: Template[] = [];
    private readonly options: RecognizerOptions;

    constructor(options: RecognizerOptions = SHAPE_OPTIONS) {
        this.options = options;
    }

    /** Register a symbol from a single raw stroke. */
    learn(name: string, stroke: readonly Point[]): void {
        if (stroke.length < 2) {
            throw new Error(`Template "${name}" needs at least 2 points, got ${stroke.length}`);
        }
        this.templates.push({ name, points: normalize(stroke, this.options.rotationInvariant) });
    }

    /** Forget every template registered under `name`. Returns how many went. */
    forget(name: string): number {
        const before = this.templates.length;
        this.templates = this.templates.filter((t) => t.name !== name);
        return before - this.templates.length;
    }

    /** Distinct symbol names currently known. */
    get names(): string[] {
        return [...new Set(this.templates.map((t) => t.name))];
    }

    get size(): number {
        return this.templates.length;
    }

    /**
     * Best match for a stroke, or null when nothing is registered or the stroke
     * is too short to mean anything.
     *
     * The score is always returned rather than thresholded here, because the
     * right cutoff depends on how forgiving the surface wants to be — the
     * caller decides.
     */
    recognize(stroke: readonly Point[]): Recognition | null {
        if (this.templates.length === 0 || stroke.length < 2) {
            return null;
        }
        const candidate = normalize(stroke, this.options.rotationInvariant);

        let best: Template | null = null;
        let bestDistance = Infinity;
        for (const template of this.templates) {
            const d = distanceAtBestAngle(candidate, template, this.options.angleRange);
            if (d < bestDistance) {
                bestDistance = d;
                best = template;
            }
        }

        if (!best) return null;
        return { name: best.name, score: Math.max(0, 1 - bestDistance / HALF_DIAGONAL) };
    }
}
