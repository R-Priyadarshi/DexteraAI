/**
 * Stroke capture for air drawing — turns a stream of hand landmarks into
 * discrete strokes the recognizer can read.
 *
 * The hard part of drawing in the air is not the recognition, it is knowing
 * when the person meant to start and stop. A finger in view is always
 * somewhere; there is no surface to press against and no button to hold. So the
 * pen has to be explicit, and here it is a **pinch**: thumb and index together
 * is pen-down, apart is pen-up. That reads as holding one, costs no extra
 * hardware, and does not collide with any pose the classifier already binds.
 *
 * Two thresholds rather than one, because a single one stutters. A hand hovers
 * around whatever value you pick and the stroke shatters into fragments; the
 * gap between `pinchDown` and `pinchUp` is what keeps one stroke whole.
 */

import type { Point } from "./dollar-recognizer";

/** A MediaPipe hand landmark. Only x and y are used; depth is far too noisy. */
export interface Landmark {
    x: number;
    y: number;
    z?: number;
}

const THUMB_TIP = 4;
const INDEX_TIP = 8;
const WRIST = 0;
const MIDDLE_MCP = 9;

export interface AirCanvasConfig {
    /** Pinch ratio below which the pen goes down. */
    pinchDown: number;
    /** Pinch ratio above which the pen lifts. Must exceed `pinchDown`. */
    pinchUp: number;
    /** Ignore samples closer than this to the previous one, in hand-scale units. */
    minPointDistance: number;
    /** Strokes shorter than this are discarded as accidental. */
    minStrokePoints: number;
    /** Total path length a stroke must cover, in hand-scale units. */
    minStrokeLength: number;
    /** Hard cap, so a stroke held for a minute cannot grow without bound. */
    maxPoints: number;
    /**
     * Flip x when capturing. The dashboard previews the camera mirrored, as
     * every selfie view does, so a person drawing a "C" produces a backwards
     * one in raw image coordinates. Mirroring is not a rotation and the
     * recognizer normalises only rotation, so without this every asymmetric
     * shape is captured as its reflection and never matches.
     */
    mirrorX: boolean;
}

export const DEFAULT_AIR_CANVAS_CONFIG: AirCanvasConfig = {
    // Fingertips touching sit near 0.25 of the wrist-to-knuckle span; a relaxed
    // open hand is past 1.0. These sit either side of the transition with room
    // to spare.
    pinchDown: 0.5,
    pinchUp: 0.75,
    minPointDistance: 0.02,
    minStrokePoints: 8,
    minStrokeLength: 0.35,
    maxPoints: 512,
    mirrorX: true,
};

export type StrokeEvent =
    | { type: "start" }
    | { type: "move"; points: readonly Point[] }
    | { type: "end"; points: readonly Point[] }
    | { type: "discard"; reason: string };

export class AirCanvas {
    private readonly config: AirCanvasConfig;
    private points: Point[] = [];
    private penDown = false;

    constructor(config: Partial<AirCanvasConfig> = {}) {
        this.config = { ...DEFAULT_AIR_CANVAS_CONFIG, ...config };
        if (this.config.pinchUp <= this.config.pinchDown) {
            throw new Error(
                `pinchUp (${this.config.pinchUp}) must exceed pinchDown ` +
                    `(${this.config.pinchDown}); without a gap the pen chatters`,
            );
        }
    }

    /** Points captured so far in the current stroke, for drawing a trail. */
    get trail(): readonly Point[] {
        return this.points;
    }

    get isDrawing(): boolean {
        return this.penDown;
    }

    /**
     * Feed one frame's landmarks. Pass null when no hand was found.
     *
     * Returns an event on a state change, or null when nothing happened worth
     * reporting. A hand leaving the frame mid-stroke ends it rather than
     * discarding it: reaching outside the camera's view is a normal way to
     * finish a big shape, and throwing the stroke away there would feel like
     * the app dropping it.
     */
    feed(landmarks: Landmark[] | null): StrokeEvent | null {
        if (!landmarks || landmarks.length <= MIDDLE_MCP) {
            return this.penDown ? this.liftPen() : null;
        }

        const scale = distance(landmarks[WRIST], landmarks[MIDDLE_MCP]);
        if (scale <= 0) {
            // Degenerate hand: every ratio would be Infinity or NaN.
            return this.penDown ? this.liftPen() : null;
        }

        const pinch = distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / scale;
        const tip = landmarks[INDEX_TIP];
        const point: Point = {
            x: this.config.mirrorX ? 1 - tip.x : tip.x,
            y: tip.y,
        };

        if (!this.penDown) {
            if (pinch < this.config.pinchDown) {
                this.penDown = true;
                this.points = [point];
                return { type: "start" };
            }
            return null;
        }

        if (pinch > this.config.pinchUp) {
            return this.liftPen();
        }

        const last = this.points[this.points.length - 1];
        if (!last || distance(point, last) / scale >= this.config.minPointDistance) {
            this.points.push(point);
            // Drop the oldest rather than stopping capture, so an over-long
            // stroke degrades into its recent shape instead of freezing.
            if (this.points.length > this.config.maxPoints) {
                this.points.shift();
            }
            return { type: "move", points: this.points };
        }
        return null;
    }

    /** Abandon any stroke in progress. */
    reset(): void {
        this.penDown = false;
        this.points = [];
    }

    private liftPen(): StrokeEvent {
        this.penDown = false;
        const stroke = this.points;
        this.points = [];

        if (stroke.length < this.config.minStrokePoints) {
            return { type: "discard", reason: `only ${stroke.length} points` };
        }
        const length = pathLength(stroke);
        if (length < this.config.minStrokeLength) {
            return { type: "discard", reason: `path too short (${length.toFixed(2)})` };
        }
        return { type: "end", points: stroke };
    }
}

function distance(a: Landmark | Point, b: Landmark | Point): number {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function pathLength(points: readonly Point[]): number {
    let total = 0;
    for (let i = 1; i < points.length; i++) {
        total += distance(points[i], points[i - 1]);
    }
    return total;
}
