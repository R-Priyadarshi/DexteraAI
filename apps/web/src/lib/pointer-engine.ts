"use client";

/**
 * Hands-free pointing: a cursor driven by the index fingertip, with dwell to click.
 *
 * This is the accessibility path. For someone who cannot use a mouse or a
 * touchscreen, gesture control is only useful if it can do what a pointer does
 * — reach an arbitrary target and activate it. A vocabulary of discrete poses
 * cannot do that; continuous positioning plus a dwell activation can.
 *
 * Dwell rather than a click gesture is deliberate. Any pose used as a click
 * moves the hand slightly as it is formed, which drags the cursor off target at
 * the exact moment precision matters. Dwell decouples the two: position with
 * the hand, activate by holding still.
 *
 * Everything here is pure state over positions; the DOM is only touched when a
 * dwell completes, so the engine can be unit-tested without a browser.
 */

export interface PointerState {
    /** Viewport coordinates, in CSS pixels. */
    x: number;
    y: number;
    /** Dwell progress in [0, 1]. Reaches 1 exactly when a click fires. */
    dwellProgress: number;
    /** True on the single frame a click was dispatched. */
    clicked: boolean;
    /** True while the fingertip is tracked. */
    active: boolean;
}

export interface PointerConfig {
    /**
     * Exponential smoothing factor in (0, 1]. Lower is steadier but laggier.
     * Landmark jitter is a few pixels frame to frame, which is enough to make
     * an unsmoothed cursor unusable for small targets.
     */
    smoothing: number;
    /**
     * How far the cursor may drift and still count as dwelling, in pixels.
     * Too small and no one with a natural tremor can ever click; too large and
     * the cursor clicks while still travelling.
     */
    dwellRadiusPx: number;
    /** How long the cursor must stay within that radius to click. */
    dwellMs: number;
    /** Dead time after a click, so one hold cannot fire repeatedly. */
    refractoryMs: number;
    /**
     * Fraction of the camera frame mapped to the full screen. The hand cannot
     * comfortably reach the edges of the camera's view, so the usable centre is
     * stretched to cover the whole viewport.
     */
    gain: number;
}

export const DEFAULT_POINTER_CONFIG: PointerConfig = {
    smoothing: 0.35,
    dwellRadiusPx: 42,
    dwellMs: 900,
    refractoryMs: 700,
    gain: 0.62,
};

/** Landmark index of the index fingertip in MediaPipe's hand model. */
const INDEX_FINGERTIP = 8;

export class PointerEngine {
    private config: PointerConfig;

    private x = 0;
    private y = 0;
    private initialised = false;

    /** Where the current dwell started, and when. */
    private anchorX = 0;
    private anchorY = 0;
    private dwellStart = 0;

    private refractoryUntil = 0;

    constructor(config: Partial<PointerConfig> = {}) {
        this.config = { ...DEFAULT_POINTER_CONFIG, ...config };
    }

    configure(config: Partial<PointerConfig>): void {
        this.config = { ...this.config, ...config };
    }

    getConfig(): PointerConfig {
        return { ...this.config };
    }

    reset(): void {
        this.initialised = false;
        this.dwellStart = 0;
    }

    /**
     * Advance one frame.
     *
     * `landmarks` is the tracked hand, or null when tracking is lost.
     * `viewport` is injectable so the mapping can be tested without a window.
     */
    update(
        landmarks: { x: number; y: number }[] | null,
        viewport: { width: number; height: number },
        now: number
    ): PointerState {
        if (!landmarks || landmarks.length <= INDEX_FINGERTIP) {
            // Hold the last position rather than snapping to a corner: a cursor
            // that jumps on every dropped frame is worse than one that pauses.
            this.dwellStart = 0;
            return {
                x: this.x,
                y: this.y,
                dwellProgress: 0,
                clicked: false,
                active: false,
            };
        }

        const tip = landmarks[INDEX_FINGERTIP];

        // The camera image is mirrored for the user, so x is flipped: moving
        // the hand right must move the cursor right.
        const targetX = this.mapAxis(1 - tip.x) * viewport.width;
        const targetY = this.mapAxis(tip.y) * viewport.height;

        if (!this.initialised) {
            this.x = targetX;
            this.y = targetY;
            this.initialised = true;
        } else {
            const a = this.config.smoothing;
            this.x += (targetX - this.x) * a;
            this.y += (targetY - this.y) * a;
        }

        if (now < this.refractoryUntil) {
            this.dwellStart = 0;
            return { x: this.x, y: this.y, dwellProgress: 0, clicked: false, active: true };
        }

        const drift = Math.hypot(this.x - this.anchorX, this.y - this.anchorY);
        if (this.dwellStart === 0 || drift > this.config.dwellRadiusPx) {
            // Moved out of the dwell zone: re-anchor and start again.
            this.anchorX = this.x;
            this.anchorY = this.y;
            this.dwellStart = now;
            return { x: this.x, y: this.y, dwellProgress: 0, clicked: false, active: true };
        }

        const held = now - this.dwellStart;
        const progress = Math.min(1, held / this.config.dwellMs);

        if (progress >= 1) {
            this.dwellStart = 0;
            this.refractoryUntil = now + this.config.refractoryMs;
            return { x: this.x, y: this.y, dwellProgress: 1, clicked: true, active: true };
        }

        return { x: this.x, y: this.y, dwellProgress: progress, clicked: false, active: true };
    }

    /**
     * Map a normalised camera axis to a normalised screen axis.
     *
     * The centre `gain` fraction of the frame covers the whole screen, so the
     * user reaches the edges without moving their arm to the edge of the
     * camera's view — where tracking degrades badly.
     */
    private mapAxis(v: number): number {
        const margin = (1 - this.config.gain) / 2;
        return Math.max(0, Math.min(1, (v - margin) / this.config.gain));
    }
}

/**
 * Activate whatever sits under the cursor.
 *
 * A real `click()` on the hit element is used rather than a synthetic
 * MouseEvent at the document level, so the target's own handlers, its label
 * associations, and native behaviour for links and form controls all work the
 * way they would for a mouse user. Anything less would make the pointer usable
 * for demos and useless for actually operating the interface.
 */
export function clickAt(x: number, y: number): Element | null {
    if (typeof document === "undefined") return null;

    const target = document.elementFromPoint(x, y);
    if (!target) return null;

    // Walk up to the nearest interactive ancestor: fingertip precision lands on
    // the text inside a button at least as often as on the button itself.
    const interactive = target.closest(
        'a, button, input, select, textarea, [role="button"], [role="link"], [tabindex]'
    );
    const el = (interactive ?? target) as HTMLElement;

    el.focus?.();
    el.click?.();
    return el;
}
