"use client";

import { type Landmark } from "./gesture-engine";

/**
 * Recorded motion clips, for training dynamic gestures.
 *
 * Both shipped models are trained on `frames_per_sequence: 1` — every sample is
 * a still image replicated across the temporal window. The architecture is a
 * 30-frame Transformer, so it is capable of motion, but it has never been shown
 * any: swipes, waves, circles and push/pull are not learned behaviours, and
 * what motion the product does expose comes from hand-written velocity
 * thresholds rather than the model.
 *
 * Closing that gap needs sequence data. The public options are awkward — Jester
 * is the obvious corpus and is distributed CC BY-NC-SA, which is a real gate on
 * shipping commercially — so this records it directly instead. A clip is
 * `frameCount` consecutive frames of landmarks, which is exactly the window the
 * model consumes at inference, so what is recorded matches what is classified.
 *
 * This is a capture format, not a recognition format: clips are exported for
 * offline training by `training/datasets/import_recordings.py`, not matched at
 * runtime.
 */

export interface MotionClip {
    id: string;
    label: string;
    /** Consecutive frames, each the 21 landmarks of one hand. */
    frames: Landmark[][];
    handedness: "left" | "right" | "unknown";
    recordedAt: number;
}

export interface MotionPack {
    format: "dextera.motion-pack";
    version: 1;
    exportedAt: string;
    /** Frames per clip. Must match the model's `seq_len` to train against it. */
    frameCount: number;
    clips: MotionClip[];
}

export const MOTION_PACK_FORMAT = "dextera.motion-pack";
export const MOTION_PACK_VERSION = 1;

const STORAGE_KEY = "dextera_motion_clips";
const LANDMARKS_PER_FRAME = 21;

/**
 * Clips are far larger than static samples — 30 frames rather than one — and
 * localStorage is a few megabytes at best. Past this count the store stops
 * accepting clips rather than throwing a quota error mid-session and losing
 * the recording the user just made.
 */
const MAX_CLIPS = 400;

export class MotionStore {
    private clips: MotionClip[] = [];

    constructor() {
        this.load();
    }

    private load() {
        if (typeof window === "undefined") return;
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return;
        try {
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) this.clips = parsed;
        } catch (err) {
            console.error("MotionStore: failed to load clips", err);
            this.clips = [];
        }
    }

    private save() {
        if (typeof window === "undefined") return;
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(this.clips));
        } catch (err) {
            // Quota exceeded. Report it rather than silently dropping the clip,
            // so the UI can tell the user to export and clear.
            console.error("MotionStore: could not save clips", err);
            throw new Error(
                "Local storage is full. Export your clips, then clear them to continue."
            );
        }
    }

    isFull(): boolean {
        return this.clips.length >= MAX_CLIPS;
    }

    getClips(): MotionClip[] {
        return [...this.clips];
    }

    /** Clip counts per label, for showing coverage while recording. */
    countsByLabel(): Record<string, number> {
        const counts: Record<string, number> = {};
        for (const clip of this.clips) {
            counts[clip.label] = (counts[clip.label] ?? 0) + 1;
        }
        return counts;
    }

    addClip(
        label: string,
        frames: Landmark[][],
        handedness: MotionClip["handedness"]
    ): MotionClip {
        if (this.isFull()) {
            throw new Error(`Clip limit reached (${MAX_CLIPS}). Export and clear first.`);
        }
        const clip: MotionClip = {
            id: `clip_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`,
            label: label.trim(),
            frames,
            handedness,
            recordedAt: Date.now(),
        };
        this.clips.push(clip);
        this.save();
        return clip;
    }

    deleteClip(id: string) {
        this.clips = this.clips.filter((c) => c.id !== id);
        this.save();
    }

    /** Remove every clip for one label. */
    deleteLabel(label: string) {
        this.clips = this.clips.filter((c) => c.label !== label);
        this.save();
    }

    clear() {
        this.clips = [];
        this.save();
    }

    exportPack(frameCount: number): MotionPack {
        return {
            format: MOTION_PACK_FORMAT,
            version: MOTION_PACK_VERSION,
            exportedAt: new Date().toISOString(),
            frameCount,
            clips: this.clips,
        };
    }

    /**
     * Merge a pack, so recordings from several people or sessions can be
     * pooled into one training set — which is the only way a self-recorded
     * dataset generalises beyond the person who recorded it.
     */
    importPack(raw: unknown): { imported: number; rejected: number } {
        const pack = raw as Partial<MotionPack>;
        if (!pack || pack.format !== MOTION_PACK_FORMAT) {
            throw new Error("Not a Dextera motion pack.");
        }
        if (pack.version !== MOTION_PACK_VERSION) {
            throw new Error(`Unsupported pack version ${String(pack.version)}.`);
        }
        if (!Array.isArray(pack.clips)) throw new Error("Pack contains no clips.");

        let imported = 0;
        let rejected = 0;
        const existing = new Set(this.clips.map((c) => c.id));

        for (const clip of pack.clips) {
            if (!this.isValidClip(clip) || existing.has(clip.id) || this.isFull()) {
                rejected++;
                continue;
            }
            this.clips.push(clip);
            existing.add(clip.id);
            imported++;
        }

        if (imported > 0) this.save();
        return { imported, rejected };
    }

    private isValidClip(clip: unknown): clip is MotionClip {
        const c = clip as Partial<MotionClip>;
        if (!c || typeof c.label !== "string" || !c.label.trim()) return false;
        if (typeof c.id !== "string" || !Array.isArray(c.frames)) return false;
        if (c.frames.length === 0) return false;
        return c.frames.every(
            (frame) =>
                Array.isArray(frame) &&
                frame.length === LANDMARKS_PER_FRAME &&
                frame.every(
                    (p) =>
                        p &&
                        Number.isFinite(p.x) &&
                        Number.isFinite(p.y) &&
                        Number.isFinite(p.z)
                )
        );
    }
}

export const motionStore = new MotionStore();
