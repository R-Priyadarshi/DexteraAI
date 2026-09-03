"use client";

import { type Landmark } from "./gesture-engine";

export interface CustomGesture {
    id: string;
    name: string;
    /** One entry per demonstration: the 21 landmarks captured for that sample. */
    samples: Landmark[][];
    actionId?: string;
    createdAt: number;
}

/**
 * Serialised form of a gesture pack.
 *
 * Custom gestures otherwise live only in this browser's localStorage, which
 * means they are lost on a cache clear and cannot move to another machine or
 * be shared. A pack is plain JSON containing landmark coordinates only — no
 * imagery, no video, nothing that identifies the person who recorded it — so
 * it stays consistent with the product's on-device posture even when a user
 * chooses to send one to someone else.
 */
export interface GesturePack {
    format: "dextera.gesture-pack";
    /** Bumped only on a breaking change to the sample representation. */
    version: 1;
    exportedAt: string;
    gestures: CustomGesture[];
}

export const GESTURE_PACK_FORMAT = "dextera.gesture-pack";
export const GESTURE_PACK_VERSION = 1;

/** Landmarks per hand, as emitted by MediaPipe Hands. */
const LANDMARKS_PER_SAMPLE = 21;

export interface ImportReport {
    imported: number;
    /** Gestures skipped because a gesture of the same name already exists. */
    skippedDuplicates: string[];
    /** Gestures rejected as malformed, with the reason. */
    rejected: { name: string; reason: string }[];
}

/**
 * Unique id for a stored gesture.
 *
 * A timestamp alone is not enough: two gestures created in the same
 * millisecond would share an id, and every lookup that excludes "this gesture"
 * by id — rename's collision check, delete — would then act on both.
 */
function newGestureId(): string {
    return `custom_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export class GestureStore {
    private static STORAGE_KEY = "dextera_custom_gestures";
    private gestures: CustomGesture[] = [];

    constructor() {
        this.load();
    }

    private load() {
        if (typeof window === "undefined") return;
        const saved = localStorage.getItem(GestureStore.STORAGE_KEY);
        if (saved) {
            try {
                this.gestures = JSON.parse(saved);
            } catch (e) {
                console.error("DexteraAI: Failed to load custom gestures", e);
                this.gestures = [];
            }
        }
    }

    private save() {
        if (typeof window === "undefined") return;
        localStorage.setItem(GestureStore.STORAGE_KEY, JSON.stringify(this.gestures));
    }

    public addGesture(name: string, samples: Landmark[][]) {
        const newGesture: CustomGesture = {
            id: newGestureId(),
            name,
            samples,
            createdAt: Date.now()
        };
        this.gestures.push(newGesture);
        this.save();
        return newGesture;
    }

    public getGestures(): CustomGesture[] {
        return this.gestures;
    }

    public deleteGesture(id: string) {
        this.gestures = this.gestures.filter(g => g.id !== id);
        this.save();
    }

    public clear() {
        this.gestures = [];
        this.save();
    }

    /** Serialise every stored gesture into a portable pack. */
    public exportPack(): GesturePack {
        return {
            format: GESTURE_PACK_FORMAT,
            version: GESTURE_PACK_VERSION,
            exportedAt: new Date().toISOString(),
            gestures: this.gestures,
        };
    }

    /**
     * Merge a pack into the store.
     *
     * Imported content is untrusted — a pack may have been hand-edited or come
     * from someone else — so every gesture is validated before it is accepted.
     * A malformed sample that reached `matchCustomGesture` would produce NaN
     * distances and quietly poison recognition for every gesture, so rejecting
     * it here is what keeps a bad file from being worse than a useless one.
     *
     * Existing gestures are never overwritten: a name collision skips the
     * incoming gesture rather than replacing work the user did themselves.
     */
    public importPack(raw: unknown): ImportReport {
        const report: ImportReport = {
            imported: 0,
            skippedDuplicates: [],
            rejected: [],
        };

        const pack = raw as Partial<GesturePack>;
        if (!pack || pack.format !== GESTURE_PACK_FORMAT) {
            throw new Error("Not a Dextera gesture pack.");
        }
        if (pack.version !== GESTURE_PACK_VERSION) {
            throw new Error(
                `Unsupported pack version ${String(pack.version)}; expected ${GESTURE_PACK_VERSION}.`
            );
        }
        if (!Array.isArray(pack.gestures)) {
            throw new Error("Pack contains no gestures.");
        }

        const existingNames = new Set(
            this.gestures.map((g) => g.name.trim().toLowerCase())
        );

        for (const incoming of pack.gestures) {
            const name = typeof incoming?.name === "string" ? incoming.name.trim() : "";
            if (!name) {
                report.rejected.push({ name: "(unnamed)", reason: "missing name" });
                continue;
            }

            const invalid = this.validateSamples(incoming.samples);
            if (invalid) {
                report.rejected.push({ name, reason: invalid });
                continue;
            }

            if (existingNames.has(name.toLowerCase())) {
                report.skippedDuplicates.push(name);
                continue;
            }

            this.gestures.push({
                // A fresh id, so importing a pack twice into different stores
                // cannot produce two gestures that collide on id.
                id: newGestureId(),
                name,
                samples: incoming.samples,
                actionId: typeof incoming.actionId === "string" ? incoming.actionId : undefined,
                createdAt:
                    typeof incoming.createdAt === "number" ? incoming.createdAt : Date.now(),
            });
            existingNames.add(name.toLowerCase());
            report.imported++;
        }

        if (report.imported > 0) this.save();
        return report;
    }

    /** Returns a reason string when the samples are unusable, else null. */
    private validateSamples(samples: unknown): string | null {
        if (!Array.isArray(samples) || samples.length === 0) {
            return "no samples";
        }
        for (const sample of samples) {
            if (!Array.isArray(sample) || sample.length !== LANDMARKS_PER_SAMPLE) {
                return `each sample needs ${LANDMARKS_PER_SAMPLE} landmarks`;
            }
            for (const point of sample) {
                if (
                    !point ||
                    typeof point.x !== "number" ||
                    typeof point.y !== "number" ||
                    typeof point.z !== "number" ||
                    !Number.isFinite(point.x) ||
                    !Number.isFinite(point.y) ||
                    !Number.isFinite(point.z)
                ) {
                    return "landmark coordinates must be finite numbers";
                }
            }
        }
        return null;
    }

    /** Rename a stored gesture. Returns false if the name is taken or empty. */
    public renameGesture(id: string, name: string): boolean {
        const trimmed = name.trim();
        if (!trimmed) return false;
        const clash = this.gestures.some(
            (g) => g.id !== id && g.name.trim().toLowerCase() === trimmed.toLowerCase()
        );
        if (clash) return false;
        const target = this.gestures.find((g) => g.id === id);
        if (!target) return false;
        target.name = trimmed;
        this.save();
        return true;
    }
}

export const gestureStore = new GestureStore();
