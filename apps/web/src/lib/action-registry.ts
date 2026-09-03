/**
 * ActionRegistry — maps recognised gestures to actions.
 *
 * Bindings are keyed by gesture **label**, not by class index. Index is a
 * property of whichever model bundle happens to be loaded: id 2 is "fist" under
 * the HaGRID vocabulary and "closed_fist" under the built-in fallback, and the
 * ASL bundle reuses the same integers for letters. Persisting indices means a
 * user's bindings silently rebind themselves to unrelated gestures the moment
 * the active bundle changes, which is exactly what happened here.
 *
 * Dispatch is driven by segment onsets rather than raw frames — see
 * `gesture-segmenter.ts`. The classifier scores ~30 frames a second, so a
 * frame-driven binding would fire a non-idempotent action 30 times for one
 * held pose.
 */

import type { GestureResult } from "./gesture-engine";

export type ActionCategory = "media" | "navigation" | "system" | "custom";

export interface GestureAction {
    id: string;
    name: string;
    category: ActionCategory;
    execute: () => void;
    description: string;
}

export class ActionRegistry {
    private static instance: ActionRegistry;
    private static STORAGE_KEY = "dextera_action_mappings";

    // The Library: All available actions the system can perform
    private actionLibrary: Map<string, GestureAction> = new Map();

    /** Gesture label -> action id. */
    private mappings: Map<string, string> = new Map();

    /** Last dispatch time per action, for the per-action cooldown. */
    private lastFiredAt: Map<string, number> = new Map();

    /**
     * Floor between two dispatches of the same action. The segmenter already
     * guarantees one event per gesture, so this only guards against a user
     * genuinely repeating a pose faster than the action can absorb.
     */
    private cooldownMs = 250;

    private constructor() {
        this.initializeActionLibrary();
        this.loadMappings();
    }

    public static getInstance(): ActionRegistry {
        if (!ActionRegistry.instance) {
            ActionRegistry.instance = new ActionRegistry();
        }
        return ActionRegistry.instance;
    }

    private initializeActionLibrary() {
        const library: GestureAction[] = [
            {
                id: "media_toggle",
                name: "Play/Pause",
                category: "media",
                description: "Toggles media playback in the active tab.",
                execute: () => this.triggerKey(32)
            },
            {
                id: "nav_scroll_up",
                name: "Scroll Up",
                category: "navigation",
                description: "Smoothly scrolls the page up.",
                execute: () => window.scrollBy({ top: -400, behavior: "smooth" })
            },
            {
                id: "nav_scroll_down",
                name: "Scroll Down",
                category: "navigation",
                description: "Smoothly scrolls the page down.",
                execute: () => window.scrollBy({ top: 400, behavior: "smooth" })
            },
            {
                id: "sys_status_reset",
                name: "Reset System",
                category: "system",
                description: "Re-calibrates the gesture engine tracking.",
                execute: () => {
                    console.log("DexteraAI: System metrics reset triggered.");
                    window.dispatchEvent(new CustomEvent("dextera_sys_reset"));
                }
            },
            {
                id: "nav_back",
                name: "Go Back",
                category: "navigation",
                description: "Action disabled for session stability.",
                execute: () => console.log("DexteraAI: Navigation-Back disabled to prevent session interrupt.")
            },
            {
                id: "deck_next",
                name: "Next Deck Slide",
                category: "navigation",
                description: "Pivots the spatial deck to the next module.",
                execute: () => window.dispatchEvent(new CustomEvent("dextera_slide", { detail: "next" }))
            },
            {
                id: "deck_prev",
                name: "Prev Deck Slide",
                category: "navigation",
                description: "Pivots the spatial deck to the previous module.",
                execute: () => window.dispatchEvent(new CustomEvent("dextera_slide", { detail: "prev" }))
            },
            {
                id: "deck_first",
                name: "Jump to First",
                category: "navigation",
                description: "Pivots the spatial deck to the very first module.",
                execute: () => window.dispatchEvent(new CustomEvent("dextera_slide", { detail: "first" }))
            },
            {
                id: "deck_last",
                name: "Jump to Last",
                category: "navigation",
                description: "Pivots the spatial deck to the very last module.",
                execute: () => window.dispatchEvent(new CustomEvent("dextera_slide", { detail: "last" }))
            },
            {
                id: "sys_lock",
                name: "Secure System",
                category: "system",
                description: "Emergency biometric lock-down.",
                execute: () => window.dispatchEvent(new CustomEvent("dextera_sys_halt"))
            }
        ];

        library.forEach(action => this.actionLibrary.set(action.id, action));
    }

    private loadMappings() {
        if (typeof window === "undefined") return;
        const saved = localStorage.getItem(ActionRegistry.STORAGE_KEY);
        if (!saved) {
            this.setDefaults();
            return;
        }
        try {
            const data = JSON.parse(saved) as Record<string, string>;
            const entries = Object.entries(data);

            // Anything stored under a numeric key predates label-based
            // bindings. Those keys indexed a vocabulary that was never
            // actually loaded at runtime, so migrating them would carry the
            // mis-binding forward. Discard and re-seed instead.
            if (entries.length > 0 && entries.every(([k]) => /^\d+$/.test(k))) {
                console.warn(
                    "ActionRegistry: discarding legacy index-keyed bindings; re-seeding defaults"
                );
                this.setDefaults();
                return;
            }

            this.mappings = new Map(entries.map(([k, v]) => [k, String(v)]));
        } catch (e) {
            console.error("ActionRegistry: failed to load mappings", e);
            this.setDefaults();
        }
    }

    /**
     * Seed bindings for the shipped general-purpose vocabulary.
     *
     * These are HaGRID labels, matching `models/hagrid/labels.json`. A label
     * absent from the active bundle simply never fires, so seeding a superset
     * is harmless.
     */
    private setDefaults() {
        this.mappings = new Map([
            ["palm", "media_toggle"],
            ["fist", "sys_lock"],
            ["like", "nav_scroll_up"],
            ["dislike", "nav_scroll_down"],
            ["peace", "deck_next"],
            ["ok", "sys_status_reset"],
            ["one", "deck_prev"],
            ["stop", "media_toggle"],
        ]);
        this.saveMappings();
    }

    private saveMappings() {
        if (typeof window === "undefined") return;
        const data = Object.fromEntries(this.mappings);
        localStorage.setItem(ActionRegistry.STORAGE_KEY, JSON.stringify(data));
    }

    /** Bind a gesture label to an action, or unbind it with `null`. */
    public remap(gestureName: string, actionId: string | null) {
        if (actionId === null) {
            this.mappings.delete(gestureName);
        } else {
            this.mappings.set(gestureName, actionId);
        }
        this.saveMappings();
    }

    /**
     * Dispatch the action bound to a completed gesture onset.
     *
     * Callers must pass an `onset` event only. Passing every frame would fire
     * the bound action for the entire duration of a held pose; that is the
     * caller's responsibility because only the caller knows whether it is
     * consuming raw frames or segment events.
     */
    public dispatch(result: GestureResult): GestureAction | null {
        if (result.phase !== "onset" || result.rejected) return null;

        const actionId = this.mappings.get(result.gestureName);
        if (!actionId) return null;

        const action = this.actionLibrary.get(actionId);
        if (!action) return null;

        const now = Date.now();
        const last = this.lastFiredAt.get(action.id) ?? 0;
        if (now - last < this.cooldownMs) return null;

        try {
            action.execute();
        } catch (err) {
            // One misbehaving action must not take down the recognition loop.
            console.error(`ActionRegistry: action "${action.id}" threw`, err);
            return null;
        }
        this.lastFiredAt.set(action.id, now);
        return action;
    }

    private triggerKey(keyCode: number) {
        const event = new KeyboardEvent("keydown", {
            keyCode: keyCode,
            which: keyCode,
            bubbles: true
        });
        document.dispatchEvent(event);
    }

    public getAllActions(): GestureAction[] {
        return Array.from(this.actionLibrary.values());
    }

    public getMappings(): Map<string, string> {
        return new Map(this.mappings);
    }

    /** Action currently bound to a label, if any. */
    public getBinding(gestureName: string): GestureAction | undefined {
        const id = this.mappings.get(gestureName);
        return id ? this.actionLibrary.get(id) : undefined;
    }

    /** Register an action at runtime, e.g. from a plugin. */
    public registerAction(action: GestureAction): void {
        this.actionLibrary.set(action.id, action);
    }

    public getActionById(actionId: string): GestureAction | undefined {
        return this.actionLibrary.get(actionId);
    }
}

export const actionRegistry = ActionRegistry.getInstance();
