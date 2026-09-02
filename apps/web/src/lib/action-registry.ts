/**
 * ActionRegistry — Industrial-grade dynamic gesture-to-action mapper.
 * 
 * Manages the "Action Library" and persistent user mappings between 
 * recognized Gesture IDs and specific system/browser actions.
 */

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

    // The Mappings: GestureID -> ActionID
    private mappings: Map<number, string> = new Map();

    private lastActionTime: number = 0;
    private COOLDOWN_MS = 1000;

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
        if (saved) {
            try {
                const data = JSON.parse(saved);
                this.mappings = new Map(Object.entries(data).map(([k, v]) => [Number(k), String(v)]));
            } catch (e) {
                console.error("DexteraAI: Failed to load mappings", e);
                this.setDefaults();
            }
        } else {
            this.setDefaults();
        }
    }

    private setDefaults() {
        // Default Industrial Mappings (Stabilized)
        this.mappings.set(1, "media_toggle");      // Open Palm
        this.mappings.set(5, "media_toggle");      // Peace Sign (Safely remapped from nav_back)
        this.mappings.set(3, "media_toggle");      // Thumbs Up
        this.mappings.set(6, "nav_scroll_up");     // Pointing Up
        this.mappings.set(4, "nav_scroll_down");   // Thumbs Down
        this.mappings.set(9, "sys_status_reset");  // Wave
        this.saveMappings();
    }

    private saveMappings() {
        if (typeof window === "undefined") return;
        const data = Object.fromEntries(this.mappings);
        localStorage.setItem(ActionRegistry.STORAGE_KEY, JSON.stringify(data));
    }

    /**
     * Remap a gesture to a specific action.
     */
    public remap(gestureId: number, actionId: string | null) {
        if (actionId === null) {
            this.mappings.delete(gestureId);
        } else {
            this.mappings.set(gestureId, actionId);
        }
        this.saveMappings();
        console.log(`DexteraAI: Remapped Gesture ${gestureId} -> ${actionId}`);
    }

    public async trigger(gestureId: number, confidence: number): Promise<GestureAction | null> {
        const actionId = this.mappings.get(gestureId);
        if (!actionId) return null;

        const action = this.actionLibrary.get(actionId);
        if (!action) return null;

        const now = Date.now();
        if (now - this.lastActionTime < this.COOLDOWN_MS) return null;

        // Minimum confidence check (can be dynamic via Calibrator elsewhere)
        if (confidence < 0.1) return null; // Logic handled in dashboard loop

        console.log(`DexteraAI: Executing Tactical Action [${action.id}]`);
        action.execute();
        this.lastActionTime = now;
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

    public getMappings(): Map<number, string> {
        return new Map(this.mappings);
    }

    public getActionById(actionId: string): GestureAction | undefined {
        return this.actionLibrary.get(actionId);
    }
}

export const actionRegistry = ActionRegistry.getInstance();
