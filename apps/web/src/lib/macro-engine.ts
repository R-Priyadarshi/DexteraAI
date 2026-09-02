"use client";

import { type GestureResult } from "./gesture-engine";

export interface MacroPattern {
    id: string;
    name: string;
    sequence: number[]; // Array of gesture IDs
    description: string;
    execute: () => void;
}

export class MacroEngine {
    private static instance: MacroEngine;
    private buffer: { id: number; timestamp: number }[] = [];
    private WINDOW_MS = 2000; // Increased window for accessibility
    private patterns: MacroPattern[] = [];
    private lastTriggerTime = 0;
    private COOLDOWN_MS = 2000;

    private constructor() {
        this.initializeDefaultMacros();
    }

    public static getInstance(): MacroEngine {
        if (!MacroEngine.instance) {
            MacroEngine.instance = new MacroEngine();
        }
        return MacroEngine.instance;
    }

    private initializeDefaultMacros() {
        // Example Macro: Peace -> Fist = "System Stealth Mode" (Redirect Disabled)
        this.register({
            id: "macro_stealth",
            name: "Stealth Mode",
            sequence: [5, 2], // Peace -> Closed Fist
            description: "Industrial status report log.",
            execute: () => {
                console.log("DexteraAI: Stealth redirect disabled for industrial continuity.");
            }
        });

        // Example Macro: Thumbs Up -> OK = "Confirm Mission"
        this.register({
            id: "macro_confirm",
            name: "Mission Confirmed",
            sequence: [3, 7], // Thumbs Up -> OK
            description: "Fires a custom industrial confirmation signal.",
            execute: () => {
                console.log("DexteraAI: Mission Confirmed Signal Sent");
                const event = new CustomEvent("dextera_macro", { detail: "MISSION_CONFIRMED" });
                window.dispatchEvent(event);
            }
        });
    }

    public register(pattern: MacroPattern) {
        this.patterns.push(pattern);
    }

    public process(result: GestureResult): MacroPattern | null {
        if (result.confidence < 0.92 || result.gestureId === -1) return null;

        const now = Date.now();
        if (now - this.lastTriggerTime < this.COOLDOWN_MS) return null;

        // Add to buffer only if it's a NEW gesture in the sequence
        const lastInEntry = this.buffer[this.buffer.length - 1];
        if (!lastInEntry || lastInEntry.id !== result.gestureId) {
            this.buffer.push({ id: result.gestureId, timestamp: now });
            console.log(`DexteraAI: Macro Buffer Update [${result.gestureName}]`, this.buffer.map(b => b.id));
        }

        // Clean window
        this.buffer = this.buffer.filter(b => now - b.timestamp < this.WINDOW_MS);

        // Matching logic (Longest patterns first to avoid partial matches)
        const sorted = [...this.patterns].sort((a, b) => b.sequence.length - a.sequence.length);

        for (const pattern of sorted) {
            if (this.isMatch(pattern.sequence)) {
                this.buffer = []; // Consume buffer
                this.lastTriggerTime = now;
                pattern.execute();
                return pattern;
            }
        }

        return null;
    }

    private isMatch(sequence: number[]): boolean {
        if (this.buffer.length < sequence.length) return false;

        const recent = this.buffer.slice(-sequence.length);
        for (let i = 0; i < sequence.length; i++) {
            if (recent[i].id !== sequence[i]) return false;
        }
        return true;
    }

    public getBuffer() {
        return this.buffer;
    }
}

export const macroEngine = MacroEngine.getInstance();
