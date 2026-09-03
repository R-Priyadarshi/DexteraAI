"use client";

import { actionRegistry } from "./action-registry";
import { type GestureResult } from "./gesture-engine";

/**
 * Macros — actions bound to a *sequence* of gestures rather than a single one.
 *
 * Sequences are the way to get a large command surface out of a small
 * vocabulary: 18 gestures give 18 single bindings but 324 ordered pairs, and a
 * two-gesture macro is far harder to trigger by accident than a single pose.
 * That matters for anything destructive.
 *
 * Sequences are expressed as gesture **labels**, matching `ActionRegistry` — an
 * index is a property of the loaded bundle, not of the gesture, so an
 * index-keyed macro silently means something different under a different model.
 *
 * Steps advance on segment onsets. Consuming raw frames would fill the buffer
 * with ~30 repeats of the first pose per second and no sequence would ever
 * match.
 */
export interface MacroPattern {
    id: string;
    name: string;
    /** Ordered gesture labels that trigger this macro. */
    sequence: string[];
    description: string;
    execute: () => void;
    /**
     * Registry action this macro runs, when it has one. Only macros with an
     * `actionId` can be persisted — an arbitrary closure cannot be serialised
     * and revived, so storing one would produce a macro that silently does
     * nothing after a reload.
     */
    actionId?: string;
}

const STORAGE_KEY = "dextera_macros";

interface StoredMacro {
    id: string;
    name: string;
    sequence: string[];
    description: string;
    /** Action id in the registry to run when the sequence completes. */
    actionId: string;
}

export class MacroEngine {
    private static instance: MacroEngine;

    /** Onsets seen recently, oldest first. */
    private buffer: { name: string; timestamp: number }[] = [];

    /**
     * How long the whole sequence has to complete. Long enough to be performed
     * deliberately, short enough that two unrelated gestures minutes apart
     * cannot combine into a macro.
     */
    private readonly windowMs = 3000;

    /** Dead time after a match, so one sequence cannot fire twice. */
    private readonly cooldownMs = 1200;

    private patterns: MacroPattern[] = [];
    private lastTriggerTime = 0;

    private constructor() {
        this.loadUserMacros();
    }

    public static getInstance(): MacroEngine {
        if (!MacroEngine.instance) {
            MacroEngine.instance = new MacroEngine();
        }
        return MacroEngine.instance;
    }

    public register(pattern: MacroPattern) {
        this.patterns = this.patterns.filter((p) => p.id !== pattern.id);
        this.patterns.push(pattern);
    }

    public unregister(id: string) {
        this.patterns = this.patterns.filter((p) => p.id !== id);
        this.persist();
    }

    public getPatterns(): MacroPattern[] {
        return [...this.patterns];
    }

    /** Define a macro that runs a registered action, and persist it. */
    public defineMacro(
        name: string,
        sequence: string[],
        actionId: string,
        description = ""
    ): MacroPattern | null {
        const action = actionRegistry.getActionById(actionId);
        if (!action || sequence.length < 2) return null;

        const pattern: MacroPattern = {
            id: `macro_${Date.now().toString(36)}`,
            name,
            sequence,
            description: description || `${sequence.join(" → ")} runs ${action.name}`,
            execute: () => action.execute(),
            actionId,
        };
        this.register(pattern);
        this.persist();
        return pattern;
    }

    /**
     * Feed one recognition result.
     *
     * Returns the macro that fired, or null. Only `onset` events advance the
     * sequence; every other phase is ignored.
     */
    public process(result: GestureResult): MacroPattern | null {
        if (result.phase !== "onset" || result.rejected) return null;

        const now = Date.now();
        if (now - this.lastTriggerTime < this.cooldownMs) return null;

        this.buffer.push({ name: result.gestureName, timestamp: now });
        this.buffer = this.buffer.filter((b) => now - b.timestamp < this.windowMs);

        // Longest first: a two-step macro must not be consumed by a one-step
        // prefix of itself.
        const sorted = [...this.patterns].sort(
            (a, b) => b.sequence.length - a.sequence.length
        );

        for (const pattern of sorted) {
            if (this.isMatch(pattern.sequence)) {
                this.buffer = [];
                this.lastTriggerTime = now;
                try {
                    pattern.execute();
                } catch (err) {
                    console.error(`MacroEngine: macro "${pattern.id}" threw`, err);
                    return null;
                }
                return pattern;
            }
        }
        return null;
    }

    private isMatch(sequence: string[]): boolean {
        if (this.buffer.length < sequence.length) return false;
        const recent = this.buffer.slice(-sequence.length);
        return sequence.every((name, i) => recent[i].name === name);
    }

    /** Onsets currently in the window, for live display. */
    public getBuffer(): { name: string; timestamp: number }[] {
        return [...this.buffer];
    }

    public reset(): void {
        this.buffer = [];
    }

    private persist() {
        if (typeof window === "undefined") return;
        const stored: StoredMacro[] = this.patterns
            .filter((p): p is MacroPattern & { actionId: string } => Boolean(p.actionId))
            .map((p) => ({
                id: p.id,
                name: p.name,
                sequence: p.sequence,
                description: p.description,
                actionId: p.actionId,
            }));
        localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
    }

    private loadUserMacros() {
        if (typeof window === "undefined") return;
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return;
        try {
            const stored = JSON.parse(raw) as StoredMacro[];
            for (const m of stored) {
                const action = actionRegistry.getActionById(m.actionId);
                if (!action) continue;
                const pattern: MacroPattern = {
                    id: m.id,
                    name: m.name,
                    sequence: m.sequence,
                    description: m.description,
                    execute: () => action.execute(),
                    actionId: m.actionId,
                };
                this.patterns.push(pattern);
            }
        } catch (err) {
            console.error("MacroEngine: failed to load macros", err);
        }
    }
}

export const macroEngine = MacroEngine.getInstance();
