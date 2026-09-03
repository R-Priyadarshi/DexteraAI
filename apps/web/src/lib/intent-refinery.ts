"use client";

import { type GestureResult } from "./gesture-engine";
import { type VoiceIntent } from "./voice-engine";
import { hapticEngine } from "./haptic-engine";
import { actionRegistry } from "./action-registry";

export interface FusedAction {
    id: string;
    name: string;
    /** Gesture label, not class index — see `action-registry.ts`. */
    gestureName: string;
    voiceIntent: VoiceIntent;
    execute: () => void;
    feedbackType: "success" | "error" | "light";
}

interface TemporalIntent {
    intent: VoiceIntent;
    timestamp: number;
}

interface TemporalGesture {
    gestureName: string;
    timestamp: number;
}

/**
 * Fuses a gesture with a spoken intent, requiring both within a short window.
 *
 * Requiring two independent modalities to agree is a deliberate safety
 * property, not a convenience: it makes accidental triggering effectively
 * impossible, which is what an irreversible action needs. A hand that drifts
 * into a pose cannot fire one, and neither can a word overheard in
 * conversation.
 *
 * Order does not matter — saying "abort" then raising a hand works as well as
 * the reverse — because forcing one order makes the interaction feel like a
 * password rather than a confirmation.
 */
export class IntentRefinery {
    private static instance: IntentRefinery;
    private fusedActions: FusedAction[] = [];
    
    // Temporal Buffers
    private voiceBuffer: TemporalIntent[] = [];
    private gestureBuffer: TemporalGesture[] = [];
    
    private readonly FUSION_WINDOW_MS = 2000;
    private readonly COOLDOWN_MS = 1500;
    private lastTriggerTime = 0;

    private constructor() {
        this.initializeDefaults();
    }

    public static getInstance(): IntentRefinery {
        if (!IntentRefinery.instance) {
            IntentRefinery.instance = new IntentRefinery();
        }
        return IntentRefinery.instance;
    }

    /**
     * Default fusions, keyed to the shipped general-purpose vocabulary.
     *
     * Each one runs a real registry action. A fusion whose only effect is a log
     * line is worse than no fusion at all, because it teaches the user the
     * interaction works when nothing is actually bound to it.
     */
    private initializeDefaults() {
        const run = (actionId: string) => () => {
            const action = actionRegistry.getActionById(actionId);
            if (!action) {
                console.warn(`IntentRefinery: no action "${actionId}" registered`);
                return;
            }
            action.execute();
        };

        this.register({
            id: "fusion_halt",
            name: "Emergency halt",
            gestureName: "stop",
            voiceIntent: "abort",
            feedbackType: "error",
            execute: run("sys_lock"),
        });

        this.register({
            id: "fusion_confirm",
            name: "Confirm",
            gestureName: "ok",
            voiceIntent: "confirm",
            feedbackType: "success",
            execute: run("media_toggle"),
        });

        this.register({
            id: "fusion_reset",
            name: "Reset tracking",
            gestureName: "palm",
            voiceIntent: "reset",
            feedbackType: "light",
            execute: run("sys_status_reset"),
        });
    }

    public register(action: FusedAction) {
        this.fusedActions.push(action);
    }

    /**
     * Probabilistically fuses gestures and voice intents within a temporal window.
     */
    public process(gesture: GestureResult, voice: VoiceIntent | null): FusedAction | null {
        const now = Date.now();

        // 1. Cooldown Check
        if (now - this.lastTriggerTime < this.COOLDOWN_MS) return null;

        // 2. Update Buffers
        if (voice) {
            this.voiceBuffer.push({ intent: voice, timestamp: now });
        }
        // Onsets only. Buffering every frame of a held pose would fill the
        // window with 60 copies of the same gesture and make the "consume both
        // triggers" step below meaningless.
        if (gesture.phase === "onset" && !gesture.rejected) {
            this.gestureBuffer.push({ gestureName: gesture.gestureName, timestamp: now });
        }

        // 3. Purge Stale Data
        this.voiceBuffer = this.voiceBuffer.filter(v => now - v.timestamp < this.FUSION_WINDOW_MS);
        this.gestureBuffer = this.gestureBuffer.filter(g => now - g.timestamp < this.FUSION_WINDOW_MS);

        if (this.voiceBuffer.length === 0 || this.gestureBuffer.length === 0) return null;

        // 4. Fusion Matcher (Cross-Product search within windows)
        for (const action of this.fusedActions) {
            const voiceMatch = this.voiceBuffer.find(v => v.intent === action.voiceIntent);
            const gestureMatch = this.gestureBuffer.find(g => g.gestureName === action.gestureName);

            if (voiceMatch && gestureMatch) {
                // Determine which came first (for potentially different logic, currently just fuse)
                const timeDiff = Math.abs(voiceMatch.timestamp - gestureMatch.timestamp);
                
                if (timeDiff < this.FUSION_WINDOW_MS) {
                    // Consume both triggers to prevent double-firing
                    this.voiceBuffer = this.voiceBuffer.filter(v => v !== voiceMatch);
                    this.gestureBuffer = this.gestureBuffer.filter(g => g !== gestureMatch);
                    
                    this.lastTriggerTime = now;
                    hapticEngine.pulse(action.feedbackType);
                    action.execute();
                    return action;
                }
            }
        }

        return null;
    }
}

export const intentRefinery = IntentRefinery.getInstance();
