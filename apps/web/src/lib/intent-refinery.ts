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
    
    /** Last intent the caller reported, so a held one is not re-buffered. */
    private lastVoiceSeen: VoiceIntent | null = null;

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
     * Clear buffers and cooldown, and restore the default fusions.
     *
     * Buffers describe one continuous session. Carrying them across a camera
     * restart means a word spoken before the restart can still fire something
     * after it.
     */
    public reset(): void {
        this.voiceBuffer = [];
        this.gestureBuffer = [];
        this.lastVoiceSeen = null;
        this.lastTriggerTime = 0;
        this.fusedActions = [];
        this.initializeDefaults();
    }

    /**
     * Probabilistically fuses gestures and voice intents within a temporal window.
     */
    public process(gesture: GestureResult, voice: VoiceIntent | null): FusedAction | null {
        const now = Date.now();

        // 1. Cooldown Check
        if (now - this.lastTriggerTime < this.COOLDOWN_MS) return null;

        // 2. Update Buffers
        //
        // Voice is edge-triggered, the same way gestures are taken on `onset`
        // below. The caller holds a recognised intent for about two seconds so
        // it can be displayed, and `process` runs every frame, so pushing on
        // every call buffered ~60 copies of one spoken word: one was consumed
        // on the match and the rest stayed, ready to fire again the moment the
        // cooldown lapsed.
        //
        // Deduplicating against the buffer is not enough, which a test caught:
        // a match empties the buffer, so the very next frame re-adds the same
        // utterance and the window slides forward for as long as the caller
        // keeps reporting it. Only a transition counts as a new utterance.
        if (voice && voice !== this.lastVoiceSeen) {
            this.voiceBuffer.push({ intent: voice, timestamp: now });
        }
        this.lastVoiceSeen = voice;
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
                // No separation check here: the purge above already dropped
                // anything older than FUSION_WINDOW_MS, so both survivors are
                // within that of now and therefore within it of each other.
                // The explicit `timeDiff < FUSION_WINDOW_MS` that used to sit
                // here could not fail, which made it read as a safety
                // condition while enforcing nothing.

                // Consume both triggers so neither can fire a second action.
                this.voiceBuffer = this.voiceBuffer.filter(v => v !== voiceMatch);
                this.gestureBuffer = this.gestureBuffer.filter(g => g !== gestureMatch);

                this.lastTriggerTime = now;
                hapticEngine.pulse(action.feedbackType);
                action.execute();
                return action;
            }
        }

        return null;
    }
}

export const intentRefinery = IntentRefinery.getInstance();
