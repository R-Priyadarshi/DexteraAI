"use client";

import { type GestureResult } from "./gesture-engine";
import { type FacialMarker } from "./face-engine";
import { type VoiceIntent } from "./voice-engine";
import { hapticEngine } from "./haptic-engine";
import { actionRegistry } from "./action-registry";

export interface FusedAction {
    id: string;
    name: string;
    /** Gesture label, not class index — see `action-registry.ts`. */
    gestureName: string;
    /**
     * The confirming signal. At least one must be set; either one satisfies.
     *
     * Two routes to the same action rather than one, because the alternative is
     * that anyone who cannot speak — non-verbal, in a shared room, on a call —
     * loses every fused action on the product whose purpose is accessible
     * input. A facial marker is the same confirmation by another channel.
     */
    voiceIntent?: VoiceIntent;
    facialMarker?: FacialMarker;
    execute: () => void;
    feedbackType: "success" | "error" | "light";
}

interface TemporalIntent {
    intent: VoiceIntent;
    timestamp: number;
}

interface TemporalMarker {
    marker: FacialMarker;
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
    private faceBuffer: TemporalMarker[] = [];
    private gestureBuffer: TemporalGesture[] = [];
    
    /** Last intent the caller reported, so a held one is not re-buffered. */
    private lastVoiceSeen: VoiceIntent | null = null;
    /** Same for the face: a held expression is one marker, not sixty. */
    private lastFaceSeen: FacialMarker | null = null;

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
     *
     * Every default carries both a spoken and a facial confirmation, so none of
     * them is reachable only by speaking.
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
            // Furrowed brows: the wh-question marker in ASL, and the same face
            // most people already make for "no, stop".
            facialMarker: "brow_furrow",
            feedbackType: "error",
            execute: run("sys_lock"),
        });

        this.register({
            id: "fusion_confirm",
            name: "Confirm",
            gestureName: "ok",
            voiceIntent: "confirm",
            // Raised brows: the yes/no question marker, and the assenting face.
            facialMarker: "brow_raise",
            feedbackType: "success",
            execute: run("media_toggle"),
        });

        this.register({
            id: "fusion_reset",
            name: "Reset tracking",
            gestureName: "palm",
            voiceIntent: "reset",
            // Deliberately the least natural of the three, because resetting
            // tracking should take an expression nobody wears by accident.
            facialMarker: "mouth_open",
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
        this.faceBuffer = [];
        this.gestureBuffer = [];
        this.lastVoiceSeen = null;
        this.lastFaceSeen = null;
        this.lastTriggerTime = 0;
        this.fusedActions = [];
        this.initializeDefaults();
    }

    /**
     * Probabilistically fuses gestures and voice intents within a temporal window.
     */
    public process(
        gesture: GestureResult,
        voice: VoiceIntent | null,
        face: FacialMarker | null = null,
    ): FusedAction | null {
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

        // Edge-triggered for the same reason as voice: an expression is held
        // across many frames, and each frame would otherwise be a fresh vote.
        if (face && face !== this.lastFaceSeen) {
            this.faceBuffer.push({ marker: face, timestamp: now });
        }
        this.lastFaceSeen = face;
        // Onsets only. Buffering every frame of a held pose would fill the
        // window with 60 copies of the same gesture and make the "consume both
        // triggers" step below meaningless.
        if (gesture.phase === "onset" && !gesture.rejected) {
            this.gestureBuffer.push({ gestureName: gesture.gestureName, timestamp: now });
        }

        // 3. Purge Stale Data
        this.voiceBuffer = this.voiceBuffer.filter(v => now - v.timestamp < this.FUSION_WINDOW_MS);
        this.faceBuffer = this.faceBuffer.filter(f => now - f.timestamp < this.FUSION_WINDOW_MS);
        this.gestureBuffer = this.gestureBuffer.filter(g => now - g.timestamp < this.FUSION_WINDOW_MS);

        if (this.gestureBuffer.length === 0) return null;
        if (this.voiceBuffer.length === 0 && this.faceBuffer.length === 0) return null;

        // 4. Fusion Matcher (Cross-Product search within windows)
        for (const action of this.fusedActions) {
            const gestureMatch = this.gestureBuffer.find(g => g.gestureName === action.gestureName);
            if (!gestureMatch) continue;

            // Either channel confirms. Voice is checked first only because it
            // is the more deliberate of the two, not because it counts more.
            const voiceMatch = action.voiceIntent
                ? this.voiceBuffer.find(v => v.intent === action.voiceIntent)
                : undefined;
            const faceMatch =
                !voiceMatch && action.facialMarker
                    ? this.faceBuffer.find(f => f.marker === action.facialMarker)
                    : undefined;

            if (voiceMatch || faceMatch) {
                // No separation check here: the purge above already dropped
                // anything older than FUSION_WINDOW_MS, so both survivors are
                // within that of now and therefore within it of each other.
                // The explicit `timeDiff < FUSION_WINDOW_MS` that used to sit
                // here could not fail, which made it read as a safety
                // condition while enforcing nothing.

                // Consume both triggers so neither can fire a second action.
                if (voiceMatch) this.voiceBuffer = this.voiceBuffer.filter(v => v !== voiceMatch);
                if (faceMatch) this.faceBuffer = this.faceBuffer.filter(f => f !== faceMatch);
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
