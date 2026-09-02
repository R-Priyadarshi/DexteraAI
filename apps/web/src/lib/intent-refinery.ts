"use client";

import { type GestureResult } from "./gesture-engine";
import { type VoiceIntent } from "./voice-engine";
import { hapticEngine } from "./haptic-engine";

export interface FusedAction {
    id: string;
    name: string;
    gestureId: number;
    voiceIntent: VoiceIntent;
    execute: () => void;
    feedbackType: "success" | "error" | "light";
}

interface TemporalIntent {
    intent: VoiceIntent;
    timestamp: number;
}

interface TemporalGesture {
    gestureId: number;
    timestamp: number;
}

/**
 * IntentRefinery — Probabilistic Multi-modal Fusion Engine.
 * 
 * Instead of instantaneous matching, this engine uses a temporal window
 * (typically 2 seconds) to fuse voice commands and physical gestures.
 * 
 * This allows for more natural interaction: e.g., saying "Abort" and 
 * then performing a peace sign within 1.5 seconds.
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

    private initializeDefaults() {
        // Emergency Abort: Peace Sign (5) + "Abort"
        this.register({
            id: "fusion_abort",
            name: "Emergency Abort",
            gestureId: 5,
            voiceIntent: "abort",
            feedbackType: "error",
            execute: () => {
                console.log("IntentRefinery: EMERGENCY ABORT EXECUTED.");
                // Dispatch system-wide halt
                window.dispatchEvent(new CustomEvent("dextera_sys_halt"));
            }
        });

        // Stealth Mode: Closed Fist (2) + "Stealth" (Redirect Disabled)
        this.register({
            id: "fusion_stealth",
            name: "Visual Stealth",
            gestureId: 2,
            voiceIntent: "stealth",
            feedbackType: "success",
            execute: () => {
                console.log("DexteraAI: Stealth redirect disabled for industrial continuity.");
            }
        });

        // Mission Launch: Thumbs Up (3) + "Launch"
        this.register({
            id: "fusion_launch",
            name: "Industrial Launch",
            gestureId: 3,
            voiceIntent: "launch",
            feedbackType: "success",
            execute: () => {
                console.log("IntentRefinery: MISSION LAUNCH SEQUENCE START.");
            }
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
        if (gesture.gestureId !== -1 && gesture.confidence > 0.9) {
            this.gestureBuffer.push({ gestureId: gesture.gestureId, timestamp: now });
        }

        // 3. Purge Stale Data
        this.voiceBuffer = this.voiceBuffer.filter(v => now - v.timestamp < this.FUSION_WINDOW_MS);
        this.gestureBuffer = this.gestureBuffer.filter(g => now - g.timestamp < this.FUSION_WINDOW_MS);

        if (this.voiceBuffer.length === 0 || this.gestureBuffer.length === 0) return null;

        // 4. Fusion Matcher (Cross-Product search within windows)
        for (const action of this.fusedActions) {
            const voiceMatch = this.voiceBuffer.find(v => v.intent === action.voiceIntent);
            const gestureMatch = this.gestureBuffer.find(g => g.gestureId === action.gestureId);

            if (voiceMatch && gestureMatch) {
                // Determine which came first (for potentially different logic, currently just fuse)
                const timeDiff = Math.abs(voiceMatch.timestamp - gestureMatch.timestamp);
                
                if (timeDiff < this.FUSION_WINDOW_MS) {
                    console.log(`IntentRefinery: Fusion Successful [${action.name}] (dt: ${timeDiff}ms)`);
                    
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
