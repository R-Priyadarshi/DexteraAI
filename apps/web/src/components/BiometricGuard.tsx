"use client";

import { useState, useEffect, useRef } from "react";
import { type GestureResult } from "@/lib/gesture-engine";

interface BiometricGuardProps {
    gesture: GestureResult | null;
    /**
     * Gesture labels forming the unlock sequence, in order.
     *
     * Passed in rather than hardcoded. This previously held class *indices*
     * against the built-in fallback vocabulary, so under the shipped HaGRID
     * bundle it displayed "PEACE_SIGN / CLOSED_FIST / THUMBS_UP" while actually
     * requiring ids 5, 2 and 3 — which are `mute`, `fist` and `four` there. The
     * guard asked for one thing and accepted another, and following its own
     * instructions never unlocked it.
     */
    sequence: string[];
    onUnlocked: () => void;
}

const CONFIDENCE_THRESHOLD = 0.6;

/**
 * Gestures preferred for the unlock signature, most-preferred first.
 *
 * Chosen to be visually distinct from one another, so a hand mid-transition
 * between two of them is unlikely to satisfy the next step by accident.
 */
export const PREFERRED_SIGNATURE = [
    "peace",
    "fist",
    "like",
    "palm",
    "ok",
    "one",
    "stop",
];

export const SIGNATURE_LENGTH = 3;

/**
 * Pick an unlock signature from the vocabulary that is actually loaded.
 *
 * Only labels the bundle contains are used. A signature naming a gesture the
 * active model can never produce would make the guard impossible to open — the
 * exact failure the previous index-keyed version had, where it displayed
 * "PEACE_SIGN / CLOSED_FIST / THUMBS_UP" while requiring `mute`, `fist` and
 * `four`.
 */
export function signatureFor(vocabulary: string[]): string[] {
    const available = PREFERRED_SIGNATURE.filter((g) => vocabulary.includes(g));
    // Bundles sharing no labels with the preferred list — the ASL alphabet, for
    // one — fall back to the head of their own vocabulary.
    const chosen = available.length >= SIGNATURE_LENGTH ? available : vocabulary;
    return chosen.slice(0, SIGNATURE_LENGTH);
}

/**
 * How long each pose must be held.
 *
 * Long enough to be deliberate, so a hand passing through a pose on its way
 * somewhere else does not advance the sequence.
 */
const HOLD_TIME_MS = 600;

export function BiometricGuard({ gesture, sequence, onUnlocked }: BiometricGuardProps) {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [holdProgress, setHoldProgress] = useState(0);
    const [isVerifying, setIsVerifying] = useState(false);
    const holdStartRef = useRef<number | null>(null);

    /**
     * Latched once the final step completes.
     *
     * `isVerifying` cannot do this job: state updates are asynchronous, so the
     * next camera frame re-enters this effect before the re-render lands, with
     * `holdStartRef` still set and `currentIndex` unchanged. Every frame of the
     * one-second verify delay therefore scheduled another `onUnlocked` — about
     * thirty calls for a single successful unlock.
     */
    const unlockedRef = useRef(false);
    const unlockTimerRef = useRef<number | null>(null);

    // Clear a pending unlock if the guard unmounts mid-verify.
    useEffect(
        () => () => {
            if (unlockTimerRef.current !== null) {
                window.clearTimeout(unlockTimerRef.current);
            }
        },
        []
    );

    useEffect(() => {
        if (!gesture || gesture.rejected || gesture.gestureId === -1) {
            holdStartRef.current = null;
            setHoldProgress(0);
            return;
        }

        const target = sequence[currentIndex];

        if (gesture.gestureName === target && gesture.confidence >= CONFIDENCE_THRESHOLD) {
            if (holdStartRef.current === null) {
                holdStartRef.current = Date.now();
            }

            const elapsed = Date.now() - holdStartRef.current;
            const progress = Math.min((elapsed / HOLD_TIME_MS) * 100, 100);
            setHoldProgress(progress);

            if (elapsed >= HOLD_TIME_MS) {
                if (currentIndex === sequence.length - 1) {
                    if (unlockedRef.current) return;
                    unlockedRef.current = true;
                    setIsVerifying(true);
                    unlockTimerRef.current = window.setTimeout(() => onUnlocked(), 1000);
                } else {
                    setCurrentIndex((prev) => prev + 1);
                    holdStartRef.current = null;
                    setHoldProgress(0);
                }
            }
        } else {
            holdStartRef.current = null;
            setHoldProgress(0);
        }
    }, [gesture, currentIndex, sequence, onUnlocked]);

    return (
        <div className="absolute inset-0 z-[100] flex items-center justify-center bg-transparent animate-in fade-in duration-1000">
            <div className="absolute inset-0 neural-grid opacity-10" />

            <div className="relative flex w-full max-w-2xl flex-col items-center text-center px-6">
                {/* Security Hexagon Icon Simulation */}
                <div className="mb-12 relative">
                    <div className="h-24 w-24 rounded-2xl border border-[var(--signal)] flex items-center justify-center rotate-45 ">
                        <div className="h-px w-12 bg-[var(--signal)] -rotate-45" />
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className={`h-2 w-2 rounded-full bg-[var(--signal)] ${isVerifying ? "scale-[10] opacity-0" : "scale-1"} transition-all duration-1000`} />
                    </div>
                </div>

                <div className="space-y-4 mb-20">
                    <span className="text-[10px] font-bold tracking-[0.4em] text-[var(--signal)] uppercase">Tracking Secured</span>
                    <h2 className="text-4xl font-light tracking-tight text-[#f5f5f7]">
                        {isVerifying ? "Authorization Granted" : "Perform Calibration"}
                    </h2>
                    <p className="text-sm text-[#86868b] tracking-wide font-light max-w-sm mx-auto">
                        Provide the temporal biometric signature to unlock industrial command controls.
                    </p>
                </div>

                {/* Signature Progress HUD */}
                <div
                    className="grid gap-8 w-full"
                    style={{
                        // Driven by the sequence length rather than fixed at
                        // three, so a shorter or longer signature still lays out.
                        gridTemplateColumns: `repeat(${sequence.length}, minmax(0, 1fr))`,
                    }}
                >
                    {sequence.map((label, i) => (
                        <div key={i} className="flex flex-col items-center gap-4">
                            <div className={`relative h-1 w-full rounded-full overflow-hidden bg-white/5`}>
                                <div
                                    className={`h-full bg-[var(--signal)] transition-all duration-300 ${i < currentIndex ? "w-full" : (i === currentIndex ? "w-0" : "w-0")}`}
                                    style={{ width: i === currentIndex ? `${holdProgress}%` : (i < currentIndex ? "100%" : "0%") }}
                                />
                            </div>
                            <div className="flex flex-col items-center gap-1">
                                <span className={`text-[9px] font-mono tracking-widest uppercase transition-colors duration-500 ${i <= currentIndex ? "text-white" : "text-white/10"}`}>
                                    Step_0{i + 1}
                                </span>
                                <span className={`text-[10px] font-bold tracking-tighter uppercase transition-colors duration-500 ${i <= currentIndex ? "text-[var(--signal)]" : "text-white/5"}`}>
                                    {label.replace(/_/g, " ")}
                                </span>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Real-time Status & Telemetry */}
                <div className="mt-20 w-full max-w-xs space-y-6">
                    <div className="flex items-center justify-center gap-3">
                        <div className="h-1 w-1 rounded-full bg-[var(--signal)] animate-pulse" />
                        <span className="text-[10px] font-mono tracking-[0.2em] text-[#86868b] uppercase">
                            {isVerifying ? "Synching_NeuralNet..." :
                                (gesture && gesture.gestureId !== -1 && gesture.confidence > 0.4 ?
                                    `Detected: ${gesture.gestureName.toUpperCase()} (${(gesture.confidence * 100).toFixed(0)}%)` :
                                    "Awaiting_Handshake")}
                        </span>
                    </div>

                    {/* Industrial Biometric HUD Overlay */}
                    {gesture && gesture.landmarks && (
                        <div className="grid grid-cols-5 gap-2 px-4 py-3 rounded-xl border border-white/5 bg-white/[0.02]">
                            {["T", "I", "M", "R", "P"].map((label, idx) => {
                                const tips = [4, 8, 12, 16, 20];
                                const pips = [3, 6, 10, 14, 18];
                                const mcps = [2, 5, 9, 13, 17];
                                
                                // Simple 2D curl approximation for the HUD
                                const tip = gesture.landmarks![tips[idx]];
                                const pip = gesture.landmarks![pips[idx]];
                                const mcp = gesture.landmarks![mcps[idx]];
                                
                                const d1 = Math.sqrt((tip.x-mcp.x)**2 + (tip.y-mcp.y)**2);
                                const d2 = Math.sqrt((mcp.x-pip.x)**2 + (mcp.y-pip.y)**2) + Math.sqrt((pip.x-tip.x)**2 + (pip.y-tip.y)**2);
                                const curl = Math.max(0, Math.min(1, 1 - (d1 / d2)));

                                return (
                                    <div key={idx} className="flex flex-col items-center gap-2">
                                        <div className="h-12 w-1 bg-white/5 rounded-full relative">
                                            <div 
                                                className="absolute bottom-0 left-0 w-full bg-[var(--signal-3)] rounded-full transition-all duration-300"
                                                style={{ height: `${curl * 100}%` }}
                                            />
                                        </div>
                                        <span className="text-[8px] font-bold text-white/40">{label}</span>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
