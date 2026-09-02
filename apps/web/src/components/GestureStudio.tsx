"use client";

import { useState, useEffect, useRef } from "react";
import { type GestureResult, type Landmark } from "@/lib/gesture-engine";
import { gestureStore } from "@/lib/gesture-store";

interface GestureStudioProps {
    gesture: GestureResult | null;
    onClose: () => void;
}

const REQUIRED_SAMPLES = 40;

export function GestureStudio({ gesture, onClose }: GestureStudioProps) {
    const [isRecording, setIsRecording] = useState(false);
    const [samples, setSamples] = useState<Landmark[][]>([]);
    const [gestureName, setGestureName] = useState("");
    const [status, setStatus] = useState<"idle" | "recording" | "saving" | "complete">("idle");

    useEffect(() => {
        if (status === "recording" && gesture?.landmarks) {
            if (samples.length < REQUIRED_SAMPLES) {
                setSamples((prev) => [...prev, gesture.landmarks!]);
            } else {
                setStatus("complete");
                setIsRecording(false);
            }
        }
    }, [gesture, status, samples]);

    const startRecording = () => {
        if (!gestureName.trim()) {
            alert("Please enter a gesture name first.");
            return;
        }
        setSamples([]);
        setStatus("recording");
        setIsRecording(true);
    };

    const saveGesture = () => {
        if (samples.length >= REQUIRED_SAMPLES) {
            setStatus("saving");
            setTimeout(() => {
                gestureStore.addGesture(gestureName, samples);
                setStatus("idle");
                setGestureName("");
                setSamples([]);
                onClose();
            }, 1000);
        }
    };

    return (
        <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/80 backdrop-blur-xl animate-in fade-in duration-500">
            <div className="spatial-card relative w-full max-w-xl rounded-[2.5rem] p-10 border-white/10">
                <button
                    onClick={onClose}
                    className="absolute top-8 right-8 text-[#86868b] hover:text-white transition-colors"
                >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>

                <div className="flex flex-col items-center text-center">
                    <div className="spatial-panel rounded-full px-4 py-1 mb-6">
                        <span className="text-[10px] font-bold tracking-[0.3em] text-blue-500 uppercase italic">Experimental Lab</span>
                    </div>

                    <h2 className="text-3xl font-light tracking-tight text-white mb-4">Gesture Studio</h2>
                    <p className="text-sm text-[#86868b] mb-12 max-w-sm">
                        Capture a new spatial signature to expand DexteraAI's biometric library.
                    </p>

                    <div className="w-full space-y-8">
                        <div className="relative group">
                            <input
                                type="text"
                                placeholder="GESTURE_ID_NAME"
                                value={gestureName}
                                onChange={(e) => setGestureName(e.target.value)}
                                disabled={status !== "idle"}
                                className="w-full bg-white/[0.03] border border-white/10 rounded-2xl px-6 py-4 text-sm font-light tracking-widest text-white placeholder:text-white/10 focus:outline-none focus:border-blue-500/50 transition-all uppercase"
                            />
                            <div className="absolute right-4 top-1/2 -translate-y-1/2">
                                <div className={`h-1.5 w-1.5 rounded-full ${gestureName ? "bg-blue-500" : "bg-white/10"}`} />
                            </div>
                        </div>

                        {status === "idle" && (
                            <button
                                onClick={startRecording}
                                className="w-full relative overflow-hidden group rounded-2xl bg-white px-8 py-5 transition-transform active:scale-[0.98]"
                            >
                                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-black/[0.05] to-transparent animate-shimmer opacity-0 group-hover:opacity-100" />
                                <span className="text-xs font-bold tracking-[0.3em] text-black uppercase">Start Capture</span>
                            </button>
                        )}

                        {(status === "recording" || status === "complete") && (
                            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2">
                                <div className="relative h-2 w-full rounded-full bg-white/5 overflow-hidden">
                                    <div
                                        className="h-full bg-blue-500 transition-all duration-300"
                                        style={{ width: `${(samples.length / REQUIRED_SAMPLES) * 100}%` }}
                                    />
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-[10px] font-mono tracking-widest text-white/30 uppercase">
                                        {status === "recording" ? "Collecting_Samples..." : "Capture_Complete"}
                                    </span>
                                    <span className="text-[10px] font-mono text-blue-500">
                                        {samples.length}/{REQUIRED_SAMPLES}__FRM
                                    </span>
                                </div>
                            </div>
                        )}

                        {status === "complete" && (
                            <button
                                onClick={saveGesture}
                                className="w-full relative overflow-hidden group rounded-2xl border border-blue-500/30 bg-blue-500/10 px-8 py-5 transition-transform active:scale-[0.98] animate-in fade-in"
                            >
                                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-500/10 to-transparent animate-shimmer" />
                                <span className="text-xs font-bold tracking-[0.3em] text-blue-500 uppercase">Commit to Local Vault</span>
                            </button>
                        )}

                        {status === "saving" && (
                            <div className="flex flex-col items-center gap-4 py-4">
                                <div className="h-2 w-2 rounded-full bg-blue-500 animate-ping" />
                                <span className="text-[10px] font-mono tracking-[0.3em] text-white/40 uppercase animate-pulse">Synchronizing_Signatures</span>
                            </div>
                        )}
                    </div>
                </div>

                <div className="mt-16 pt-8 border-t border-white/[0.03]">
                    <div className="flex items-start gap-4">
                        <div className="mt-1 p-2 rounded-lg bg-blue-500/10">
                            <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                        </div>
                        <p className="text-[10px] leading-relaxed text-[#86868b] uppercase tracking-wider">
                            Move your hand naturally while recording. The engine captures 40 spatial snapshots to create a high-fidelity biometric template.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
