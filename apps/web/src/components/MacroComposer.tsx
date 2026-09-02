"use client";

import { useState, useEffect } from "react";
import { type GestureResult } from "@/lib/gesture-engine";
import { macroEngine, type MacroPattern } from "@/lib/macro-engine";

interface MacroComposerProps {
    gesture: GestureResult | null;
    onClose: () => void;
}

const GESTURE_LABELS = [
    "none", "open_palm", "closed_fist", "thumbs_up", "thumbs_down",
    "peace", "pointing_up", "ok_sign", "pinch", "wave"
];

export function MacroComposer({ gesture, onClose }: MacroComposerProps) {
    const [sequence, setSequence] = useState<number[]>([]);
    const [macroName, setMacroName] = useState("");
    const [status, setStatus] = useState<"idle" | "composing" | "saving">("idle");
    const [lastLoggedId, setLastLoggedId] = useState<number | null>(null);

    useEffect(() => {
        if (status === "composing" && gesture && gesture.gestureId !== -1 && gesture.confidence > 0.95) {
            if (gesture.gestureId !== lastLoggedId) {
                setSequence(prev => [...prev, gesture.gestureId].slice(-4)); // Limit to 4 gestures
                setLastLoggedId(gesture.gestureId);
            }
        }
    }, [gesture, status, lastLoggedId]);

    const saveMacro = () => {
        if (sequence.length < 2) {
            alert("Macros require at least 2 gestures.");
            return;
        }
        if (!macroName.trim()) {
            alert("Please enter a macro name.");
            return;
        }

        setStatus("saving");
        setTimeout(() => {
            macroEngine.register({
                id: `macro_${Date.now()}`,
                name: macroName,
                sequence: sequence,
                description: "Custom spatial shortcut.",
                execute: () => {
                    console.log(`DexteraAI: Custom Macro [${macroName}] fired.`);
                    alert(`Macro Executed: ${macroName}`);
                }
            });
            onClose();
        }, 800);
    };

    return (
        <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/80 backdrop-blur-xl animate-in fade-in duration-500">
            <div className="spatial-card relative w-full max-w-2xl rounded-[2.5rem] p-10 border-white/10">
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
                        <span className="text-[10px] font-bold tracking-[0.3em] text-blue-500 uppercase italic">Automation Deck</span>
                    </div>

                    <h2 className="text-3xl font-light tracking-tight text-white mb-4">Macro Composer</h2>
                    <p className="text-sm text-[#86868b] mb-12 max-w-sm">
                        Chain gestures together to create high-frequency system shortcuts.
                    </p>

                    <div className="w-full space-y-10">
                        {/* Sequence Visualizer */}
                        <div className="flex items-center justify-center gap-4 min-h-[100px]">
                            {sequence.length === 0 ? (
                                <div className="text-[10px] font-mono tracking-widest text-white/10 uppercase">Awaiting_Initial_Gesture...</div>
                            ) : (
                                sequence.map((id, i) => (
                                    <div key={i} className="flex items-center gap-4 animate-in zoom-in duration-300">
                                        <div className="flex flex-col items-center gap-2">
                                            <div className="h-14 w-14 rounded-2xl border border-blue-500/20 bg-blue-500/5 flex items-center justify-center">
                                                <span className="text-xs font-bold text-blue-500">{id}</span>
                                            </div>
                                            <span className="text-[9px] font-mono text-[#86868b] uppercase tracking-tighter">
                                                {GESTURE_LABELS[id]}
                                            </span>
                                        </div>
                                        {i < sequence.length - 1 && (
                                            <div className="w-4 h-px bg-white/10" />
                                        )}
                                    </div>
                                ))
                            )}
                        </div>

                        <div className="flex flex-col gap-4">
                            <input
                                type="text"
                                placeholder="MACRO_ID_NAME"
                                value={macroName}
                                onChange={(e) => setMacroName(e.target.value)}
                                className="w-full bg-white/[0.03] border border-white/10 rounded-2xl px-6 py-4 text-sm font-light tracking-widest text-white placeholder:text-white/10 focus:outline-none focus:border-blue-500/50 transition-all uppercase"
                            />

                            <div className="grid grid-cols-2 gap-4">
                                <button
                                    onClick={() => { setStatus("composing"); setSequence([]); setLastLoggedId(null); }}
                                    className={`rounded-2xl border py-4 text-[10px] font-bold uppercase tracking-[0.2em] transition-all ${status === "composing" ? "border-blue-500/50 bg-blue-500/10 text-blue-500" : "border-white/5 bg-white/[0.02] text-white/40 hover:text-white"}`}
                                >
                                    {status === "composing" ? "Recording..." : "New Sequence"}
                                </button>
                                <button
                                    onClick={saveMacro}
                                    disabled={sequence.length < 2 || !macroName}
                                    className="rounded-2xl bg-white py-4 text-[10px] font-bold uppercase tracking-[0.2em] text-black transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-20 disabled:scale-100"
                                >
                                    Save Macro
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="mt-16 pt-8 border-t border-white/[0.03]">
                    <p className="text-[10px] leading-relaxed text-[#86868b] uppercase tracking-wider text-center">
                        Perform gestures in order while "Recording" is active. The engine will detect each unique pose and build your temporal shortcut.
                    </p>
                </div>
            </div>
        </div>
    );
}
