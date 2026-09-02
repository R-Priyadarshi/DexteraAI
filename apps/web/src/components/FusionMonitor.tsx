"use client";

import { useEffect, useState } from "react";
import { type FusedAction } from "@/lib/intent-refinery";

interface FusionMonitorProps {
    lastFusedAction: FusedAction | null;
}

export function FusionMonitor({ lastFusedAction }: FusionMonitorProps) {
    const [history, setHistory] = useState<{ action: FusedAction; timestamp: number }[]>([]);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    useEffect(() => {
        if (lastFusedAction) {
            setHistory((prev) => [{ action: lastFusedAction, timestamp: Date.now() }, ...prev].slice(0, 5));
        }
    }, [lastFusedAction]);

    return (
        <div className="spatial-card rounded-3xl p-6 border-white/[0.02]">
            <h4 className="hud-label mb-6 flex items-center justify-between">
                <span>Fusion Monitor</span>
                <div className="flex gap-1">
                    <div className="h-1 w-1 rounded-full bg-blue-500 animate-pulse" />
                    <div className="h-1 w-1 rounded-full bg-purple-500 animate-pulse delay-75" />
                </div>
            </h4>

            <div className="space-y-4">
                {history.map((entry, i) => (
                    <div key={i} className="flex flex-col gap-2 rounded-xl bg-white/[0.02] p-3 border border-white/[0.05] animate-in fade-in slide-in-from-right-2 duration-500">
                        <div className="flex items-center justify-between">
                            <span className="text-[10px] font-bold text-blue-500 uppercase tracking-widest">{entry.action.name}</span>
                            <span className="text-[8px] font-mono text-white/20">{new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
                        </div>
                        <div className="flex items-center gap-3">
                            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-white/5 border border-white/5">
                                <span className="text-[8px] text-white/40 uppercase">Gesture</span>
                                <span className="text-[9px] font-mono text-white/80">{entry.action.gestureId}</span>
                            </div>
                            <span className="text-[9px] text-white/20">+</span>
                            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-white/5 border border-white/5">
                                <span className="text-[8px] text-white/40 uppercase">Voice</span>
                                <span className="text-[9px] font-mono text-white/80">{entry.action.voiceIntent}</span>
                            </div>
                        </div>
                    </div>
                ))}

                {history.length === 0 && (
                    <div className="flex flex-col items-center justify-center py-10 opacity-20 text-center relative overflow-hidden group">
                        <div className="h-10 w-10 rounded-full border border-dashed border-cyan-500/50 mb-4 animate-spin-slow" />
                        <span className="text-[9px] uppercase tracking-[0.5em] animate-pulse italic text-cyan-400">Awaiting_Neural_Sync...</span>
                        
                        {/* Live Bitstream Visualizer */}
                        <div className="absolute bottom-0 left-0 w-full h-8 flex items-end gap-[1px] opacity-20 pointer-events-none">
                            {mounted && Array.from({ length: 50 }).map((_, i) => (
                                <div 
                                    key={i}
                                    className="flex-1 bg-cyan-500"
                                    style={{ 
                                        height: `${Math.sin((Date.now() / 1000) + i) * 50 + 50}%`,
                                        transition: 'height 0.2s ease-in-out'
                                    }}
                                />
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
