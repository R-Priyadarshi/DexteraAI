"use client";

import { useState, useEffect } from "react";
import { actionRegistry, type GestureAction } from "@/lib/action-registry";

interface ActionMapperProps {
    onClose: () => void;
}

const GESTURE_LABELS = [
    "none", "open_palm", "closed_fist", "thumbs_up", "thumbs_down",
    "peace", "pointing_up", "ok_sign", "pinch", "wave"
];

export function ActionMapper({ onClose }: ActionMapperProps) {
    const [selectedGesture, setSelectedGesture] = useState<number | null>(null);
    const [mappings, setMappings] = useState<Record<number, string>>({});
    const allActions = actionRegistry.getAllActions();

    useEffect(() => {
        const currentMappings = actionRegistry.getMappings();
        setMappings(Object.fromEntries(currentMappings));
    }, []);

    const handleRemap = (actionId: string) => {
        if (selectedGesture !== null) {
            actionRegistry.remap(selectedGesture, actionId);
            setMappings(prev => ({ ...prev, [selectedGesture]: actionId }));
            setSelectedGesture(null);
        }
    };

    return (
        <div className="fixed inset-0 z-[120] flex items-center justify-center bg-black/90 backdrop-blur-2xl animate-in fade-in duration-500">
            <div className="spatial-card relative w-full max-w-4xl rounded-[3rem] p-12 border-white/5">
                <button
                    onClick={onClose}
                    className="absolute top-10 right-10 text-[#86868b] hover:text-white transition-colors"
                >
                    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>

                <div className="flex flex-col h-full">
                    <div className="mb-12">
                        <div className="spatial-panel inline-block rounded-full px-5 py-1.5 mb-6">
                            <span className="text-[10px] font-bold tracking-[0.4em] text-blue-500 uppercase italic">Command Central</span>
                        </div>
                        <h2 className="text-4xl font-light tracking-tight text-white">Action Mapper</h2>
                        <p className="text-sm text-[#86868b] mt-4 max-w-lg">
                            Configure the industrial mapping between neural spatial signatures and system execution logic.
                        </p>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 overflow-hidden h-[500px]">
                        {/* Gestures List */}
                        <div className="space-y-3 overflow-y-auto pr-4 custom-scrollbar">
                            <h4 className="hud-label mb-6">Spatial Signatures</h4>
                            {GESTURE_LABELS.map((label, id) => (
                                id === 0 ? null : (
                                    <button
                                        key={id}
                                        onClick={() => setSelectedGesture(id)}
                                        className={`w-full group flex items-center justify-between p-5 rounded-2xl border transition-all ${selectedGesture === id ? "border-blue-500 bg-blue-500/10" : "border-white/5 bg-white/[0.02] hover:bg-white/[0.05]"}`}
                                    >
                                        <div className="flex items-center gap-5">
                                            <div className={`h-10 w-10 rounded-xl flex items-center justify-center text-xs font-bold ${selectedGesture === id ? "bg-blue-500 text-white" : "bg-white/5 text-white/40"}`}>
                                                {id}
                                            </div>
                                            <span className={`text-sm font-medium tracking-widest uppercase ${selectedGesture === id ? "text-white" : "text-white/60"}`}>
                                                {label.replace("_", " ")}
                                            </span>
                                        </div>
                                        <div className="flex flex-col items-end">
                                            <span className="text-[9px] font-mono text-white/20 uppercase mb-1">Assigned_To</span>
                                            <span className="text-[11px] font-bold text-blue-400 uppercase tracking-tighter">
                                                {mappings[id] ? actionRegistry.getActionById(mappings[id])?.name : "UNASSIGNED"}
                                            </span>
                                        </div>
                                    </button>
                                )
                            ))}
                        </div>

                        {/* Actions List */}
                        <div className={`space-y-3 transition-all duration-500 ${selectedGesture === null ? "opacity-30 pointer-events-none grayscale" : "opacity-100"}`}>
                            <h4 className="hud-label mb-6">Execution Logic {selectedGesture && `for [${GESTURE_LABELS[selectedGesture].toUpperCase()}]`}</h4>
                            <div className="space-y-3 overflow-y-auto h-full pr-4 custom-scrollbar">
                                {allActions.map((action) => (
                                    <button
                                        key={action.id}
                                        onClick={() => handleRemap(action.id)}
                                        className="w-full text-left p-6 rounded-2xl border border-white/5 bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/10 transition-all group"
                                    >
                                        <div className="flex justify-between items-start mb-2">
                                            <span className="text-sm font-bold text-white uppercase tracking-wider group-hover:text-blue-400 transition-colors">{action.name}</span>
                                            <span className="text-[9px] font-mono text-white/20 uppercase tracking-widest">{action.category}</span>
                                        </div>
                                        <p className="text-[11px] text-[#86868b] leading-relaxed uppercase tracking-tighter">
                                            {action.description}
                                        </p>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                <div className="mt-12 pt-8 border-t border-white/[0.03] flex justify-between items-center">
                    <div className="flex items-center gap-4">
                        <div className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-pulse" />
                        <span className="text-[10px] font-mono tracking-[0.2em] text-white/30 uppercase">Neural_Link_Ready</span>
                    </div>
                    <button 
                        onClick={onClose}
                        className="text-[10px] font-bold tracking-[0.3em] text-white/60 hover:text-white uppercase transition-all"
                    >
                        Close_Session
                    </button>
                </div>
            </div>
        </div>
    );
}
