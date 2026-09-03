"use client";

import { useEffect, useState } from "react";
import { type FusedAction } from "@/lib/intent-refinery";

interface FusionMonitorProps {
    lastFusedAction: FusedAction | null;
}

/**
 * History of fused gesture+voice actions.
 *
 * Each row is one dispatch that required both modalities to agree — the gesture
 * that matched and the spoken intent that confirmed it.
 */
export function FusionMonitor({ lastFusedAction }: FusionMonitorProps) {
    const [history, setHistory] = useState<{ action: FusedAction; timestamp: number }[]>([]);

    useEffect(() => {
        if (lastFusedAction) {
            setHistory((prev) =>
                [{ action: lastFusedAction, timestamp: Date.now() }, ...prev].slice(0, 5)
            );
        }
    }, [lastFusedAction]);

    if (history.length === 0) {
        return (
            <p className="label leading-relaxed">
                Waiting for a gesture and a spoken intent to coincide.
            </p>
        );
    }

    return (
        <ul className="space-y-2">
            {history.map((entry, i) => (
                <li
                    key={`${entry.timestamp}-${i}`}
                    className="border-l border-[var(--signal-2)] pl-3"
                >
                    <div className="flex items-baseline justify-between gap-3">
                        <span className="mono text-[11px] text-[var(--signal)]">
                            {entry.action.name}
                        </span>
                        <span className="readout text-[10px] text-[var(--ink-4)]">
                            {new Date(entry.timestamp).toLocaleTimeString([], { hour12: false })}
                        </span>
                    </div>
                    <div className="mt-1 flex items-center gap-2">
                        <span className="mono text-[10px] text-[var(--ink-3)]">
                            {entry.action.gestureName.replace(/_/g, " ")}
                        </span>
                        <span className="text-[10px] text-[var(--ink-4)]">+</span>
                        <span className="mono text-[10px] text-[var(--ink-3)]">
                            &ldquo;{entry.action.voiceIntent}&rdquo;
                        </span>
                    </div>
                </li>
            ))}
        </ul>
    );
}
