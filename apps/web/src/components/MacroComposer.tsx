"use client";

import { useEffect, useState } from "react";
import { actionRegistry } from "@/lib/action-registry";
import { type GestureResult } from "@/lib/gesture-engine";
import { macroEngine } from "@/lib/macro-engine";

interface MacroComposerProps {
    gesture: GestureResult | null;
    onClose: () => void;
}

const MAX_STEPS = 4;

/**
 * Record a gesture sequence and bind it to an action.
 *
 * Steps are captured on segment onsets, so holding a pose contributes exactly
 * one step. The previous version sampled raw frames and de-duplicated by hand,
 * which meant a slow performer added the same gesture twice and a fast one
 * dropped a step entirely.
 */
export function MacroComposer({ gesture, onClose }: MacroComposerProps) {
    const [sequence, setSequence] = useState<string[]>([]);
    const [name, setName] = useState("");
    const [actionId, setActionId] = useState("");
    const [recording, setRecording] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const actions = actionRegistry.getAllActions();

    useEffect(() => {
        if (!recording || !gesture) return;
        if (gesture.phase !== "onset" || gesture.rejected) return;

        setSequence((prev) => {
            if (prev.length >= MAX_STEPS) return prev;
            return [...prev, gesture.gestureName];
        });
    }, [gesture, recording]);

    // Stop automatically once the sequence is full, so the last step isn't
    // immediately overwritten by whatever the hand does next.
    useEffect(() => {
        if (sequence.length >= MAX_STEPS) setRecording(false);
    }, [sequence.length]);

    const save = () => {
        if (sequence.length < 2) {
            setError("A macro needs at least two gestures.");
            return;
        }
        if (!name.trim()) {
            setError("Give the macro a name.");
            return;
        }
        if (!actionId) {
            setError("Choose the action this macro should run.");
            return;
        }
        const created = macroEngine.defineMacro(name.trim(), sequence, actionId);
        if (!created) {
            setError("Could not create the macro.");
            return;
        }
        onClose();
    };

    return (
        <div className="modal-backdrop fixed inset-0 z-[110] flex items-center justify-center p-6">
            <div className="panel-raised flex max-h-[86vh] w-full max-w-2xl flex-col">
                <header className="flex items-baseline justify-between border-b border-[var(--rule)] px-8 py-6">
                    <div>
                        <span className="label">Sequences</span>
                        <h2 className="display mt-2 text-2xl text-[var(--ink)]">Macro composer</h2>
                    </div>
                    <button onClick={onClose} className="label hover:text-[var(--ink)]">
                        Close
                    </button>
                </header>

                <div className="min-h-0 flex-1 overflow-y-auto px-8 py-6">
                    <p className="max-w-md text-xs leading-relaxed text-[var(--ink-2)]">
                        Chain up to {MAX_STEPS} gestures into one command. A sequence is far
                        harder to trigger by accident than a single pose, which is what makes
                        it the right binding for anything you would not want fired by a
                        stray hand.
                    </p>

                    {/* Captured sequence */}
                    <div className="mt-8">
                        <div className="flex items-baseline justify-between border-b border-[var(--rule)] pb-3">
                            <span className="label">Sequence</span>
                            <span className="readout text-[10px] text-[var(--ink-3)]">
                                {sequence.length} / {MAX_STEPS}
                            </span>
                        </div>

                        <div className="flex min-h-[76px] flex-wrap items-center gap-3 py-5">
                            {sequence.length === 0 ? (
                                <span className="label">
                                    {recording ? "Waiting for a gesture…" : "Nothing recorded"}
                                </span>
                            ) : (
                                sequence.map((label, i) => (
                                    <div key={i} className="flex items-center gap-3">
                                        <div className="border border-[var(--signal-2)] bg-[var(--signal-4)] px-3 py-2">
                                            <span className="mono text-[11px] text-[var(--signal)]">
                                                {label.replace(/_/g, " ")}
                                            </span>
                                        </div>
                                        {i < sequence.length - 1 && (
                                            <span className="text-[var(--ink-4)]">→</span>
                                        )}
                                    </div>
                                ))
                            )}
                        </div>

                        <div className="flex gap-3">
                            <button
                                onClick={() => {
                                    setError(null);
                                    setSequence([]);
                                    setRecording(true);
                                }}
                                className={`btn ${recording ? "btn-signal" : ""}`}
                            >
                                {recording ? "Recording" : "Record sequence"}
                            </button>
                            {recording && (
                                <button onClick={() => setRecording(false)} className="btn">
                                    Stop
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Name */}
                    <div className="mt-10">
                        <span className="label">Name</span>
                        <input
                            type="text"
                            value={name}
                            onChange={(e) => {
                                setName(e.target.value);
                                setError(null);
                            }}
                            placeholder="e.g. Mute everything"
                            className="mono mt-3 w-full border border-[var(--rule-strong)] bg-[var(--field-1)] px-4 py-3 text-xs text-[var(--ink)] placeholder:text-[var(--ink-4)] focus:border-[var(--signal)] focus:outline-none"
                        />
                    </div>

                    {/* Action */}
                    <div className="mt-8">
                        <span className="label">Runs</span>
                        <div className="mt-3">
                            {actions.map((a) => (
                                <button
                                    key={a.id}
                                    onClick={() => {
                                        setActionId(a.id);
                                        setError(null);
                                    }}
                                    className="flex w-full items-baseline justify-between gap-4 border-b border-[var(--rule-2)] py-3 text-left"
                                >
                                    <span
                                        className="mono text-xs"
                                        style={{
                                            color:
                                                actionId === a.id ? "var(--signal)" : "var(--ink)",
                                        }}
                                    >
                                        {a.name}
                                    </span>
                                    <span className="label">{a.category}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                <footer className="flex items-center justify-between gap-4 border-t border-[var(--rule)] px-8 py-5">
                    <span
                        className="label"
                        style={{ color: error ? "var(--alert)" : "var(--ink-3)" }}
                    >
                        {error ?? "Stored on this device only"}
                    </span>
                    <button
                        onClick={save}
                        disabled={sequence.length < 2 || !name.trim() || !actionId}
                        className="btn btn-solid disabled:cursor-not-allowed disabled:opacity-25"
                    >
                        Save macro
                    </button>
                </footer>
            </div>
        </div>
    );
}
