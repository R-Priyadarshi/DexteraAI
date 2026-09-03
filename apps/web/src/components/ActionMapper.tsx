"use client";

import { useEffect, useState } from "react";
import { actionRegistry } from "@/lib/action-registry";

interface ActionMapperProps {
    /**
     * Labels of the model bundle currently loaded. Passed in rather than
     * hardcoded: the vocabulary differs per bundle (18 general gestures vs 26
     * ASL letters), and a fixed list here would let a user bind gestures the
     * running model cannot produce.
     */
    vocabulary: string[];
    onClose: () => void;
}

/**
 * Bind gestures to actions.
 *
 * Bindings are keyed by label, so they survive a bundle swap intact instead of
 * silently re-pointing at whatever class happens to hold the same index.
 */
export function ActionMapper({ vocabulary, onClose }: ActionMapperProps) {
    const [selected, setSelected] = useState<string | null>(null);
    const [mappings, setMappings] = useState<Record<string, string>>({});

    const allActions = actionRegistry.getAllActions();

    useEffect(() => {
        setMappings(Object.fromEntries(actionRegistry.getMappings()));
    }, []);

    const bind = (actionId: string | null) => {
        if (!selected) return;
        actionRegistry.remap(selected, actionId);
        setMappings((prev) => {
            const next = { ...prev };
            if (actionId === null) delete next[selected];
            else next[selected] = actionId;
            return next;
        });
        setSelected(null);
    };

    return (
        <div className="modal-backdrop fixed inset-0 z-[120] flex items-center justify-center p-6">
            <div className="panel-raised relative flex max-h-[86vh] w-full max-w-5xl flex-col">
                <header className="flex items-baseline justify-between border-b border-[var(--rule)] px-8 py-6">
                    <div>
                        <span className="label">Bindings</span>
                        <h2 className="display mt-2 text-2xl text-[var(--ink)]">Mapper</h2>
                    </div>
                    <button onClick={onClose} className="label hover:text-[var(--ink)]">
                        Close
                    </button>
                </header>

                <p className="border-b border-[var(--rule)] px-8 py-4 text-xs leading-relaxed text-[var(--ink-2)]">
                    Bindings fire once when a gesture starts, not on every frame it is held.
                    They are stored by label, so they follow the gesture rather than its
                    position in the current model.
                </p>

                <div className="grid min-h-0 flex-1 grid-cols-1 gap-px overflow-hidden bg-[var(--rule)] md:grid-cols-2">
                    {/* Vocabulary */}
                    <div className="flex min-h-0 flex-col bg-[var(--field-2)]">
                        <div className="flex items-baseline justify-between border-b border-[var(--rule)] px-6 py-3">
                            <span className="label">Vocabulary</span>
                            <span className="readout text-[10px] text-[var(--ink-3)]">
                                {vocabulary.length}
                            </span>
                        </div>
                        <div className="min-h-0 flex-1 overflow-y-auto px-6 py-2">
                            {vocabulary.length === 0 ? (
                                <p className="label py-6">Start the console to load a vocabulary</p>
                            ) : (
                                vocabulary.map((label) => {
                                    const bound = mappings[label];
                                    const isSel = selected === label;
                                    return (
                                        <button
                                            key={label}
                                            onClick={() => setSelected(isSel ? null : label)}
                                            className="flex w-full items-baseline justify-between gap-4 border-b border-[var(--rule-2)] py-3 text-left"
                                            style={{
                                                borderLeft: isSel
                                                    ? "2px solid var(--signal)"
                                                    : "2px solid transparent",
                                                paddingLeft: 10,
                                            }}
                                        >
                                            <span
                                                className="mono text-xs"
                                                style={{ color: isSel ? "var(--signal)" : "var(--ink)" }}
                                            >
                                                {label.replace(/_/g, " ")}
                                            </span>
                                            <span
                                                className="readout text-[10px]"
                                                style={{
                                                    color: bound ? "var(--ink-2)" : "var(--ink-4)",
                                                }}
                                            >
                                                {bound
                                                    ? actionRegistry.getActionById(bound)?.name ?? bound
                                                    : "unbound"}
                                            </span>
                                        </button>
                                    );
                                })
                            )}
                        </div>
                    </div>

                    {/* Actions */}
                    <div
                        className="flex min-h-0 flex-col bg-[var(--field-2)]"
                        style={{
                            opacity: selected ? 1 : 0.35,
                            pointerEvents: selected ? "auto" : "none",
                        }}
                    >
                        <div className="flex items-baseline justify-between border-b border-[var(--rule)] px-6 py-3">
                            <span className="label">
                                {selected ? `Bind “${selected.replace(/_/g, " ")}” to` : "Select a gesture"}
                            </span>
                            {selected && mappings[selected] && (
                                <button
                                    onClick={() => bind(null)}
                                    className="label hover:text-[var(--alert)]"
                                >
                                    Unbind
                                </button>
                            )}
                        </div>
                        <div className="min-h-0 flex-1 overflow-y-auto px-6 py-2">
                            {allActions.map((action) => {
                                const active = selected ? mappings[selected] === action.id : false;
                                return (
                                    <button
                                        key={action.id}
                                        onClick={() => bind(action.id)}
                                        className="w-full border-b border-[var(--rule-2)] py-3 text-left"
                                    >
                                        <div className="flex items-baseline justify-between gap-4">
                                            <span
                                                className="mono text-xs"
                                                style={{ color: active ? "var(--signal)" : "var(--ink)" }}
                                            >
                                                {action.name}
                                            </span>
                                            <span className="label">{action.category}</span>
                                        </div>
                                        <p className="mt-1 text-[11px] leading-relaxed text-[var(--ink-3)]">
                                            {action.description}
                                        </p>
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
