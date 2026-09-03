"use client";

import { useEffect, useState } from "react";
import {
    bridgeClient,
    BridgeClient,
    type BridgeAction,
    type BridgeStatus,
} from "@/lib/bridge-client";

interface DesktopBridgeProps {
    vocabulary: string[];
    onClose: () => void;
}

/**
 * Connect to the local desktop bridge and bind gestures to OS actions.
 *
 * The token step is the point of the panel. It is friction on purpose:
 * browsers do not apply the same-origin policy to WebSocket connections, so an
 * unauthenticated local daemon that injects keystrokes could be driven by any
 * page in any tab. The user pastes the token once per session, and it is never
 * persisted.
 */
export function DesktopBridge({ vocabulary, onClose }: DesktopBridgeProps) {
    const [status, setStatus] = useState<BridgeStatus>(bridgeClient.getStatus());
    const [detail, setDetail] = useState<string | null>(null);
    const [token, setToken] = useState("");
    const [actions, setActions] = useState<BridgeAction[]>(bridgeClient.getActions());
    const [bindings, setBindings] = useState<Record<string, string>>({});
    const [selected, setSelected] = useState<string | null>(null);
    const [daemonUp, setDaemonUp] = useState<boolean | null>(null);

    useEffect(() => {
        bridgeClient.on({
            onStatus: (s, d) => {
                setStatus(s);
                setDetail(d ?? null);
            },
            onActions: setActions,
        });
        setBindings(Object.fromEntries(bridgeClient.getBindings()));
        void BridgeClient.isAvailable().then(setDaemonUp);
    }, []);

    const bind = (actionId: string | null) => {
        if (!selected) return;
        bridgeClient.bind(selected, actionId);
        setBindings(Object.fromEntries(bridgeClient.getBindings()));
        setSelected(null);
    };

    const connected = status === "connected";

    return (
        <div className="modal-backdrop fixed inset-0 z-[120] flex items-center justify-center p-6">
            <div className="panel-raised flex max-h-[88vh] w-full max-w-4xl flex-col">
                <header className="flex items-baseline justify-between border-b border-[var(--rule)] px-8 py-6">
                    <div>
                        <span className="label">Beyond the tab</span>
                        <h2 className="display mt-2 text-2xl text-[var(--ink)]">
                            Desktop control
                        </h2>
                    </div>
                    <button onClick={onClose} className="label hover:text-[var(--ink)]">
                        Close
                    </button>
                </header>

                <div className="min-h-0 flex-1 overflow-y-auto px-8 py-6">
                    <p className="max-w-xl text-xs leading-relaxed text-[var(--ink-2)]">
                        A web page can only act inside its own tab. The bridge is a small
                        local daemon that lets a gesture drive whatever you are actually
                        using — the media player, the slides, the focused window. It listens
                        on loopback only and nothing it handles leaves the machine.
                    </p>

                    {/* Connection */}
                    <div className="mt-8">
                        <div className="flex items-baseline justify-between border-b border-[var(--rule)] pb-3">
                            <span className="label">Connection</span>
                            <div className="flex items-center gap-2">
                                <div className={connected ? "live-dot" : "idle-dot"} />
                                <span className={`label ${connected ? "label-signal" : ""}`}>
                                    {status}
                                </span>
                            </div>
                        </div>

                        {!connected && (
                            <>
                                <p className="mt-4 text-[11px] leading-relaxed text-[var(--ink-3)]">
                                    {daemonUp === false
                                        ? "No bridge is running. Start it, then paste the token it prints."
                                        : "Paste the token the bridge printed on startup."}
                                </p>
                                <pre className="mono mt-3 overflow-x-auto border border-[var(--rule)] bg-[var(--field-1)] p-4 text-[11px] text-[var(--ink-2)]">
python -m bridge.server
                                </pre>

                                <div className="mt-4 flex flex-wrap gap-3">
                                    <input
                                        type="password"
                                        value={token}
                                        onChange={(e) => setToken(e.target.value)}
                                        placeholder="Bridge token"
                                        autoComplete="off"
                                        className="mono min-w-[260px] flex-1 border border-[var(--rule-strong)] bg-[var(--field-1)] px-4 py-3 text-xs text-[var(--ink)] placeholder:text-[var(--ink-4)] focus:border-[var(--signal)] focus:outline-none"
                                    />
                                    <button
                                        onClick={() => bridgeClient.connect(token)}
                                        disabled={!token.trim() || status === "connecting"}
                                        className="btn btn-signal disabled:opacity-30"
                                    >
                                        {status === "connecting" ? "Connecting" : "Connect"}
                                    </button>
                                </div>
                                <p className="mt-3 max-w-xl text-[11px] leading-relaxed text-[var(--ink-3)]">
                                    The token is kept in memory for this session only. Storing a
                                    credential that can inject input into your machine in browser
                                    storage would leave it readable by anything with script access
                                    to this page.
                                </p>
                            </>
                        )}

                        {connected && (
                            <div className="mt-4 flex items-center gap-3">
                                <button onClick={() => bridgeClient.disconnect()} className="btn">
                                    Disconnect
                                </button>
                                <span className="label">{actions.length} actions available</span>
                            </div>
                        )}
                    </div>

                    {/* Bindings */}
                    {connected && (
                        <div className="mt-10 grid grid-cols-1 gap-8 md:grid-cols-2">
                            <div>
                                <div className="border-b border-[var(--rule)] pb-3">
                                    <span className="label">Gesture</span>
                                </div>
                                <div className="mt-2 max-h-[38vh] overflow-y-auto">
                                    {vocabulary.map((label) => {
                                        const bound = bindings[label];
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
                                                        ? actions.find((a) => a.id === bound)?.name ?? bound
                                                        : "unbound"}
                                                </span>
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>

                            <div
                                style={{
                                    opacity: selected ? 1 : 0.35,
                                    pointerEvents: selected ? "auto" : "none",
                                }}
                            >
                                <div className="flex items-baseline justify-between border-b border-[var(--rule)] pb-3">
                                    <span className="label">
                                        {selected ? "Desktop action" : "Select a gesture"}
                                    </span>
                                    {selected && bindings[selected] && (
                                        <button
                                            onClick={() => bind(null)}
                                            className="label hover:text-[var(--alert)]"
                                        >
                                            Unbind
                                        </button>
                                    )}
                                </div>
                                <div className="mt-2 max-h-[38vh] overflow-y-auto">
                                    {actions.map((a) => (
                                        <button
                                            key={a.id}
                                            onClick={() => bind(a.id)}
                                            className="w-full border-b border-[var(--rule-2)] py-3 text-left"
                                        >
                                            <div className="flex items-baseline justify-between gap-4">
                                                <span
                                                    className="mono text-xs"
                                                    style={{
                                                        color:
                                                            selected && bindings[selected] === a.id
                                                                ? "var(--signal)"
                                                                : "var(--ink)",
                                                    }}
                                                >
                                                    {a.name}
                                                </span>
                                                <span className="label">{a.category}</span>
                                            </div>
                                            <p className="mt-1 text-[11px] leading-relaxed text-[var(--ink-3)]">
                                                {a.description}
                                            </p>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                <footer className="border-t border-[var(--rule)] px-8 py-5">
                    <span
                        className="label"
                        style={{ color: status === "error" ? "var(--alert)" : "var(--ink-3)" }}
                    >
                        {detail ?? "Loopback only · nothing leaves this machine"}
                    </span>
                </footer>
            </div>
        </div>
    );
}
