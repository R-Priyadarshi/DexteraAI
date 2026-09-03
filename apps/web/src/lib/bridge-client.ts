"use client";

/**
 * Client for the local desktop bridge.
 *
 * Without it, everything the app can do is confined to its own tab — which is
 * the right default for a web page and the wrong one for gesture control,
 * whose whole point is to drive the media player, the slides, and the window
 * you are actually looking at.
 *
 * The bridge is a daemon the user starts themselves (`python -m bridge.server`)
 * and binds to loopback. It requires a token, printed by that daemon, which the
 * user pastes in once. That is deliberate friction: a localhost WebSocket that
 * injects keystrokes would otherwise be reachable by any page in any tab, since
 * browsers do not apply the same-origin policy to WebSocket connections.
 *
 * The token is held in memory only. Persisting it to localStorage would leave a
 * credential for injecting input into the machine sitting in browser storage,
 * readable by anything with script access to this origin.
 */

export interface BridgeAction {
    id: string;
    name: string;
    category: string;
    description: string;
}

export type BridgeStatus = "disconnected" | "connecting" | "connected" | "error";

interface BridgeEvents {
    onStatus?: (status: BridgeStatus, detail?: string) => void;
    onActions?: (actions: BridgeAction[]) => void;
}

const DEFAULT_URL = "ws://127.0.0.1:8765";
const HEALTH_URL = "http://127.0.0.1:8765/health";

export class BridgeClient {
    private socket: WebSocket | null = null;
    private status: BridgeStatus = "disconnected";
    private actions: BridgeAction[] = [];
    private events: BridgeEvents = {};

    /** Gesture label -> bridge action id. */
    private bindings = new Map<string, string>();

    private static readonly BINDINGS_KEY = "dextera_bridge_bindings";

    constructor() {
        this.loadBindings();
    }

    on(events: BridgeEvents) {
        this.events = { ...this.events, ...events };
    }

    getStatus(): BridgeStatus {
        return this.status;
    }

    getActions(): BridgeAction[] {
        return [...this.actions];
    }

    getBindings(): Map<string, string> {
        return new Map(this.bindings);
    }

    /**
     * Is a bridge daemon running?
     *
     * Checked over plain HTTP first, because a failed WebSocket connection
     * produces an uncatchable console error in every browser — so probing by
     * connecting would spam the console every time the daemon is simply not
     * running, which is the normal case.
     */
    static async isAvailable(): Promise<boolean> {
        try {
            const res = await fetch(HEALTH_URL, {
                signal: AbortSignal.timeout(1200),
            });
            return res.ok;
        } catch {
            return false;
        }
    }

    connect(token: string, url: string = DEFAULT_URL): void {
        if (this.socket) this.disconnect();
        if (!token.trim()) {
            this.setStatus("error", "A token is required.");
            return;
        }

        this.setStatus("connecting");

        let socket: WebSocket;
        try {
            socket = new WebSocket(url);
        } catch {
            this.setStatus("error", "Could not open a connection to the bridge.");
            return;
        }
        this.socket = socket;

        socket.onopen = () => {
            // The token goes in the first frame; the daemon closes the socket
            // if it does not arrive promptly.
            socket.send(JSON.stringify({ token: token.trim() }));
        };

        socket.onmessage = (event) => {
            let payload: Record<string, unknown>;
            try {
                payload = JSON.parse(event.data as string);
            } catch {
                return;
            }

            if (payload.type === "ready") {
                this.actions = (payload.actions as BridgeAction[]) ?? [];
                this.setStatus("connected");
                this.events.onActions?.(this.getActions());
            }
        };

        socket.onclose = (event) => {
            this.socket = null;
            // 4401/4403 are the daemon's own rejections. Reporting them
            // specifically is the difference between a user fixing a mistyped
            // token and concluding the feature is broken.
            if (event.code === 4401) {
                this.setStatus("error", "The bridge rejected that token.");
            } else if (event.code === 4403) {
                this.setStatus(
                    "error",
                    "The bridge refused this origin. Start it with --allow-origin for this URL."
                );
            } else if (this.status === "connecting") {
                this.setStatus("error", "No bridge is listening. Start it first.");
            } else {
                this.setStatus("disconnected");
            }
        };

        socket.onerror = () => {
            // `onclose` always follows and carries the useful detail, so the
            // status is left for it to set rather than being overwritten here.
        };
    }

    disconnect(): void {
        const socket = this.socket;
        this.socket = null;
        if (socket) {
            socket.onclose = null;
            socket.close();
        }
        this.actions = [];
        this.setStatus("disconnected");
    }

    /** Bind a gesture label to a bridge action, or unbind with null. */
    bind(gestureName: string, actionId: string | null): void {
        if (actionId === null) this.bindings.delete(gestureName);
        else this.bindings.set(gestureName, actionId);
        this.saveBindings();
    }

    /**
     * Fire the desktop action bound to a completed gesture onset.
     *
     * Returns the action id sent, or null. As with the in-page registry, only
     * onsets dispatch: a held pose emits ~30 results a second, and injecting
     * keystrokes at that rate would make the desktop unusable.
     */
    dispatch(gestureName: string, phase: string, rejected: boolean): string | null {
        if (phase !== "onset" || rejected) return null;
        if (!this.socket || this.status !== "connected") return null;

        const actionId = this.bindings.get(gestureName);
        if (!actionId) return null;

        this.socket.send(JSON.stringify({ type: "action", action: actionId }));
        return actionId;
    }

    private setStatus(status: BridgeStatus, detail?: string) {
        this.status = status;
        this.events.onStatus?.(status, detail);
    }

    private loadBindings() {
        if (typeof window === "undefined") return;
        const raw = localStorage.getItem(BridgeClient.BINDINGS_KEY);
        if (!raw) return;
        try {
            this.bindings = new Map(Object.entries(JSON.parse(raw) as Record<string, string>));
        } catch {
            this.bindings = new Map();
        }
    }

    private saveBindings() {
        if (typeof window === "undefined") return;
        // Bindings are safe to persist — they name actions, not credentials.
        localStorage.setItem(
            BridgeClient.BINDINGS_KEY,
            JSON.stringify(Object.fromEntries(this.bindings))
        );
    }
}

export const bridgeClient = new BridgeClient();
