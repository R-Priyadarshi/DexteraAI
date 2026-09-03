"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { type GestureResult, type Landmark } from "@/lib/gesture-engine";
import { motionStore, type MotionClip } from "@/lib/motion-store";

interface MotionStudioProps {
    /**
     * Register a callback receiving every recognition frame, or null to
     * unregister. Frames arrive at the camera's full rate; the console's
     * throttled `gesture` state samples too slowly to capture a motion.
     */
    subscribe: (fn: ((r: GestureResult) => void) | null) => void;
    /** Frames per clip — the model's temporal window. */
    frameCount: number;
    onClose: () => void;
}

/**
 * Gestures worth recording first.
 *
 * These are the motions the current models cannot represent at all: every
 * shipped class is a static pose, so anything defined by how the hand *moves*
 * has no label to fall under. Suggesting a fixed starter set also keeps labels
 * consistent between people, which matters because pooled recordings are the
 * only way a self-recorded set generalises past its author.
 */
const SUGGESTED = [
    "swipe_left",
    "swipe_right",
    "swipe_up",
    "swipe_down",
    "wave",
    "circle_cw",
    "circle_ccw",
    "push",
    "pull",
    "dismiss",
];

/** Clips per label before a class is likely to train usefully. */
const TARGET_PER_LABEL = 20;

/** Milliseconds of countdown before capture begins. */
const COUNTDOWN_MS = 1200;

type Phase = "idle" | "countdown" | "capturing" | "saved";

/**
 * Record motion clips for training dynamic gestures.
 *
 * A clip is exactly `frameCount` consecutive frames — the same window the model
 * consumes at inference — so what is recorded is what will be classified. A
 * countdown precedes capture because the first frames of a clip started on a
 * button press are always the hand travelling back from the mouse.
 */
export function MotionStudio({ subscribe, frameCount, onClose }: MotionStudioProps) {
    const [label, setLabel] = useState("");
    const [phase, setPhase] = useState<Phase>("idle");
    const [frames, setFrames] = useState<Landmark[][]>([]);
    const [counts, setCounts] = useState<Record<string, number>>({});
    const [clips, setClips] = useState<MotionClip[]>([]);
    const [notice, setNotice] = useState<{ text: string; kind: "info" | "error" } | null>(null);
    const [countdownLeft, setCountdownLeft] = useState(0);

    const fileRef = useRef<HTMLInputElement>(null);
    const handednessRef = useRef<MotionClip["handedness"]>("unknown");

    const refresh = useCallback(() => {
        setCounts(motionStore.countsByLabel());
        setClips(motionStore.getClips());
    }, []);

    useEffect(refresh, [refresh]);

    // Countdown, then capture.
    useEffect(() => {
        if (phase !== "countdown") return;
        const started = performance.now();
        const timer = window.setInterval(() => {
            const left = COUNTDOWN_MS - (performance.now() - started);
            if (left <= 0) {
                setCountdownLeft(0);
                setFrames([]);
                setPhase("capturing");
            } else {
                setCountdownLeft(left);
            }
        }, 50);
        return () => window.clearInterval(timer);
    }, [phase]);

    // Accumulate frames while capturing, at the camera's full rate.
    useEffect(() => {
        if (phase !== "capturing") return;

        subscribe((result) => {
            if (!result.landmarks) return;
            handednessRef.current = result.handedness;
            setFrames((prev) =>
                prev.length >= frameCount ? prev : [...prev, result.landmarks!]
            );
        });

        // Unsubscribing on cleanup matters: a stale callback would keep
        // appending frames after the clip is saved and the modal is closed.
        return () => subscribe(null);
    }, [phase, frameCount, subscribe]);

    // Save as soon as the clip is full.
    useEffect(() => {
        if (phase !== "capturing" || frames.length < frameCount) return;
        try {
            motionStore.addClip(label.trim(), frames, handednessRef.current);
            refresh();
            setPhase("saved");
            setNotice({ text: `Recorded a clip of “${label.trim()}”.`, kind: "info" });
        } catch (err) {
            setPhase("idle");
            setNotice({
                text: err instanceof Error ? err.message : "Could not save the clip.",
                kind: "error",
            });
        }
    }, [frames, phase, frameCount, label, refresh]);

    const start = () => {
        if (!label.trim()) {
            setNotice({ text: "Choose or type a gesture name first.", kind: "error" });
            return;
        }
        setNotice(null);
        setFrames([]);
        setCountdownLeft(COUNTDOWN_MS);
        setPhase("countdown");
    };

    const exportPack = () => {
        const pack = motionStore.exportPack(frameCount);
        if (pack.clips.length === 0) {
            setNotice({ text: "Nothing recorded yet.", kind: "error" });
            return;
        }
        const blob = new Blob([JSON.stringify(pack)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `dextera-motion-${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 0);
    };

    const importPack = async (file: File) => {
        try {
            const report = motionStore.importPack(JSON.parse(await file.text()));
            refresh();
            setNotice({
                text: `Imported ${report.imported} clips${report.rejected ? `, ${report.rejected} rejected` : ""}.`,
                kind: report.imported > 0 ? "info" : "error",
            });
        } catch (err) {
            setNotice({
                text: err instanceof Error ? err.message : "Could not read that file.",
                kind: "error",
            });
        }
    };

    const progress =
        phase === "capturing" ? frames.length / frameCount : phase === "saved" ? 1 : 0;
    const labels = Object.keys(counts).sort();

    return (
        <div className="modal-backdrop fixed inset-0 z-[110] flex items-center justify-center p-6">
            <div className="panel-raised flex max-h-[88vh] w-full max-w-3xl flex-col">
                <header className="flex items-baseline justify-between border-b border-[var(--rule)] px-8 py-6">
                    <div>
                        <span className="label">Dynamic gestures</span>
                        <h2 className="display mt-2 text-2xl text-[var(--ink)]">Motion capture</h2>
                    </div>
                    <button onClick={onClose} className="label hover:text-[var(--ink)]">
                        Close
                    </button>
                </header>

                <div className="min-h-0 flex-1 overflow-y-auto px-8 py-6">
                    <p className="max-w-xl text-xs leading-relaxed text-[var(--ink-2)]">
                        Both shipped models are trained on still frames, so the 30-frame
                        Transformer has never actually seen movement — swipes, waves and
                        circles have no class to fall under. Recording them here builds the
                        sequence data that gap needs. Clips are exported as JSON and trained
                        offline; nothing is uploaded.
                    </p>

                    {/* Label */}
                    <div className="mt-8">
                        <div className="border-b border-[var(--rule)] pb-3">
                            <span className="label">Gesture</span>
                        </div>
                        <div className="mt-4 flex flex-wrap gap-2">
                            {SUGGESTED.map((s) => {
                                const active = label === s;
                                const n = counts[s] ?? 0;
                                return (
                                    <button
                                        key={s}
                                        onClick={() => setLabel(s)}
                                        disabled={phase === "countdown" || phase === "capturing"}
                                        className="border px-3 py-1.5 label transition-colors disabled:opacity-40"
                                        style={{
                                            borderRadius: 2,
                                            borderColor: active ? "var(--signal)" : "var(--rule-strong)",
                                            color: active ? "var(--signal)" : "var(--ink-3)",
                                            background: active ? "var(--signal-4)" : "transparent",
                                        }}
                                    >
                                        {s.replace(/_/g, " ")}
                                        {n > 0 && ` · ${n}`}
                                    </button>
                                );
                            })}
                        </div>

                        <input
                            type="text"
                            value={label}
                            onChange={(e) => setLabel(e.target.value)}
                            placeholder="or type a name"
                            disabled={phase === "countdown" || phase === "capturing"}
                            className="mono mt-4 w-full border border-[var(--rule-strong)] bg-[var(--field-1)] px-4 py-3 text-xs text-[var(--ink)] placeholder:text-[var(--ink-4)] focus:border-[var(--signal)] focus:outline-none disabled:opacity-40"
                        />
                    </div>

                    {/* Capture */}
                    <div className="mt-10">
                        <div className="flex items-baseline justify-between border-b border-[var(--rule)] pb-3">
                            <span className="label">
                                {phase === "countdown"
                                    ? "Get ready"
                                    : phase === "capturing"
                                        ? "Perform the gesture now"
                                        : "Capture"}
                            </span>
                            <span className="readout text-xs text-[var(--ink)]">
                                {phase === "countdown"
                                    ? `${(countdownLeft / 1000).toFixed(1)}s`
                                    : `${frames.length} / ${frameCount}`}
                            </span>
                        </div>

                        <div className="meter mt-3">
                            <div
                                className="meter-fill"
                                style={{
                                    width:
                                        phase === "countdown"
                                            ? `${(1 - countdownLeft / COUNTDOWN_MS) * 100}%`
                                            : `${progress * 100}%`,
                                    background:
                                        phase === "countdown" ? "var(--ink-3)" : "var(--signal)",
                                }}
                            />
                        </div>

                        <div className="mt-5 flex gap-3">
                            <button
                                onClick={start}
                                disabled={phase === "countdown" || phase === "capturing"}
                                className={`btn ${phase === "capturing" ? "btn-signal" : ""} disabled:opacity-50`}
                            >
                                {phase === "capturing" ? "Recording" : "Record clip"}
                            </button>
                            {label && (counts[label] ?? 0) > 0 && (
                                <button
                                    onClick={() => {
                                        motionStore.deleteLabel(label);
                                        refresh();
                                    }}
                                    className="btn"
                                >
                                    Clear “{label}”
                                </button>
                            )}
                        </div>

                        {label && (
                            <p className="mt-4 text-[11px] leading-relaxed text-[var(--ink-3)]">
                                {counts[label] ?? 0} of ~{TARGET_PER_LABEL} clips for this
                                gesture. Vary speed, distance and angle between takes — clips
                                that are all identical teach the model the recording setup
                                rather than the gesture.
                            </p>
                        )}
                    </div>

                    {/* Coverage */}
                    <div className="mt-10">
                        <div className="flex items-baseline justify-between border-b border-[var(--rule)] pb-3">
                            <span className="label">Recorded</span>
                            <span className="readout text-[10px] text-[var(--ink-3)]">
                                {clips.length} clips · {labels.length} classes
                            </span>
                        </div>

                        {labels.length === 0 ? (
                            <p className="label py-5">Nothing recorded yet</p>
                        ) : (
                            <div className="mt-2">
                                {labels.map((l) => {
                                    const n = counts[l];
                                    const ratio = Math.min(1, n / TARGET_PER_LABEL);
                                    return (
                                        <div key={l} className="border-b border-[var(--rule-2)] py-3">
                                            <div className="flex items-baseline justify-between gap-4">
                                                <span className="mono text-xs text-[var(--ink)]">
                                                    {l.replace(/_/g, " ")}
                                                </span>
                                                <span className="readout text-[10px] text-[var(--ink-3)]">
                                                    {n} / {TARGET_PER_LABEL}
                                                </span>
                                            </div>
                                            <div className="meter mt-2">
                                                <div
                                                    className={`meter-fill ${ratio < 1 ? "meter-fill-dim" : ""}`}
                                                    style={{ width: `${ratio * 100}%` }}
                                                />
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}

                        <div className="mt-5 flex flex-wrap gap-3">
                            <button onClick={exportPack} className="btn btn-signal">
                                Export for training
                            </button>
                            <button onClick={() => fileRef.current?.click()} className="btn">
                                Import pack
                            </button>
                            <button
                                onClick={() => {
                                    motionStore.clear();
                                    refresh();
                                    setNotice({ text: "Cleared all clips.", kind: "info" });
                                }}
                                className="btn"
                            >
                                Clear all
                            </button>
                            <input
                                ref={fileRef}
                                type="file"
                                accept="application/json,.json"
                                className="hidden"
                                onChange={(e) => {
                                    const file = e.target.files?.[0];
                                    if (file) void importPack(file);
                                    e.target.value = "";
                                }}
                            />
                        </div>

                        <pre className="mono mt-5 overflow-x-auto border border-[var(--rule)] bg-[var(--field-1)] p-4 text-[11px] leading-relaxed text-[var(--ink-2)]">
{`# Then, to train on what you recorded:
python training/datasets/import_recordings.py \\
    --pack ~/Downloads/dextera-motion-*.json \\
    --out data/sequences/motion

python dextera.py train --dataset data/sequences/motion \\
    --epochs 120 --calibrate --export models/motion`}
                        </pre>
                    </div>
                </div>

                <footer className="border-t border-[var(--rule)] px-8 py-5">
                    <span
                        className="label"
                        style={{ color: notice?.kind === "error" ? "var(--alert)" : "var(--ink-3)" }}
                    >
                        {notice?.text ?? "Recorded on this device only"}
                    </span>
                </footer>
            </div>
        </div>
    );
}
