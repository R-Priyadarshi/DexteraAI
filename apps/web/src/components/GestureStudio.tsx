"use client";

import { useEffect, useRef, useState } from "react";
import { type GestureResult, type Landmark } from "@/lib/gesture-engine";
import { gestureStore, type CustomGesture } from "@/lib/gesture-store";

interface GestureStudioProps {
    gesture: GestureResult | null;
    onClose: () => void;
}

const REQUIRED_SAMPLES = 40;

/**
 * Teach the system a gesture it does not know, and manage the ones it has learned.
 *
 * The base model's vocabulary is necessarily finite. This is the escape hatch:
 * demonstrate a pose ~40 times and it becomes a matchable class immediately, with
 * no retraining and nothing leaving the device.
 */
export function GestureStudio({ gesture, onClose }: GestureStudioProps) {
    const [samples, setSamples] = useState<Landmark[][]>([]);
    const [name, setName] = useState("");
    const [status, setStatus] = useState<"idle" | "recording" | "complete">("idle");
    const [library, setLibrary] = useState<CustomGesture[]>([]);
    const [notice, setNotice] = useState<{ text: string; kind: "info" | "error" } | null>(null);

    const fileRef = useRef<HTMLInputElement>(null);

    const refresh = () => setLibrary([...gestureStore.getGestures()]);
    useEffect(refresh, []);

    useEffect(() => {
        if (status !== "recording" || !gesture?.landmarks) return;
        setSamples((prev) => {
            if (prev.length + 1 >= REQUIRED_SAMPLES) setStatus("complete");
            return prev.length < REQUIRED_SAMPLES ? [...prev, gesture.landmarks!] : prev;
        });
    }, [gesture, status]);

    const start = () => {
        if (!name.trim()) {
            setNotice({ text: "Name the gesture before recording.", kind: "error" });
            return;
        }
        setNotice(null);
        setSamples([]);
        setStatus("recording");
    };

    const save = () => {
        if (samples.length < REQUIRED_SAMPLES) return;
        gestureStore.addGesture(name.trim(), samples);
        setName("");
        setSamples([]);
        setStatus("idle");
        refresh();
        setNotice({ text: `Saved “${name.trim()}”.`, kind: "info" });
    };

    const exportPack = () => {
        const pack = gestureStore.exportPack();
        if (pack.gestures.length === 0) {
            setNotice({ text: "Nothing to export yet.", kind: "error" });
            return;
        }
        const blob = new Blob([JSON.stringify(pack, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `dextera-gestures-${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        // Revoking immediately can cancel the download in some browsers; one
        // tick is enough for the click to be dispatched.
        setTimeout(() => URL.revokeObjectURL(url), 0);
    };

    const importPack = async (file: File) => {
        try {
            const report = gestureStore.importPack(JSON.parse(await file.text()));
            refresh();
            const parts = [`Imported ${report.imported}`];
            if (report.skippedDuplicates.length) {
                parts.push(`${report.skippedDuplicates.length} already present`);
            }
            if (report.rejected.length) {
                parts.push(`${report.rejected.length} rejected`);
            }
            setNotice({
                text: parts.join(" · "),
                kind: report.imported > 0 ? "info" : "error",
            });
        } catch (err) {
            setNotice({
                text: err instanceof Error ? err.message : "Could not read that file.",
                kind: "error",
            });
        }
    };

    const progress = Math.min(1, samples.length / REQUIRED_SAMPLES);

    return (
        <div className="modal-backdrop fixed inset-0 z-[110] flex items-center justify-center p-6">
            <div className="panel-raised flex max-h-[86vh] w-full max-w-2xl flex-col">
                <header className="flex items-baseline justify-between border-b border-[var(--rule)] px-8 py-6">
                    <div>
                        <span className="label">Few-shot</span>
                        <h2 className="display mt-2 text-2xl text-[var(--ink)]">Studio</h2>
                    </div>
                    <button onClick={onClose} className="label hover:text-[var(--ink)]">
                        Close
                    </button>
                </header>

                <div className="min-h-0 flex-1 overflow-y-auto px-8 py-6">
                    <p className="max-w-md text-xs leading-relaxed text-[var(--ink-2)]">
                        Demonstrate a pose {REQUIRED_SAMPLES} times and it becomes matchable
                        straight away. Nothing is uploaded and no retraining happens — the
                        samples are compared directly against what the camera sees.
                    </p>

                    {/* Record */}
                    <div className="mt-8">
                        <div className="border-b border-[var(--rule)] pb-3">
                            <span className="label">Record</span>
                        </div>

                        <input
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            placeholder="Gesture name"
                            disabled={status === "recording"}
                            className="mono mt-5 w-full border border-[var(--rule-strong)] bg-[var(--field-1)] px-4 py-3 text-xs text-[var(--ink)] placeholder:text-[var(--ink-4)] focus:border-[var(--signal)] focus:outline-none disabled:opacity-40"
                        />

                        <div className="mt-5 flex items-baseline justify-between">
                            <span className="label">
                                {status === "recording"
                                    ? "Hold the pose, vary the angle slightly"
                                    : status === "complete"
                                        ? "Ready to save"
                                        : "Samples"}
                            </span>
                            <span className="readout text-xs text-[var(--ink)]">
                                {samples.length} / {REQUIRED_SAMPLES}
                            </span>
                        </div>
                        <div className="meter mt-2">
                            <div className="meter-fill" style={{ width: `${progress * 100}%` }} />
                        </div>

                        <div className="mt-5 flex gap-3">
                            <button
                                onClick={start}
                                disabled={status === "recording"}
                                className={`btn ${status === "recording" ? "btn-signal" : ""} disabled:opacity-60`}
                            >
                                {status === "recording" ? "Recording" : "Record"}
                            </button>
                            <button
                                onClick={save}
                                disabled={status !== "complete"}
                                className="btn btn-solid disabled:cursor-not-allowed disabled:opacity-25"
                            >
                                Save gesture
                            </button>
                        </div>
                    </div>

                    {/* Library */}
                    <div className="mt-12">
                        <div className="flex items-baseline justify-between border-b border-[var(--rule)] pb-3">
                            <span className="label">Your gestures</span>
                            <span className="readout text-[10px] text-[var(--ink-3)]">
                                {library.length}
                            </span>
                        </div>

                        {library.length === 0 ? (
                            <p className="label py-5">None taught yet</p>
                        ) : (
                            <div className="mt-2">
                                {library.map((g) => (
                                    <div
                                        key={g.id}
                                        className="flex items-baseline justify-between gap-4 border-b border-[var(--rule-2)] py-3"
                                    >
                                        <span className="mono text-xs text-[var(--ink)]">{g.name}</span>
                                        <div className="flex items-baseline gap-5">
                                            <span className="readout text-[10px] text-[var(--ink-3)]">
                                                {g.samples.length} samples
                                            </span>
                                            <button
                                                onClick={() => {
                                                    gestureStore.deleteGesture(g.id);
                                                    refresh();
                                                }}
                                                className="label hover:text-[var(--alert)]"
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        <div className="mt-5 flex gap-3">
                            <button onClick={exportPack} className="btn">
                                Export pack
                            </button>
                            <button onClick={() => fileRef.current?.click()} className="btn">
                                Import pack
                            </button>
                            <input
                                ref={fileRef}
                                type="file"
                                accept="application/json,.json"
                                className="hidden"
                                onChange={(e) => {
                                    const file = e.target.files?.[0];
                                    if (file) void importPack(file);
                                    // Clear the input, or selecting the same file
                                    // twice in a row fires no change event.
                                    e.target.value = "";
                                }}
                            />
                        </div>
                        <p className="mt-4 max-w-md text-[11px] leading-relaxed text-[var(--ink-3)]">
                            A pack is a JSON file of landmark coordinates — no images, no video.
                            It moves your gestures to another browser or machine, which is
                            otherwise impossible since they live only in this browser&apos;s
                            local storage.
                        </p>
                    </div>
                </div>

                <footer className="border-t border-[var(--rule)] px-8 py-5">
                    <span
                        className="label"
                        style={{
                            color:
                                notice?.kind === "error" ? "var(--alert)" : "var(--ink-3)",
                        }}
                    >
                        {notice?.text ?? "Stored on this device only"}
                    </span>
                </footer>
            </div>
        </div>
    );
}
