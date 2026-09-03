"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { GestureEngine, type GestureResult } from "@/lib/gesture-engine";
import { type GestureAction } from "@/lib/action-registry";
import { PluginEngine } from "@/lib/plugin-engine";
import { PresentationPlugin } from "@/lib/plugins/presentation";
import { BiometricGuard, signatureFor } from "@/components/BiometricGuard";
import { GestureStudio } from "@/components/GestureStudio";
import { MotionStudio } from "@/components/MotionStudio";
import { DesktopBridge } from "@/components/DesktopBridge";
import { bridgeClient } from "@/lib/bridge-client";
import { MacroComposer } from "@/components/MacroComposer";
import { macroEngine } from "@/lib/macro-engine";
import { voiceEngine, type VoiceIntent } from "@/lib/voice-engine";
import { faceEngine, type FacialMarker } from "@/lib/face-engine";
import { intentRefinery, type FusedAction } from "@/lib/intent-refinery";
import { hapticEngine } from "@/lib/haptic-engine";
import { tacticalAudio } from "@/lib/tactical-audio";
import { FusionMonitor } from "@/components/FusionMonitor";
import { calibrator, type CalibrationMetrics } from "@/lib/calibrator";
import { SpatialDeck } from "@/components/SpatialDeck";
import { ActionMapper } from "@/components/ActionMapper";
import { actionRegistry } from "@/lib/action-registry";
import { coverGeometry, projectLandmark } from "@/lib/overlay-geometry";
import { PointerEngine, clickAt, type PointerState } from "@/lib/pointer-engine";
import { PointerOverlay } from "@/components/PointerOverlay";
import { CalibrationWizard } from "@/components/CalibrationWizard";
import { biometricEngine } from "@/lib/biometric-engine";
import { Sparkline, StatusFlag } from "@/components/Telemetry";

/**
 * Trained model bundles. Each directory holds gesture.onnx + labels.json, so the
 * vocabulary travels with the model rather than being hardcoded here.
 * Produced by: python dextera.py train --export models/<name>
 */
const MODEL_BUNDLES = [
    { id: "hagrid", name: "General gestures", url: "/onnx/hagrid/gesture.onnx", classes: 18 },
    { id: "asl_alphabet", name: "ASL fingerspelling", url: "/onnx/asl_alphabet/gesture.onnx", classes: 26 },
] as const;

type BundleId = (typeof MODEL_BUNDLES)[number]["id"];

const LATENCY_WINDOW = 60;

export default function Console() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const engineRef = useRef<GestureEngine | null>(null);
    const pluginEngine = PluginEngine.getInstance();

    const [isRunning, setIsRunning] = useState(false);
    const [bundleId, setBundleId] = useState<BundleId>("hagrid");
    const [twoHanded, setTwoHanded] = useState(false);

    /**
     * MediaPipe model complexity. Lite by default — see `gesture-engine.ts`.
     * Exposed so a fast machine can opt into the precise model and a slow one
     * can confirm that is where its frame time is going.
     */
    const [preciseModel, setPreciseModel] = useState(false);
    useEffect(() => {
        engineRef.current?.setModelComplexity(preciseModel ? 1 : 0);
    }, [preciseModel]);

    /** Rolling detect/classify split, for attributing a slow frame. */
    const [timing, setTiming] = useState<{ detect: number; classify: number }>({
        detect: 0,
        classify: 0,
    });

    /**
     * Hands-free pointing. Off by default: while it is on, the hand drives a
     * cursor that clicks whatever it rests on, which is not what someone
     * exploring the console expects to happen.
     */
    const [pointerMode, setPointerMode] = useState(false);
    const pointerEngineRef = useRef<PointerEngine | null>(null);
    const [pointer, setPointer] = useState<PointerState | null>(null);
    const [clickFlash, setClickFlash] = useState(false);

    // Read from inside the animation loop, which must not be re-created when
    // this toggles.
    const pointerModeRef = useRef(pointerMode);
    useEffect(() => {
        pointerModeRef.current = pointerMode;
        if (!pointerMode) {
            pointerEngineRef.current?.reset();
            setPointer(null);
        }
    }, [pointerMode]);

    // Read inside the camera-start callback, which must not re-create itself
    // when this toggles — doing so tears the camera down mid-session.
    const twoHandedRef = useRef(twoHanded);
    useEffect(() => {
        twoHandedRef.current = twoHanded;
        // Applied live: MediaPipe accepts a new hand count without a restart,
        // so there is no reason to make the user stop the camera for it.
        void engineRef.current?.setMaxHands(twoHanded ? 2 : 1);
    }, [twoHanded]);
    const [vocabulary, setVocabulary] = useState<string[]>([]);
    const [gesture, setGesture] = useState<GestureResult | null>(null);
    const [fps, setFps] = useState(0);
    const [latency, setLatency] = useState(0);
    const [latencyHistory, setLatencyHistory] = useState<number[]>([]);
    const [error, setError] = useState<string | null>(null);
    const isProcessingRef = useRef(false);

    const [recentActions, setRecentActions] = useState<{ action: GestureAction; timestamp: number }[]>([]);
    const [isLocked, setIsLocked] = useState(true);
    const [isStudioOpen, setIsStudioOpen] = useState(false);
    const [isMotionOpen, setIsMotionOpen] = useState(false);
    const [isBridgeOpen, setIsBridgeOpen] = useState(false);

    /**
     * Full-rate frame tap.
     *
     * `gesture` state is throttled to 10Hz, which is fine for readouts but
     * useless for recording motion: a 30-frame clip would span three seconds
     * and sample a fraction of the movement. Consumers that need every frame
     * subscribe here instead.
     */
    const frameSubscriberRef = useRef<((r: GestureResult) => void) | null>(null);
    const subscribeToFrames = useCallback(
        (fn: ((r: GestureResult) => void) | null) => {
            frameSubscriberRef.current = fn;
        },
        []
    );
    const [isMacroOpen, setIsMacroOpen] = useState(false);
    const [isMapperOpen, setIsMapperOpen] = useState(false);
    const [isCalibrating, setIsCalibrating] = useState(false);
    const [activeMacro, setActiveMacro] = useState<string | null>(null);
    const [voiceIntent, setVoiceIntent] = useState<VoiceIntent | null>(null);
    const [isVoiceActive, setIsVoiceActive] = useState(false);
    const [lastFusedAction, setLastFusedAction] = useState<FusedAction | null>(null);
    const [calibration, setCalibration] = useState<CalibrationMetrics>({
        stability: 1,
        lighting: 1,
        latencyJitter: 0,
    });

    const isLockedRef = useRef(isLocked);
    // Off by default: a second landmarker is real per-frame cost, and the
    // gesture path must not get slower for people who never turn this on.
    const [faceEnabled, setFaceEnabled] = useState(false);
    const [facialMarker, setFacialMarker] = useState<FacialMarker | null>(null);
    const [faceReady, setFaceReady] = useState(false);
    const facialMarkerRef = useRef<FacialMarker | null>(null);
    const faceEnabledRef = useRef(faceEnabled);

    const voiceIntentRef = useRef(voiceIntent);
    useEffect(() => { isLockedRef.current = isLocked; }, [isLocked]);
    useEffect(() => { voiceIntentRef.current = voiceIntent; }, [voiceIntent]);
    useEffect(() => { faceEnabledRef.current = faceEnabled; }, [faceEnabled]);

    // Load on demand rather than at boot: most sessions never enable it, and
    // the model is a few megabytes.
    useEffect(() => {
        if (!faceEnabled) return;
        let cancelled = false;
        faceEngine.init().then((ok) => {
            if (!cancelled) setFaceReady(ok);
        });
        return () => { cancelled = true; };
    }, [faceEnabled]);

    const startCamera = useCallback(async () => {
        // Held outside the try so the catch can release it. `engine.initialize`
        // now throws deliberately when WebGL is unavailable, and without this
        // the camera stayed on — recording light and all — after a failure the
        // user was told about but could not undo.
        let stream: MediaStream | null = null;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
            }

            // Booting again (or switching vocabulary) must not strand the old engine.
            engineRef.current?.dispose();
            engineRef.current = null;

            const engine = new GestureEngine();
            const bundle = MODEL_BUNDLES.find((b) => b.id === bundleId) ?? MODEL_BUNDLES[0];
            await engine.initialize(bundle.url);
            await engine.setMaxHands(twoHandedRef.current ? 2 : 1);
            engineRef.current = engine;
            setVocabulary(engine.getLabels());

            voiceEngine.start({
                onIntent: (intent) => {
                    setVoiceIntent(intent);
                    setTimeout(() => setVoiceIntent(null), 2000);
                },
                onStatus: (status) =>
                    setIsVoiceActive(status === "listening" || status === "processing"),
            });

            setIsRunning(true);
            setError(null);
            tacticalAudio.startNeuralHum();
            hapticEngine.pulse("success");
        } catch (err) {
            stream?.getTracks().forEach((track) => track.stop());
            if (videoRef.current) videoRef.current.srcObject = null;
            engineRef.current?.dispose();
            engineRef.current = null;
            setIsRunning(false);
            setError(
                `Camera unavailable: ${err instanceof Error ? err.message : String(err)}`
            );
        }
    }, [bundleId]);

    // Stable identity: an effect below calls stopCamera() in its cleanup.
    // Recreating it each render tore the camera down on every state update,
    // and the frame loop updates state ~10x a second.
    const stopCamera = useCallback(() => {
        if (videoRef.current?.srcObject) {
            (videoRef.current.srcObject as MediaStream)
                .getTracks()
                .forEach((track) => track.stop());
            videoRef.current.srcObject = null;
        }
        // Release the ONNX session and MediaPipe WebGL context. Nulling the ref
        // alone leaks a WebGL context per boot.
        engineRef.current?.dispose();
        engineRef.current = null;
        setIsRunning(false);
        setGesture(null);
        voiceEngine.stop();
        faceEngine.close();
        setFaceReady(false);
        setFacialMarker(null);
        facialMarkerRef.current = null;
        tacticalAudio.stopNeuralHum();
        hapticEngine.pulse("sonar_ping");
    }, []);

    const frameCountRef = useRef(0);
    const lastFpsTimeRef = useRef(0);
    const lastStateUpdateTimeRef = useRef(0);
    const loopRef = useRef<number | null>(null);

    const drawLandmarks = useCallback((
        canvas: HTMLCanvasElement,
        video: HTMLVideoElement,
        result: GestureResult
    ) => {
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        // Back the canvas at device resolution. Sizing it in CSS pixels leaves
        // the skeleton visibly soft on any HiDPI display, where one CSS pixel
        // is two or three physical ones. Every coordinate below is then in
        // device pixels, which is consistent because the mapping derives from
        // `canvas.width`/`canvas.height` rather than from the CSS box.
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const targetW = Math.round(rect.width * dpr);
        const targetH = Math.round(rect.height * dpr);
        if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw every tracked hand. Rendering only the primary would leave the
        // second hand's skeleton missing while it is still driving recognition,
        // which reads as a tracking failure rather than a display choice.
        const hands = result.hands.length > 0
            ? result.hands
            : result.landmarks
                ? [{ landmarks: result.landmarks, rejected: result.rejected }]
                : [];
        if (hands.length === 0) return;

        const CONNECTIONS = [
            [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13], [13, 17],
        ];

        // Landmarks are normalised over the source frame, while the video is
        // displayed with `object-fit: cover`. See `overlay-geometry.ts` — the
        // mapping is non-obvious and unit-tested there rather than inline.
        const geometry = coverGeometry(
            video.videoWidth,
            video.videoHeight,
            canvas.width,
            canvas.height
        );
        const px = (p: { x: number; y: number }) => projectLandmark(p, geometry);

        const tips = new Set([4, 8, 12, 16, 20]);

        for (const hand of hands) {
            const landmarks = hand.landmarks;
            if (!landmarks || landmarks.length < 21) continue;

            // A rejected hand is still tracked, just not classified. Dimming it
            // says so, rather than implying the same confidence as an accepted one.
            const colour = hand.rejected ? "#605e58" : "#ffb627";

            ctx.strokeStyle = colour;
            // Line weights and dot radii are authored in CSS pixels, so they
            // scale with the backing store or they would shrink on HiDPI.
            ctx.lineWidth = 2 * dpr;
            ctx.lineCap = "round";
            ctx.beginPath();
            for (const [i, j] of CONNECTIONS) {
                const a = landmarks[i];
                const b = landmarks[j];
                if (!a || !b) continue;
                const pa = px(a);
                const pb = px(b);
                ctx.moveTo(pa.x, pa.y);
                ctx.lineTo(pb.x, pb.y);
            }
            ctx.stroke();

            landmarks.forEach((p, i) => {
                const c = px(p);
                ctx.beginPath();
                ctx.arc(c.x, c.y, (i === 0 ? 5 : tips.has(i) ? 4 : 2.5) * dpr, 0, Math.PI * 2);
                if (i === 0 || tips.has(i)) {
                    ctx.fillStyle = colour;
                    ctx.fill();
                } else {
                    ctx.fillStyle = "#0a0a0b";
                    ctx.fill();
                    ctx.strokeStyle = colour;
                    ctx.lineWidth = 1.2 * dpr;
                    ctx.stroke();
                }
            });
        }
    }, []);

    useEffect(() => {
        if (!isRunning || !engineRef.current) {
            if (loopRef.current) cancelAnimationFrame(loopRef.current);
            return;
        }

        const loop = async () => {
            const video = videoRef.current;
            const canvas = canvasRef.current;

            if (!video || !canvas || !engineRef.current || video.readyState < 2) {
                loopRef.current = requestAnimationFrame(loop);
                return;
            }

            loopRef.current = requestAnimationFrame(loop);

            if (!isProcessingRef.current) {
                isProcessingRef.current = true;
                try {
                    const result = await engineRef.current.processFrame(video);
                    const ctx = canvas.getContext("2d");

                    if (result) {
                        drawLandmarks(canvas, video, result);
                        calibrator.record(result.landmarks);

                        frameSubscriberRef.current?.(result);

                        // Every third frame, ~10Hz. A brow raise lasts the
                        // better part of a second, so the extra resolution
                        // would buy nothing and this is a second model running
                        // against the same frame budget as hand detection.
                        if (
                            faceEnabledRef.current &&
                            faceEngine.isReady &&
                            frameCountRef.current % 3 === 0
                        ) {
                            const reading = faceEngine.detect(video);
                            const marker = reading?.marker ?? null;
                            if (marker !== facialMarkerRef.current) {
                                facialMarkerRef.current = marker;
                                setFacialMarker(marker);
                            }
                        }

                        // Pointer runs every frame, not on the throttled state
                        // tick: a cursor updated ten times a second is unusable.
                        if (pointerModeRef.current) {
                            const engine = (pointerEngineRef.current ??= new PointerEngine());
                            const state = engine.update(
                                result.landmarks,
                                { width: window.innerWidth, height: window.innerHeight },
                                performance.now()
                            );
                            setPointer(state);
                            if (state.clicked) {
                                clickAt(state.x, state.y);
                                hapticEngine.pulse("light");
                                setClickFlash(true);
                                window.setTimeout(() => setClickFlash(false), 180);
                            }
                        }

                        const now = performance.now();
                        if (now - lastStateUpdateTimeRef.current > 100) {
                            const ms = Math.round(result.inferenceTimeMs);
                            setLatency(ms);
                            setLatencyHistory((h) => {
                                const next = [...h, result.inferenceTimeMs];
                                return next.length > LATENCY_WINDOW
                                    ? next.slice(-LATENCY_WINDOW)
                                    : next;
                            });
                            setGesture(result);
                            setTiming({
                                detect: result.detectMs,
                                classify: result.classifyMs,
                            });
                            setCalibration(calibrator.getMetrics());
                            lastStateUpdateTimeRef.current = now;
                        }

                        if (!isLockedRef.current) {
                            pluginEngine.broadcast(result);

                            // Bound actions fire on segment onset only. The
                            // registry enforces that, but the log below must
                            // also only record real dispatches.
                            // A combo takes precedence: when two hands form a
                            // bound pair, firing the single-hand binding as
                            // well would run two actions for one intent.
                            // Desktop bindings are independent of in-page ones:
                            // a gesture may drive the OS, this page, or both,
                            // and the bridge is a no-op when not connected.
                            bridgeClient.dispatch(
                                result.gestureName,
                                result.phase,
                                result.rejected
                            );

                            const fired =
                                actionRegistry.dispatchCombo(result) ??
                                actionRegistry.dispatch(result);
                            if (fired) {
                                hapticEngine.pulse("light");
                                setRecentActions((log) =>
                                    [{ action: fired, timestamp: Date.now() }, ...log].slice(0, 40)
                                );
                            }

                            const macroMatch = macroEngine.process(result);
                            if (macroMatch) setActiveMacro(macroMatch.name);

                            const fusedMatch = intentRefinery.process(
                                result,
                                voiceIntentRef.current,
                                facialMarkerRef.current,
                            );
                            if (fusedMatch) {
                                // `process` already executed the action and
                                // pulsed. Doing it again here fired every fused
                                // action twice — which locked the screen twice
                                // for the emergency halt, and would silently
                                // cancel itself for anything that toggles.
                                setLastFusedAction(fusedMatch);
                                // Record it, so the log reflects real dispatches.
                                setRecentActions((log) =>
                                    [
                                        {
                                            action: {
                                                id: fusedMatch.id,
                                                name: fusedMatch.name,
                                            } as GestureAction,
                                            timestamp: Date.now(),
                                        },
                                        ...log,
                                    ].slice(0, 40)
                                );
                            }
                        }
                    } else if (ctx) {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                    }
                    isProcessingRef.current = false;
                } catch {
                    isProcessingRef.current = false;
                }
            }

            frameCountRef.current++;
            const t = performance.now();
            if (t - lastFpsTimeRef.current >= 1000) {
                setFps(frameCountRef.current);
                frameCountRef.current = 0;
                lastFpsTimeRef.current = t;
            }
        };

        loopRef.current = requestAnimationFrame(loop);
        return () => {
            if (loopRef.current) cancelAnimationFrame(loopRef.current);
        };
    }, [isRunning, drawLandmarks, pluginEngine]);

    useEffect(() => {
        const handleSysHalt = () => {
            stopCamera();
            setIsLocked(true);
            setError("Stopped — abort gesture recognised.");
        };
        const handleSysReset = () => {
            setRecentActions([]);
            setLastFusedAction(null);
            hapticEngine.pulse("success");
        };

        window.addEventListener("dextera_sys_halt", handleSysHalt);
        window.addEventListener("dextera_sys_reset", handleSysReset);
        pluginEngine.register(PresentationPlugin);
        return () => {
            stopCamera();
            window.removeEventListener("dextera_sys_halt", handleSysHalt);
            window.removeEventListener("dextera_sys_reset", handleSysReset);
        };
    }, [stopCamera, pluginEngine]);

    useEffect(() => {
        if (!error) return;
        const t = setTimeout(() => setError(null), 4000);
        return () => clearTimeout(t);
    }, [error]);

    const detected = Boolean(gesture && gesture.gestureId !== -1 && gesture.landmarks?.length);
    const activeBundle = MODEL_BUNDLES.find((b) => b.id === bundleId) ?? MODEL_BUNDLES[0];
    const wrist = gesture?.landmarks?.[0];

    return (
        <main className="min-h-screen">
            {/* Modals */}
            {pointerMode && pointer && (
                <PointerOverlay
                    x={pointer.x}
                    y={pointer.y}
                    dwellProgress={pointer.dwellProgress}
                    active={pointer.active}
                    flash={clickFlash}
                />
            )}

            {isStudioOpen && <GestureStudio gesture={gesture} onClose={() => setIsStudioOpen(false)} />}
            {isMapperOpen && (
                <ActionMapper vocabulary={vocabulary} onClose={() => setIsMapperOpen(false)} />
            )}
            {isMacroOpen && <MacroComposer gesture={gesture} onClose={() => setIsMacroOpen(false)} />}
            {isBridgeOpen && (
                <DesktopBridge vocabulary={vocabulary} onClose={() => setIsBridgeOpen(false)} />
            )}

            {isMotionOpen && (
                <MotionStudio
                    subscribe={subscribeToFrames}
                    frameCount={engineRef.current?.getSequenceLength() ?? 30}
                    onClose={() => setIsMotionOpen(false)}
                />
            )}
            {isCalibrating && (
                <CalibrationWizard
                    landmarks={gesture?.landmarks || null}
                    onClose={() => setIsCalibrating(false)}
                    onComplete={() => {
                        setIsCalibrating(false);
                        setError("Calibration saved.");
                    }}
                />
            )}

            {/* ── Header ─────────────────────────────────── */}
            <header className="sticky top-0 z-50 border-b border-[var(--rule)] bg-[var(--field)]/95 backdrop-blur-[2px]">
                <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4 px-6 py-3 lg:px-8">
                    <div className="flex items-center gap-6">
                        <Link href="/" className="display text-[15px] text-[var(--ink)] hover:opacity-70">
                            Dextera
                        </Link>
                        <span className="label hidden sm:inline">Console</span>
                        <StatusFlag live={isRunning} label={isRunning ? "Acquiring" : "Camera off"} />
                    </div>

                    <div className="flex items-center gap-5 sm:gap-8">
                        <div className="flex flex-col items-end">
                            <span className="label">Frame</span>
                            <span className="readout text-sm text-[var(--ink)]">
                                {isRunning ? `${latency} ms` : "—"}
                            </span>
                        </div>
                        <div className="flex flex-col items-end">
                            <span className="label">Rate</span>
                            <span className="readout text-sm text-[var(--ink)]">
                                {isRunning ? `${fps} fps` : "—"}
                            </span>
                        </div>
                        <div className="flex flex-col items-end border-l border-[var(--rule)] pl-5 sm:pl-8">
                            <span className="label">Guard</span>
                            <span
                                className="readout text-sm"
                                style={{ color: isLocked ? "var(--alert)" : "var(--signal)" }}
                            >
                                {isLocked ? "Locked" : "Open"}
                            </span>
                        </div>
                    </div>
                </div>
            </header>

            <div className="mx-auto max-w-[1600px] px-6 py-6 lg:px-8">
                <div className="grid grid-cols-1 items-start gap-5 xl:grid-cols-[minmax(0,2.1fr)_minmax(320px,1fr)]">
                    {/* ── Viewport ───────────────────────── */}
                    <section className="panel overflow-hidden">
                        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--rule)] px-4 py-2.5">
                            <span className="label">Sensor</span>
                            <div className="flex items-center gap-2">
                                {MODEL_BUNDLES.map((bundle) => {
                                    const active = bundleId === bundle.id;
                                    return (
                                        <button
                                            key={bundle.id}
                                            onClick={() => {
                                                if (isRunning) {
                                                    setError("Stop the camera before switching vocabulary.");
                                                    return;
                                                }
                                                setBundleId(bundle.id);
                                            }}
                                            className="border px-3 py-1.5 label transition-colors"
                                            style={{
                                                borderRadius: 2,
                                                borderColor: active ? "var(--signal)" : "var(--rule-strong)",
                                                color: active ? "var(--signal)" : "var(--ink-3)",
                                                background: active ? "var(--signal-4)" : "transparent",
                                            }}
                                        >
                                            {bundle.name} · {bundle.classes}
                                        </button>
                                    );
                                })}
                                <button
                                    onClick={() => setTwoHanded((v) => !v)}
                                    className="border px-3 py-1.5 label transition-colors"
                                    style={{
                                        borderRadius: 2,
                                        borderColor: twoHanded ? "var(--signal)" : "var(--rule-strong)",
                                        color: twoHanded ? "var(--signal)" : "var(--ink-3)",
                                        background: twoHanded ? "var(--signal-4)" : "transparent",
                                    }}
                                    title="Track both hands. Roughly doubles landmark-detection cost."
                                >
                                    {twoHanded ? "2 hands" : "1 hand"}
                                </button>
                                <button
                                    onClick={() => setPreciseModel((v) => !v)}
                                    className="border px-3 py-1.5 label transition-colors"
                                    style={{
                                        borderRadius: 2,
                                        borderColor: preciseModel ? "var(--signal)" : "var(--rule-strong)",
                                        color: preciseModel ? "var(--signal)" : "var(--ink-3)",
                                        background: preciseModel ? "var(--signal-4)" : "transparent",
                                    }}
                                    title="Full-complexity landmarks. Two to three times slower per frame."
                                >
                                    {preciseModel ? "Precise" : "Fast"}
                                </button>
                                <button
                                    onClick={() => setPointerMode((v) => !v)}
                                    className="border px-3 py-1.5 label transition-colors"
                                    style={{
                                        borderRadius: 2,
                                        borderColor: pointerMode ? "var(--signal)" : "var(--rule-strong)",
                                        color: pointerMode ? "var(--signal)" : "var(--ink-3)",
                                        background: pointerMode ? "var(--signal-4)" : "transparent",
                                    }}
                                    title="Point with your index finger; hold still to click."
                                >
                                    Pointer
                                </button>
                            </div>
                        </div>

                        <div className="brackets relative aspect-video max-h-[62vh] bg-[var(--field)]">
                            <video
                                ref={videoRef}
                                className="absolute inset-0 h-full w-full object-cover"
                                style={{
                                    transform: "scaleX(-1)",
                                    filter: "grayscale(1) brightness(0.4) contrast(1.15)",
                                    opacity: isRunning ? 1 : 0,
                                }}
                                autoPlay
                                playsInline
                                muted
                            />
                            <canvas
                                ref={canvasRef}
                                className="pointer-events-none absolute inset-0 h-full w-full"
                            />
                            {!isRunning && <div className="absolute inset-0 grid-field opacity-50" />}

                            {/* Recognised label, over the feed */}
                            {isRunning && (
                                <div className="absolute left-4 top-4">
                                    <span className="label">Recognised</span>
                                    <p
                                        className="display mt-1 text-3xl lg:text-4xl"
                                        style={{ color: detected ? "var(--signal)" : "var(--ink-3)" }}
                                    >
                                        {detected ? gesture!.gestureName.replace(/_/g, " ") : "no hand"}
                                    </p>
                                    {detected && (
                                        <p className="readout mt-1 text-xs text-[var(--ink-2)]">
                                            {gesture!.confidence.toFixed(3)} · {gesture!.handedness}
                                            {gesture!.phase !== "idle" && ` · ${gesture!.phase}`}
                                        </p>
                                    )}

                                    {/* Two-handed pair, when one is formed */}
                                    {gesture?.combo && (
                                        <div className="mt-3 border-l-2 border-[var(--signal)] pl-3">
                                            <span className="label label-signal">Combo</span>
                                            <p className="mono mt-1 text-sm text-[var(--ink)]">
                                                {gesture.combo.left.replace(/_/g, " ")}
                                                {" + "}
                                                {gesture.combo.right.replace(/_/g, " ")}
                                            </p>
                                            <p className="readout text-[10px] text-[var(--ink-3)]">
                                                {gesture.combo.confidence.toFixed(3)} · sep{" "}
                                                {gesture.combo.separation.toFixed(3)}
                                            </p>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Per-hand readout, only meaningful with two hands */}
                            {isRunning && (gesture?.hands.length ?? 0) > 1 && (
                                <div className="absolute right-4 top-4 border border-[var(--rule)] bg-[var(--field)]/85 px-3 py-2">
                                    <span className="label">Hands</span>
                                    {gesture!.hands.map((h) => (
                                        <div
                                            key={h.handedness}
                                            className="mt-1.5 flex items-baseline justify-between gap-4"
                                        >
                                            <span className="label">{h.handedness}</span>
                                            <span
                                                className="mono text-[11px]"
                                                style={{
                                                    color: h.rejected
                                                        ? "var(--ink-3)"
                                                        : "var(--signal)",
                                                }}
                                            >
                                                {h.rejected ? "—" : h.gestureName.replace(/_/g, " ")}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Wrist coordinate, a real streaming value */}
                            {isRunning && (
                                <div className="absolute bottom-3 right-3 flex gap-4 border border-[var(--rule)] bg-[var(--field)]/80 px-3 py-2">
                                    <div>
                                        <span className="label">Wrist x</span>
                                        <p className="readout text-[11px] text-[var(--signal-2)]">
                                            {wrist ? wrist.x.toFixed(4) : "—"}
                                        </p>
                                    </div>
                                    <div>
                                        <span className="label">Wrist y</span>
                                        <p className="readout text-[11px] text-[var(--signal-2)]">
                                            {wrist ? wrist.y.toFixed(4) : "—"}
                                        </p>
                                    </div>
                                </div>
                            )}

                            {isRunning && isLocked && (
                                <div className="absolute inset-0 z-20 bg-[var(--field)]/40">
                                    <BiometricGuard
                                        gesture={gesture}
                                        sequence={signatureFor(vocabulary)}
                                        onUnlocked={() => setIsLocked(false)}
                                    />
                                </div>
                            )}

                            {!isRunning && (
                                <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
                                    <p className="label max-w-xs text-center leading-relaxed">
                                        {activeBundle.name} · {activeBundle.classes} classes · runs on this device
                                    </p>
                                    <button onClick={startCamera} className="btn btn-signal">
                                        Start camera
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Frame time trace */}
                        <div className="border-t border-[var(--rule)] px-4 py-3">
                            <div className="mb-2 flex items-center justify-between">
                                <span className="label">Frame time, last {LATENCY_WINDOW}</span>
                                <span className="label">ceiling 60 ms</span>
                            </div>
                            <Sparkline values={latencyHistory} max={60} height={28} />

                            {/* Where the frame actually went. Detection and
                                classification have entirely different remedies,
                                so a single total cannot be acted on. */}
                            <div className="mt-3 flex items-baseline gap-6">
                                <div className="flex items-baseline gap-2">
                                    <span className="label">Landmarks</span>
                                    <span
                                        className="readout text-xs"
                                        style={{
                                            color:
                                                timing.detect > 60
                                                    ? "var(--alert)"
                                                    : "var(--signal)",
                                        }}
                                    >
                                        {timing.detect.toFixed(1)} ms
                                    </span>
                                </div>
                                <div className="flex items-baseline gap-2">
                                    <span className="label">Classify</span>
                                    <span className="readout text-xs text-[var(--ink-2)]">
                                        {timing.classify.toFixed(1)} ms
                                    </span>
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* ── Right rail ─────────────────────── */}
                    <div className="flex flex-col gap-5">
                        {/* Vocabulary */}
                        <section className="panel">
                            <div className="flex items-center justify-between border-b border-[var(--rule)] px-4 py-2.5">
                                <span className="label">Vocabulary</span>
                                <span className="label">
                                    {vocabulary.length || activeBundle.classes} classes
                                </span>
                            </div>
                            <div className="max-h-[220px] overflow-y-auto p-3">
                                <div className="flex flex-wrap gap-1.5">
                                    {(vocabulary.length ? vocabulary : []).map((label) => {
                                        const active = detected && gesture!.gestureName === label;
                                        return (
                                            <span
                                                key={label}
                                                className="mono border px-2 py-1 text-[10px] transition-colors"
                                                style={{
                                                    borderRadius: 2,
                                                    borderColor: active ? "var(--signal)" : "var(--rule)",
                                                    color: active ? "var(--signal)" : "var(--ink-3)",
                                                    background: active ? "var(--signal-4)" : "transparent",
                                                }}
                                            >
                                                {label}
                                            </span>
                                        );
                                    })}
                                    {!vocabulary.length && (
                                        <span className="label">Loads with the model</span>
                                    )}
                                </div>
                            </div>
                        </section>

                        {/* Tracking stability */}
                        <section className="panel px-4 py-4">
                            <div className="mb-3 flex items-baseline justify-between">
                                <span className="label">Tracking stability</span>
                                <span className="readout text-lg text-[var(--ink)]">
                                    {(calibration.stability * 100).toFixed(0)}
                                    <span className="ml-0.5 text-[10px] text-[var(--ink-3)]">%</span>
                                </span>
                            </div>
                            <div className="meter">
                                <div
                                    className="meter-fill"
                                    style={{ width: `${calibration.stability * 100}%` }}
                                />
                            </div>
                            <div className="mt-4 grid grid-cols-2 gap-4">
                                <div>
                                    <span className="label">Lighting</span>
                                    <p className="readout text-xs text-[var(--ink-2)]">
                                        {(calibration.lighting * 100).toFixed(0)}%
                                    </p>
                                </div>
                                <div>
                                    <span className="label">Jitter</span>
                                    <p className="readout text-xs text-[var(--ink-2)]">
                                        {calibration.latencyJitter.toFixed(2)}
                                    </p>
                                </div>
                            </div>
                        </section>

                        {/* Fusion */}
                        <section className="panel px-4 py-4">
                            <div className="mb-3 flex items-center justify-between">
                                <span className="label">Gesture + confirmation</span>
                                <StatusFlag live={isVoiceActive} label={isVoiceActive ? "Listening" : "Mic idle"} />
                            </div>

                            {/* Every fused action accepts either channel, so
                                speech is never the only way to confirm one. */}
                            <div className="mb-3 flex items-center justify-between gap-3 border border-[var(--rule)] px-3 py-2">
                                <label className="label flex cursor-pointer items-center gap-2">
                                    <input
                                        type="checkbox"
                                        checked={faceEnabled}
                                        onChange={(e) => setFaceEnabled(e.target.checked)}
                                        className="accent-[var(--signal)]"
                                    />
                                    Face marker
                                </label>
                                <span className="readout text-xs text-[var(--ink-3)]">
                                    {!faceEnabled
                                        ? "off"
                                        : !faceReady
                                          ? "loading…"
                                          : facialMarker
                                            ? facialMarker.replace("_", " ")
                                            : "neutral"}
                                </span>
                            </div>

                            <FusionMonitor lastFusedAction={lastFusedAction} />
                            {activeMacro && (
                                <p className="readout mt-3 text-xs text-[var(--signal)]">
                                    macro · {activeMacro}
                                </p>
                            )}
                        </section>

                        {/* Action log */}
                        <section className="panel flex min-h-[150px] flex-col">
                            <div className="flex items-center justify-between border-b border-[var(--rule)] px-4 py-2.5">
                                <span className="label">Action log</span>
                                <span className="label">{recentActions.length}</span>
                            </div>
                            <div className="flex-1 overflow-y-auto px-4 py-3">
                                {recentActions.length === 0 ? (
                                    <p className="label">
                                        {isLocked ? "Unlock to dispatch actions" : "No actions yet"}
                                    </p>
                                ) : (
                                    <ul className="space-y-2">
                                        {recentActions.map((item, i) => (
                                            <li
                                                key={`${item.timestamp}-${i}`}
                                                className="flex items-baseline justify-between border-l border-[var(--signal-2)] pl-3"
                                            >
                                                <span className="mono text-[11px] text-[var(--ink)]">
                                                    {item.action.name}
                                                </span>
                                                <span className="readout text-[10px] text-[var(--ink-4)]">
                                                    {new Date(item.timestamp).toLocaleTimeString([], {
                                                        hour12: false,
                                                    })}
                                                </span>
                                            </li>
                                        ))}
                                    </ul>
                                )}
                            </div>
                        </section>

                        {/* Modules */}
                        <div className="grid grid-cols-2 gap-2">
                            <button
                                onClick={() => (isRunning ? setIsStudioOpen(true) : setError("Start the camera first."))}
                                className="btn"
                            >
                                Studio
                            </button>
                            <button
                                onClick={() => (isRunning ? setIsMapperOpen(true) : setError("Start the camera first."))}
                                className="btn"
                            >
                                Mapper
                            </button>
                            <button onClick={() => setIsMacroOpen(true)} className="btn">
                                Macros
                            </button>
                            <button
                                onClick={() =>
                                    isRunning ? setIsMotionOpen(true) : setError("Start the camera first.")
                                }
                                className="btn"
                            >
                                Motion
                            </button>
                            <button onClick={() => setIsBridgeOpen(true)} className="btn">
                                Desktop
                            </button>
                            <button
                                onClick={() =>
                                    isRunning ? setIsCalibrating(true) : setError("Start the camera first.")
                                }
                                className="btn"
                            >
                                {biometricEngine.isCalibrated() ? "Recalibrate" : "Calibrate"}
                            </button>
                            <button
                                onClick={() => { stopCamera(); setIsLocked(true); }}
                                className="btn col-span-2"
                                style={{ borderColor: "var(--alert)", color: "var(--alert)" }}
                            >
                                Stop &amp; lock
                            </button>
                        </div>
                    </div>
                </div>

                {/* Spatial deck */}
                <section className="panel mt-5 p-5">
                    <SpatialDeck />
                </section>
            </div>

            {/* Notices */}
            {error && (
                <div
                    className="fixed bottom-5 right-5 z-[100] border bg-[var(--field-1)] px-4 py-3"
                    style={{ borderColor: "var(--alert)", borderRadius: 2 }}
                >
                    <span className="mono text-[11px] text-[var(--ink)]">{error}</span>
                </div>
            )}
        </main>
    );
}
