"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { GestureEngine, type GestureResult } from "@/lib/gesture-engine";
import { ActionRegistry, type GestureAction } from "@/lib/action-registry";
import { PluginEngine } from "@/lib/plugin-engine";
import { PresentationPlugin } from "@/lib/plugins/presentation";
import { BiometricGuard } from "@/components/BiometricGuard";
import { GestureStudio } from "@/components/GestureStudio";
import { MacroComposer } from "@/components/MacroComposer";
import { macroEngine } from "@/lib/macro-engine";
import { voiceEngine, type VoiceIntent } from "@/lib/voice-engine";
import { intentRefinery, type FusedAction } from "@/lib/intent-refinery";
import { hapticEngine } from "@/lib/haptic-engine";
import { tacticalAudio } from "@/lib/tactical-audio";
import { FusionMonitor } from "@/components/FusionMonitor";
import { calibrator, type CalibrationMetrics } from "@/lib/calibrator";
import { SpatialDeck } from "@/components/SpatialDeck";
import { ActionMapper } from "@/components/ActionMapper";
import { CalibrationWizard } from "@/components/CalibrationWizard";
import { biometricEngine } from "@/lib/biometric-engine";

/**
 * Trained model bundles. Each directory holds gesture.onnx + labels.json, so the
 * vocabulary travels with the model rather than being hardcoded here.
 * Produced by: python dextera.py train --export models/<name>
 */
const MODEL_BUNDLES = [
    { id: "hagrid", name: "General Gestures", url: "/onnx/hagrid/gesture.onnx" },
    { id: "asl_alphabet", name: "ASL Fingerspelling", url: "/onnx/asl_alphabet/gesture.onnx" },
] as const;

type BundleId = (typeof MODEL_BUNDLES)[number]["id"];

export default function Home() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const engineRef = useRef<GestureEngine | null>(null);
    const actionRegistry = ActionRegistry.getInstance();
    const pluginEngine = PluginEngine.getInstance();

    // Neural Versioning: Synchronous Initialization
    const BUILD_ID = "v5.8.5-FINAL";
    if (typeof window !== "undefined") {
        (window as any).DEXTERA_BUILD = BUILD_ID;
    }

    const [isRunning, setIsRunning] = useState(false);
    const [bundleId, setBundleId] = useState<BundleId>("hagrid");
    const [vocabulary, setVocabulary] = useState<string[]>([]);
    const [gesture, setGesture] = useState<GestureResult | null>(null);
    const [fps, setFps] = useState(0);
    const [latency, setLatency] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [engineStatus, setEngineStatus] = useState("OFFLINE");
    const isProcessingRef = useRef(false);
    const lastResultRef = useRef<GestureResult | null>(null);

    const [recentActions, setRecentActions] = useState<{ action: GestureAction, timestamp: number }[]>([]);
    const [isLocked, setIsLocked] = useState(true);
    const [isStudioOpen, setIsStudioOpen] = useState(false);
    const [isMacroOpen, setIsMacroOpen] = useState(false);
    const [isMapperOpen, setIsMapperOpen] = useState(false);
    const [isCalibrating, setIsCalibrating] = useState(false);
    const [isActionActive, setIsActionActive] = useState(false);
    const [memoryLog, setMemoryLog] = useState("INIT_CORE...");
    const heatmapEnabledRef = useRef(false);
    const [heatmapEnabledState, setHeatmapEnabledState] = useState(false); // For UI sync
    const heatmapBufferRef = useRef<{ x: number, y: number, t: number }[]>([]);
    const [activeMacro, setActiveMacro] = useState<string | null>(null);
    const [voiceIntent, setVoiceIntent] = useState<VoiceIntent | null>(null);
    const [isVoiceActive, setIsVoiceActive] = useState(false);
    const [isHapticActive, setIsHapticActive] = useState(false);
    const [lastFusedAction, setLastFusedAction] = useState<FusedAction | null>(null);
    const isLockedRef = useRef(isLocked);
    const voiceIntentRef = useRef(voiceIntent);
    
    useEffect(() => { isLockedRef.current = isLocked; }, [isLocked]);
    useEffect(() => { voiceIntentRef.current = voiceIntent; }, [voiceIntent]);
    const [calibration, setCalibration] = useState<CalibrationMetrics>({ stability: 1, lighting: 1, latencyJitter: 0 });

    // Neural Chromatics: Dynamic background glow based on system state
    const getGlowColor = () => {
        if (isLocked) return "rgba(239, 68, 68, 0.05)"; // Red alert
        if (isCalibrating) return "rgba(245, 158, 11, 0.08)"; // Amber calibration
        if (isActionActive) return "rgba(52, 211, 153, 0.12)"; // Emerald success
        return "rgba(6, 182, 212, 0.05)"; // Industrial Cyan
    };

    const startCamera = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "user",
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                stream.getVideoTracks().forEach(track => {
                    console.log(`DexteraAI: Camera track acquired [${track.label}] - Status: ${track.readyState}`);
                    track.onended = () => console.warn(`DexteraAI: Camera track TERMINATED externally [${track.label}]`);
                });
                await videoRef.current.play();
            }

            const engine = new GestureEngine();
            const bundle = MODEL_BUNDLES.find((b) => b.id === bundleId) ?? MODEL_BUNDLES[0];
            await engine.initialize(bundle.url);
            engineRef.current = engine;
            setVocabulary(engine.getLabels());

            // Initialize Voice Engine only on manual start
            voiceEngine.start({
                onIntent: (intent) => {
                    setVoiceIntent(intent);
                    setTimeout(() => setVoiceIntent(null), 2000);
                    if (intent === "stealth") console.log("DexteraAI: Voice stealth triggered (redirect disabled)");
                },
                onStatus: (status) => setIsVoiceActive(status === "listening" || status === "processing")
            });

            setIsRunning(true);
            setError(null);
            tacticalAudio.startNeuralHum();
            hapticEngine.pulse("success");
        } catch (err) {
            setError(
                `Biometric access denied: ${err instanceof Error ? err.message : String(err)}`
            );
        }
    }, [bundleId]);

    // Decommissioned erratic sync useEffect in favor of In-Loop Sync

    const stopCamera = () => {
        // 1. Release Hardware Tracks
        if (videoRef.current?.srcObject) {
            (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => {
                track.stop();
                console.log("[SYS] CAMERA_TRACK_TERMINATED:", track.label);
            });
            videoRef.current.srcObject = null;
        }
        engineRef.current = null;
        setIsRunning(false);
        setGesture(null);
        voiceEngine.stop();
        tacticalAudio.stopNeuralHum();
        hapticEngine.pulse("sonar_ping");
    };

    // Stable Refs for Persistent Loop Data
    const frameCountRef = useRef(0);
    const lastFpsTimeRef = useRef(performance.now());
    const lastStateUpdateTimeRef = useRef(0);
    const loopRef = useRef<number | null>(null);

    // Sync lock state ref for the high-speed loop
    useEffect(() => {
        isLockedRef.current = isLocked;
    }, [isLocked]);

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

            // 1. Synchronize Loop
            loopRef.current = requestAnimationFrame(loop);

            if (!isProcessingRef.current) {
                isProcessingRef.current = true;
                try {
                    const result = await engineRef.current.processFrame(video);
                    const ctx = canvas.getContext("2d");

                    if (result) {
                        // Hardware-locked rendering
                        drawLandmarks(canvas, result);
                        calibrator.record(result.landmarks);

                        const now = performance.now();
                        if (now - lastStateUpdateTimeRef.current > 100) {
                            setLatency(Math.round(result.inferenceTimeMs));
                            setGesture(result);
                            setCalibration(calibrator.getMetrics());
                            lastStateUpdateTimeRef.current = now;
                        }

                        if (!isLockedRef.current) {
                            pluginEngine.broadcast(result);
                            const macroMatch = macroEngine.process(result);
                            if (macroMatch) setActiveMacro(macroMatch.name);

                            const fusedMatch = intentRefinery.process(result, voiceIntentRef.current);
                            if (fusedMatch) {
                                setLastFusedAction(fusedMatch);
                                fusedMatch.execute();
                                hapticEngine.pulse(fusedMatch.feedbackType === "error" ? "error" : "light");
                            }
                        }
                    } else if (ctx) {
                        // No result this frame: clear the overlay.
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                    }
                    isProcessingRef.current = false;
            } catch (err) {
                console.error("DexteraAI Loop Error:", err);
                isProcessingRef.current = false;
            }
            }

            frameCountRef.current++;
            if (performance.now() - lastFpsTimeRef.current >= 1000) {
                setFps(frameCountRef.current);
                frameCountRef.current = 0;
                lastFpsTimeRef.current = performance.now();
            }
        };
        
        loopRef.current = requestAnimationFrame(loop);
        return () => {
            if (loopRef.current) cancelAnimationFrame(loopRef.current);
        };
    }, [isRunning]); // ONLY restart on hard boot/shutdown

    const drawLandmarks = (canvas: HTMLCanvasElement, result: GestureResult) => {
        const ctx = canvas.getContext("2d");
        if (!ctx || !result.landmarks || result.landmarks.length < 10) return;

        ctx.strokeStyle = "#00ffff";
        ctx.lineWidth = 4;
        ctx.shadowBlur = 10;
        ctx.shadowColor = "#00ffff";
        
        const GESTURE_CONNECTIONS = [
            [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13], [13, 17]
        ];

        GESTURE_CONNECTIONS.forEach(([i, j]) => {
            const p1 = result.landmarks![i];
            const p2 = result.landmarks![j];
            ctx.moveTo((1.0 - p1.x) * canvas.width, p1.y * canvas.height);
            ctx.lineTo((1.0 - p2.x) * canvas.width, p2.y * canvas.height);
        });
        ctx.stroke();

        result.landmarks.forEach((p) => {
            ctx.beginPath();
            ctx.arc((1.0 - p.x) * canvas.width, p.y * canvas.height, 5, 0, Math.PI * 2);
            ctx.fillStyle = "#00ffff";
            ctx.fill();
        });
    };

    useEffect(() => {
        const handleHaptic = () => {
            setIsHapticActive(true);
            setTimeout(() => setIsHapticActive(false), 150);
        };

        const handleSysHalt = () => {
            console.log("DexteraAI: SYSTEM HALT SIGNAL RECEIVED.");
            stopCamera();
            setIsLocked(true);
            setError("SYSTEM_HALT: Emergency Abort Pattern Recognized.");
        };

        const handleSysReset = () => {
            console.log("DexteraAI: SYSTEM RESET SIGNAL RECEIVED.");
            setRecentActions([]);
            setLastFusedAction(null);
            hapticEngine.pulse("success");
        };

        window.addEventListener("dextera-haptic", handleHaptic);
        window.addEventListener("dextera_sys_halt", handleSysHalt);
        window.addEventListener("dextera_sys_reset", handleSysReset);
        pluginEngine.register(PresentationPlugin);
        return () => {
            stopCamera();
            window.removeEventListener("dextera-haptic", handleHaptic);
            window.removeEventListener("dextera_sys_halt", handleSysHalt);
            window.removeEventListener("dextera_sys_reset", handleSysReset);
        };
    }, [stopCamera, pluginEngine]);



    return (
        <main className="relative flex min-h-screen w-full flex-col bg-[#020203] font-sans text-white antialiased selection:bg-cyan-500/30 overflow-x-hidden overflow-y-auto custom-scrollbar">
            {/* Neural Background Layer (Fixed) */}
            <div className="fixed inset-0 hex-grid opacity-[0.03] pointer-events-none" />
            <div className="fixed inset-0 scanlines pointer-events-none z-50 opacity-[0.05]" />
            
            {/* Reactive Neural Chromatics */}
            <div 
                className="fixed inset-0 pointer-events-none transition-all duration-1000 ease-in-out z-0"
                style={{ background: `radial-gradient(circle at 50% 50%, ${getGlowColor()} 0%, transparent 70%)` }}
            />
            
            {/* Biometric Security Overlay Removed from here */}

            {/* Gesture Studio Portal */}
            {isStudioOpen && (
                <GestureStudio gesture={gesture} onClose={() => setIsStudioOpen(false)} />
            )}

            {/* Action Mapper Portal */}
            {isMapperOpen && (
                <ActionMapper onClose={() => setIsMapperOpen(false)} />
            )}

            {/* Calibration Wizard Portal */}
            {isCalibrating && (
                <CalibrationWizard 
                    landmarks={gesture?.landmarks || null} 
                    onClose={() => setIsCalibrating(false)}
                    onComplete={() => {
                        setIsCalibrating(false);
                        setError("BIOMETRICS_CALIBRATED: Neural Signature Authorized.");
                        setTimeout(() => setError(null), 3000);
                    }}
                />
            )}

            {/* TOP BAR: Tactical Status (Premium Centered) */}
            <header className="sticky top-0 z-[60] flex h-20 w-full items-center border-b border-white/10 bg-black/80 shadow-[0_4px_30px_rgba(0,0,0,0.5)]">
                <div className="mx-auto flex w-full max-w-[1600px] items-center justify-between px-10">
                    <div className="flex items-center gap-10">
                        <div className="flex items-center gap-4">
                            <div className="relative">
                                <div className="h-3 w-3 rounded-full bg-cyan-500 shadow-[0_0_15px_rgba(0,255,255,1)] animate-pulse" />
                                <div className="absolute inset-0 h-3 w-3 rounded-full bg-cyan-400 animate-ping opacity-40" />
                            </div>
                            <div className="flex flex-col">
                                <span className="text-[14px] font-black tracking-[0.4em] uppercase text-white/90">DexteraAI</span>
                                <span className="text-[8px] font-mono text-cyan-500/60 tracking-[0.2em] uppercase">Industrial_Intelligence_Core</span>
                            </div>
                        </div>
                        <div className="h-10 w-px bg-white/10" />
                        <div className="flex flex-col border-l border-white/10 pl-10">
                            <span className="hud-label !text-[7px]">Neural_Handshake</span>
                            <button 
                                onClick={() => isRunning ? setIsCalibrating(true) : setError("Boot Core First")}
                                className={`text-[10px] font-mono tracking-widest uppercase transition-all ${biometricEngine.isCalibrated() ? "text-emerald-400/60 hover:text-emerald-400" : "text-cyan-400 hover:text-cyan-300 animate-pulse"}`}
                            >
                                {biometricEngine.isCalibrated() ? "[ RECALIBRATE ]" : "[ CALIBRATE_NOW ]"}
                            </button>
                        </div>
                    </div>

                    <div className="flex items-center gap-12">
                        <div className="flex flex-col items-end">
                            <span className="hud-label !text-[7px]">Latency</span>
                            <span className="text-xl font-extralight text-cyan-400 tracking-tighter">{latency}<span className="text-[10px] ml-1 opacity-40">MS</span></span>
                        </div>
                        <div className="flex flex-col items-end border-l border-white/10 pl-12">
                            <span className="hud-label !text-[7px]">Biometric_Auth</span>
                            <div className="flex items-center gap-2">
                                <span className={`text-[11px] font-bold tracking-widest ${isLocked ? "text-red-500" : "text-emerald-400"}`}>
                                    {isLocked ? "ENCRYPTED" : "AUTHORIZED"}
                                </span>
                                <div className={`h-1.5 w-1.5 rounded-full ${isLocked ? "bg-red-500 shadow-[0_0_8px_rgba(255,0,0,0.8)]" : "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]"}`} />
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            {/* FIXED NEURAL ANCHOR (Always Centered for Biometric Precision) */}
            <div className="relative mx-auto flex w-full max-w-[1600px] flex-1 flex-col p-10 gap-10">
                
                <div className="grid grid-cols-12 gap-10 items-start">
                    {/* 1. CENTRAL VIEWING PORTAL (THE ANCHOR) */}
                    <div 
                        className="col-span-12 xl:col-span-8 relative aspect-video max-h-[70vh] rounded-[2.5rem] overflow-hidden border transition-all duration-500 bg-black/60 group tactical-border shadow-[0_20px_50px_rgba(0,0,0,0.8)] z-50"
                        style={{
                            borderColor: isActionActive ? "rgba(0, 255, 255, 0.8)" : "rgba(255, 255, 255, 0.1)",
                            boxShadow: isActionActive ? "0 0 50px rgba(0, 255, 255, 0.2)" : "none"
                        }}
                    >
                        {/* 1. THE SENSOR CHAMBER */}
                        <div className="absolute inset-0">
                            <video
                                ref={videoRef}
                                className="h-full w-full object-contain bg-black/20"
                                style={{ transform: "scaleX(-1)" }}
                                autoPlay
                                playsInline
                                muted
                            />
                            <canvas 
                                ref={canvasRef} 
                                className="absolute inset-0 h-full w-full pointer-events-none z-[101]" 
                                style={{ transform: "scaleX(-1)" }}
                            />
                        </div>

                        {/* 2. SECURITY HANDSHAKE (Un-mirrored Overlay with Neural Purity Clarity) */}
                        {isRunning && isLocked && (
                            <div className="absolute inset-0 z-[110] pointer-events-auto bg-black/30 transition-all duration-700">
                                <BiometricGuard 
                                    gesture={gesture} 
                                    onUnlocked={() => setIsLocked(false)} 
                                />
                            </div>
                        )}

                        {/* HUD Overlays */}
                        <div className="absolute top-10 left-10 p-8 glass-panel rounded-3xl border border-white/10 bg-black/40 shadow-2xl">
                            <div className="flex items-center gap-3 mb-3">
                                <div className="h-1.5 w-1.5 rounded-full bg-cyan-500 animate-pulse" />
                                <span className="hud-label !text-[8px] !tracking-[0.5em] text-cyan-500/80">Neural_Sync_Active</span>
                            </div>
                            <h2 className="text-4xl font-extralight tracking-[-0.05em] uppercase text-white italic">
                                {gesture && gesture.gestureId !== -1 ? gesture.gestureName.replace(/_/g, ' ') : "Awaiting Input"}
                            </h2>
                        </div>

                        <div className="absolute bottom-10 right-10 flex gap-6 bg-black/40 p-5 rounded-2xl border border-white/10 shadow-2xl">
                            <div className="flex flex-col">
                                <span className="text-[7px] hud-label opacity-40 mb-1">POS_X</span>
                                <span className="text-[10px] font-mono text-cyan-400/60">{gesture?.landmarks?.[0].x.toFixed(6) || "0.000000"}</span>
                            </div>
                            <div className="flex flex-col">
                                <span className="text-[7px] hud-label opacity-40 mb-1">POS_Y</span>
                                <span className="text-[10px] font-mono text-cyan-400/60">{gesture?.landmarks?.[0].y.toFixed(6) || "0.000000"}</span>
                            </div>
                        </div>

                        {!isRunning && (
                            <div className="absolute inset-0 flex items-center justify-center bg-black/95 z-50">
                                <button
                                    onClick={startCamera}
                                    className="group relative flex flex-col items-center gap-10 transition-all hover:scale-105"
                                >
                                    <div className="relative h-24 w-24 border border-cyan-500/20 rounded-full flex items-center justify-center group-hover:border-cyan-500/60 transition-all">
                                        <div className="h-2 w-2 bg-cyan-500 rounded-full shadow-[0_0_20px_rgba(0,255,255,1)]" />
                                        <div className="absolute inset-0 border border-cyan-500/10 rounded-full animate-ping" />
                                    </div>
                                    <span className="text-[11px] font-black tracking-[0.8em] text-white/50 uppercase group-hover:text-cyan-400 transition-colors">Boot_Neural_Core</span>
                                </button>

                                {/* Vocabulary selector: each bundle ships its own labels.json */}
                                <div className="absolute bottom-12 flex flex-col items-center gap-3">
                                    <span className="text-[8px] font-mono tracking-[0.3em] text-white/30 uppercase">
                                        Vocabulary
                                    </span>
                                    <div className="flex gap-2">
                                        {MODEL_BUNDLES.map((bundle) => (
                                            <button
                                                key={bundle.id}
                                                onClick={() => setBundleId(bundle.id)}
                                                className={`rounded-full border px-4 py-1.5 text-[9px] font-bold uppercase tracking-[0.15em] transition-all ${
                                                    bundleId === bundle.id
                                                        ? "border-cyan-500/60 bg-cyan-500/10 text-cyan-300"
                                                        : "border-white/10 text-white/40 hover:border-white/30 hover:text-white/70"
                                                }`}
                                            >
                                                {bundle.name}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}

                        {isRunning && vocabulary.length > 0 && (
                            <div className="absolute bottom-3 left-3 z-40 rounded-full border border-white/10 bg-black/60 px-3 py-1">
                                <span className="text-[8px] font-mono uppercase tracking-[0.2em] text-white/40">
                                    {MODEL_BUNDLES.find((b) => b.id === bundleId)?.name} · {vocabulary.length} classes
                                </span>
                            </div>
                        )}
                    </div>

                    {/* 2. TACTICAL SIDEBAR (Optimized Density) */}
                    <div className="col-span-12 xl:col-span-4 space-y-8">
                        {/* Command Log */}
                        <div className="spatial-card rounded-[2.5rem] p-8 flex flex-col min-h-[300px] max-h-[350px] bitstream-bg">
                            <h4 className="hud-label mb-6 border-b border-white/10 pb-4 flex justify-between">
                                <span>Command_Log</span>
                                <span className="text-cyan-500/30">STREAM_ON</span>
                            </h4>
                            <div className="flex-1 overflow-y-auto space-y-5 pr-2 custom-scrollbar text-[10px] font-mono">
                                <div className="text-cyan-500/40">[SYS] INFRA_READY_{memoryLog}</div>
                                <div className="text-cyan-500/40">[SYS] NEURAL_CORE_INF_LATENCY_{latency || 12}MS</div>
                                {recentActions.map((item, i) => (
                                    <div key={`${item.timestamp}-${i}`} className="group flex flex-col gap-1 border-l border-cyan-500/50 pl-5 animate-in slide-in-from-right duration-500">
                                        <div className="flex justify-between items-center">
                                            <span className="font-bold text-white uppercase tracking-wider">{item.action.name}</span>
                                            <span className="text-[7px] text-cyan-500/30 font-mono tracking-tighter">EXEC_OK</span>
                                        </div>
                                        <span className="text-[8px] opacity-30">[{new Date(item.timestamp).toLocaleTimeString([], { hour12: false, second: '2-digit' })}]</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Fusion Monitor (Multi-modal Sync) */}
                        <div className="spatial-card rounded-[2.5rem] p-8 border-cyan-500/10 shadow-2xl">
                            <FusionMonitor lastFusedAction={lastFusedAction} />
                        </div>

                        {/* Stability Index */}
                        <div className="spatial-card rounded-[2.5rem] p-8">
                            <h4 className="hud-label mb-5 !text-[7px] opacity-40 tracking-[0.3em]">Stability_Link</h4>
                            <div className="flex items-center gap-6">
                                <span className="text-5xl font-extralight tracking-tighter text-white">{(calibration.stability * 100).toFixed(0)}<span className="text-lg opacity-20 ml-1">%</span></span>
                                <div className="flex-1 h-[2px] bg-white/5 rounded-full overflow-hidden">
                                    <div className="h-full bg-cyan-500 shadow-[0_0_15px_rgba(0,255,255,0.6)] transition-all duration-1000" style={{ width: `${calibration.stability * 100}%` }} />
                                </div>
                            </div>
                        </div>

                        {/* Control Cluster */}
                        <div className="grid grid-cols-2 gap-4">
                            <button onClick={() => isRunning ? setIsStudioOpen(true) : setError("System Locked")} className="spatial-card rounded-3xl py-6 hover:bg-cyan-500/10 transition-all group border-white/5">
                                <span className="text-[9px] font-black uppercase tracking-[0.4em] text-white/30 group-hover:text-cyan-400">Studio</span>
                            </button>
                            <button onClick={() => isRunning ? setIsMapperOpen(true) : setError("System Locked")} className="spatial-card rounded-3xl py-6 hover:bg-cyan-500/10 transition-all group border-white/5">
                                <span className="text-[9px] font-black uppercase tracking-[0.4em] text-white/30 group-hover:text-cyan-400">Mapper</span>
                            </button>
                            <button
                                onClick={() => {
                                    heatmapEnabledRef.current = !heatmapEnabledRef.current;
                                    setHeatmapEnabledState(heatmapEnabledRef.current);
                                }}
                                className={`spatial-card rounded-3xl py-6 transition-all group ${heatmapEnabledState ? "border-cyan-500/50 bg-cyan-500/10 shadow-[0_0_20px_rgba(0,255,255,0.1)]" : "border-white/5 hover:bg-cyan-500/10"}`}
                            >
                                <span className={`text-[9px] font-black uppercase tracking-[0.4em] ${heatmapEnabledState ? "text-cyan-400" : "text-white/30 group-hover:text-cyan-400"}`}>Heatmap</span>
                            </button>
                            <button
                                onClick={() => { stopCamera(); setIsLocked(true); }}
                                className="spatial-card rounded-3xl py-6 hover:bg-red-500/20 transition-all group border-red-500/10"
                            >
                                <span className="text-[9px] font-black uppercase tracking-[0.4em] text-white/20 group-hover:text-red-500">Secure</span>
                            </button>
                        </div>
                    </div>
                </div>

                {/* 3. EXPANSIVE SPATIAL DECK */}
                <div className="spatial-card rounded-[3rem] p-12 shadow-[0_30px_70px_rgba(0,0,0,0.9)] border-white/5">
                    <SpatialDeck />
                </div>
            </div>

            {/* Error Notifications */}
            {error && (
                <div className="z-[100] fixed top-20 right-8 glass-panel rounded-xl border-red-500/30 bg-black/80 px-6 py-3 backdrop-blur-3xl animate-in fade-in slide-in-from-right">
                    <div className="flex items-center gap-3">
                        <div className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" />
                        <span className="text-[10px] font-bold tracking-widest text-white/80 uppercase">{error}</span>
                    </div>
                </div>
            )}
        </main>
    );
}
