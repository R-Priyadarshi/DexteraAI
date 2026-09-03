"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { GestureEngine, type GestureResult } from "@/lib/gesture-engine";
import { FALLBACK_CONNECTIONS } from "./HandPlate";

export type ViewportStatus =
  | "idle"
  | "requesting"
  | "loading"
  | "live"
  | "denied"
  | "error";

export interface ViewportTelemetry {
  result: GestureResult | null;
  fps: number;
  latencyHistory: number[];
  frames: number;
}

interface LiveViewportProps {
  modelUrl?: string;
  /** Called on every processed frame, so parents can render their own readouts. */
  onTelemetry?: (t: ViewportTelemetry) => void;
  onStatusChange?: (s: ViewportStatus) => void;
  /** Mirror the feed, which matches how people expect to see themselves. */
  mirror?: boolean;
  className?: string;
}

const LATENCY_WINDOW = 60;

/**
 * Camera acquisition + skeleton overlay.
 *
 * The video itself is deliberately held back — dimmed and desaturated — so the
 * tracked skeleton is the subject of the frame rather than the person. What the
 * model sees is what gets drawn.
 */
export function LiveViewport({
  modelUrl = "/onnx/hagrid/gesture.onnx",
  onTelemetry,
  onStatusChange,
  mirror = true,
  className = "",
}: LiveViewportProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<GestureEngine | null>(null);
  const rafRef = useRef<number | null>(null);
  const runningRef = useRef(false);

  const frameTimesRef = useRef<number[]>([]);
  const latencyRef = useRef<number[]>([]);
  const framesRef = useRef(0);

  const [status, setStatus] = useState<ViewportStatus>("idle");
  const [message, setMessage] = useState<string>("");

  const setStatusSafe = useCallback(
    (s: ViewportStatus) => {
      setStatus(s);
      onStatusChange?.(s);
    },
    [onStatusChange]
  );

  /** Draw the skeleton the model is actually working from. */
  const draw = useCallback(
    (result: GestureResult | null) => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;

      const w = video.videoWidth || 640;
      const h = video.videoHeight || 480;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, w, h);

      const landmarks = result?.landmarks;
      if (!landmarks?.length) return;

      const signal = "#ffb627";
      const px = (p: { x: number; y: number }) => ({ x: p.x * w, y: p.y * h });

      // Bones
      ctx.strokeStyle = signal;
      ctx.lineWidth = Math.max(1.5, w / 480);
      ctx.lineCap = "round";
      ctx.beginPath();
      for (const [a, b] of FALLBACK_CONNECTIONS) {
        const p = landmarks[a];
        const q = landmarks[b];
        if (!p || !q) continue;
        const pa = px(p);
        const qb = px(q);
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(qb.x, qb.y);
      }
      ctx.stroke();

      // Joints
      const tips = new Set([4, 8, 12, 16, 20]);
      for (let i = 0; i < landmarks.length; i++) {
        const p = px(landmarks[i]);
        const r = i === 0 ? 5 : tips.has(i) ? 4 : 2.5;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r * (w / 640), 0, Math.PI * 2);
        if (i === 0 || tips.has(i)) {
          ctx.fillStyle = signal;
          ctx.fill();
        } else {
          ctx.fillStyle = "#0a0a0b";
          ctx.fill();
          ctx.strokeStyle = signal;
          ctx.lineWidth = 1.2 * (w / 640);
          ctx.stroke();
        }
      }

      // Bounding extent of the tracked hand — a real measurement, drawn as
      // the crop the classifier reasons about.
      let minX = 1;
      let minY = 1;
      let maxX = 0;
      let maxY = 0;
      for (const p of landmarks) {
        minX = Math.min(minX, p.x);
        minY = Math.min(minY, p.y);
        maxX = Math.max(maxX, p.x);
        maxY = Math.max(maxY, p.y);
      }
      const pad = 0.04;
      const bx = (minX - pad) * w;
      const by = (minY - pad) * h;
      const bw = (maxX - minX + pad * 2) * w;
      const bh = (maxY - minY + pad * 2) * h;

      ctx.strokeStyle = "rgba(255,182,39,0.32)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(bx, by, bw, bh);
      ctx.setLineDash([]);
    },
    []
  );

  const stop = useCallback(() => {
    runningRef.current = false;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    const video = videoRef.current;
    if (video?.srcObject) {
      (video.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    engineRef.current?.dispose();
    engineRef.current = null;
    setStatusSafe("idle");
  }, [setStatusSafe]);

  const start = useCallback(async () => {
    if (runningRef.current) return;

    // Declared outside the try so the failure paths below can release it.
    // `engine.initialize` throws deliberately when WebGL is unavailable, and
    // an abandoned stream leaves the camera light on with nothing using it.
    let stream: MediaStream | null = null;

    try {
      setStatusSafe("requesting");
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });

      const video = videoRef.current;
      if (!video) {
        // Unmounted between the permission prompt and the grant.
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      video.srcObject = stream;
      await video.play();

      setStatusSafe("loading");
      const engine = new GestureEngine();
      await engine.initialize(modelUrl);
      engineRef.current = engine;

      runningRef.current = true;
      setStatusSafe("live");

      const loop = async () => {
        if (!runningRef.current) return;
        const v = videoRef.current;
        const eng = engineRef.current;

        if (v && eng && v.readyState >= 2) {
          try {
            const result = await eng.processFrame(v);

            const now = performance.now();
            frameTimesRef.current.push(now);
            while (
              frameTimesRef.current.length > 0 &&
              now - frameTimesRef.current[0] > 1000
            ) {
              frameTimesRef.current.shift();
            }

            if (result) {
              latencyRef.current.push(result.inferenceTimeMs);
              if (latencyRef.current.length > LATENCY_WINDOW) {
                latencyRef.current.shift();
              }
              framesRef.current += 1;
              draw(result.landmarks?.length ? result : null);
              onTelemetry?.({
                result,
                fps: frameTimesRef.current.length,
                latencyHistory: [...latencyRef.current],
                frames: framesRef.current,
              });
            }
          } catch {
            // A dropped frame is not a failure worth interrupting the loop for.
          }
        }
        rafRef.current = requestAnimationFrame(() => void loop());
      };

      rafRef.current = requestAnimationFrame(() => void loop());
    } catch (err) {
      stream?.getTracks().forEach((t) => t.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
      engineRef.current?.dispose();
      engineRef.current = null;
      runningRef.current = false;

      const name = (err as { name?: string })?.name;
      if (name === "NotAllowedError" || name === "SecurityError") {
        setStatusSafe("denied");
        setMessage("Camera access denied. Nothing is sent anywhere — the model runs here.");
      } else if (name === "NotFoundError") {
        setStatusSafe("error");
        setMessage("No camera found on this device.");
      } else {
        setStatusSafe("error");
        setMessage((err as Error)?.message ?? "Could not start the camera.");
      }
    }
  }, [draw, modelUrl, onTelemetry, setStatusSafe]);

  useEffect(() => () => stop(), [stop]);

  const isLive = status === "live";
  const isBusy = status === "requesting" || status === "loading";

  return (
    <div className={`relative overflow-hidden bg-[var(--field-1)] ${className}`}>
      <video
        ref={videoRef}
        playsInline
        muted
        className="absolute inset-0 h-full w-full object-cover"
        style={{
          transform: mirror ? "scaleX(-1)" : undefined,
          // The person is context; the skeleton is the subject.
          filter: "grayscale(1) brightness(0.42) contrast(1.15)",
          opacity: isLive ? 1 : 0,
          transition: "opacity 400ms ease",
        }}
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full object-cover"
        style={{ transform: mirror ? "scaleX(-1)" : undefined }}
      />

      {/* Reference grid, only while acquiring */}
      {isLive && (
        <div className="pointer-events-none absolute inset-0 grid-field opacity-40" />
      )}

      {!isLive && (
        <div className="absolute inset-0 grid-field opacity-50" />
      )}

      {/* Idle / permission / error states */}
      {!isLive && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-5 px-8 text-center">
          {isBusy ? (
            <>
              <div className="live-dot" />
              <p className="label">
                {status === "requesting" ? "Awaiting camera permission" : "Loading model — 2.1 MB"}
              </p>
            </>
          ) : (
            <>
              <p className="label max-w-xs leading-relaxed">
                {status === "denied" || status === "error"
                  ? message
                  : "Nothing here is a video. Start the camera and the model runs in this tab."}
              </p>
              <button onClick={() => void start()} className="btn btn-signal">
                {status === "denied" || status === "error" ? "Try again" : "Start camera"}
              </button>
            </>
          )}
        </div>
      )}

      {/* Stop control, once acquiring */}
      {isLive && (
        <button
          onClick={stop}
          className="absolute bottom-3 right-3 z-10 border border-[var(--rule-strong)] bg-[var(--field)]/80 px-3 py-1.5 label hover:text-[var(--ink)]"
          style={{ borderRadius: 2 }}
        >
          Stop
        </button>
      )}
    </div>
  );
}
