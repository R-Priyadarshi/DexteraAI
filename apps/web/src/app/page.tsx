"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { GestureEngine, type GestureResult } from "@/lib/gesture-engine";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<GestureEngine | null>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [gesture, setGesture] = useState<GestureResult | null>(null);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
          frameRate: { ideal: 60 },
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // Initialize gesture engine
      const engine = new GestureEngine();
      await engine.initialize();
      engineRef.current = engine;

      setIsRunning(true);
      setError(null);
      runInferenceLoop();
    } catch (err) {
      setError(
        `Camera access denied or unavailable: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    engineRef.current?.dispose();
    engineRef.current = null;
    setIsRunning(false);
    setGesture(null);
  }, []);

  const runInferenceLoop = useCallback(() => {
    let frameCount = 0;
    let lastFpsTime = performance.now();

    const loop = async () => {
      if (
        !videoRef.current ||
        !canvasRef.current ||
        !engineRef.current ||
        videoRef.current.readyState < 2
      ) {
        if (engineRef.current) requestAnimationFrame(loop);
        return;
      }

      const t0 = performance.now();
      const result = await engineRef.current.processFrame(videoRef.current);
      const t1 = performance.now();

      setLatency(Math.round(t1 - t0));

      if (result) {
        setGesture(result);
        drawLandmarks(canvasRef.current, result);
      }

      // FPS counter
      frameCount++;
      if (t1 - lastFpsTime >= 1000) {
        setFps(frameCount);
        frameCount = 0;
        lastFpsTime = t1;
      }

      requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);
  }, []);

  const drawLandmarks = (
    canvas: HTMLCanvasElement,
    result: GestureResult
  ) => {
    const ctx = canvas.getContext("2d");
    if (!ctx || !result.landmarks) return;

    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw landmarks as circles
    ctx.fillStyle = "#6366f1";
    for (const lm of result.landmarks) {
      ctx.beginPath();
      ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8">
      {/* Header */}
      <div className="mb-8 text-center">
        <h1 className="text-5xl font-bold tracking-tight">
          Dextera<span className="text-indigo-500">AI</span>
        </h1>
        <p className="mt-2 text-gray-400">
          Real-time gesture intelligence — on device, private, instant
        </p>
      </div>

      {/* Video + Canvas Container */}
      <div className="relative w-full max-w-3xl overflow-hidden rounded-2xl border border-gray-800 bg-gray-950">
        <video
          ref={videoRef}
          className="w-full mirror"
          style={{ transform: "scaleX(-1)" }}
          playsInline
          muted
        />
        <canvas ref={canvasRef} className="landmark-canvas" />

        {/* Overlay: gesture label */}
        {gesture && (
          <div className="absolute left-4 top-4 rounded-lg bg-black/60 px-4 py-2 backdrop-blur-sm">
            <p className="text-2xl font-bold text-indigo-400">
              {gesture.gestureName}
            </p>
            <p className="text-sm text-gray-400">
              Confidence: {(gesture.confidence * 100).toFixed(1)}%
            </p>
          </div>
        )}

        {/* Overlay: performance stats */}
        <div className="absolute bottom-4 right-4 rounded-lg bg-black/60 px-3 py-1 text-xs text-gray-400 backdrop-blur-sm">
          {fps} FPS · {latency}ms
        </div>
      </div>

      {/* Controls */}
      <div className="mt-6 flex gap-4">
        {!isRunning ? (
          <button
            onClick={startCamera}
            className="rounded-lg bg-indigo-600 px-6 py-3 font-semibold transition hover:bg-indigo-500"
          >
            Start Gesture Detection
          </button>
        ) : (
          <button
            onClick={stopCamera}
            className="rounded-lg bg-red-600 px-6 py-3 font-semibold transition hover:bg-red-500"
          >
            Stop
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mt-4 rounded-lg bg-red-900/30 px-4 py-2 text-red-400">
          {error}
        </div>
      )}

      {/* Privacy Notice */}
      <p className="mt-8 text-center text-xs text-gray-600">
        🛡 All processing happens on your device. No video or images leave your
        browser. Zero cloud. Zero tracking.
      </p>
    </main>
  );
}
