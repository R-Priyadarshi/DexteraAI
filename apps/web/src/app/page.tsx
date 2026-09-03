"use client";

import Link from "next/link";
import { useCallback, useState } from "react";
import { HandPlate, useGestureAtlas } from "@/components/HandPlate";
import { LiveViewport, type ViewportTelemetry } from "@/components/LiveViewport";
import { FieldRow, Sparkline, Stat, StatusFlag } from "@/components/Telemetry";

/* Measured, not claimed. Every figure here traces to reports/eval.json or the
   exported model bundle — see models/hagrid/labels.json. */
const MEASURED = {
  gestures: 18,
  aslLetters: 26,
  testAccuracy: "98.1",
  aslAccuracy: "93.8",
  modelSizeMb: "2.1",
  classifierMs: "0.07",
  landmarks: 21,
  windowFrames: 30,
  features: 86,
  params: "550,731",
  trainingSamples: "65,802",
};

const PIPELINE = [
  { step: "01", name: "Camera", detail: "640×480 · 30 fps" },
  { step: "02", name: "Landmarks", detail: "21 points, x y z" },
  { step: "03", name: "Features", detail: "86 per frame" },
  { step: "04", name: "Window", detail: "30 frames buffered" },
  { step: "05", name: "Transformer", detail: "550k params" },
  { step: "06", name: "Label", detail: "18 classes + reject" },
];

export default function LandingPage() {
  const { atlas } = useGestureAtlas();
  const [telemetry, setTelemetry] = useState<ViewportTelemetry | null>(null);
  const [live, setLive] = useState(false);

  const handleTelemetry = useCallback((t: ViewportTelemetry) => setTelemetry(t), []);
  const handleStatus = useCallback((s: string) => setLive(s === "live"), []);

  const result = telemetry?.result ?? null;
  const detected = Boolean(result?.landmarks?.length);
  const latency = telemetry?.latencyHistory ?? [];
  const avgLatency =
    latency.length > 0
      ? (latency.reduce((a, b) => a + b, 0) / latency.length).toFixed(0)
      : null;

  return (
    <div className="min-h-screen">
      {/* ── Nav ────────────────────────────────────────── */}
      <nav className="sticky top-0 z-50 border-b border-[var(--rule)] bg-[var(--field)]/92 backdrop-blur-[2px]">
        <div className="mx-auto flex max-w-[1400px] items-center justify-between px-6 py-4 lg:px-10">
          <div className="flex items-baseline gap-3">
            <span className="display text-[15px] tracking-[-0.02em] text-[var(--ink)]">
              Dextera
            </span>
            <span className="label hidden sm:inline">Gesture recognition</span>
          </div>
          <div className="flex items-center gap-6">
            <a href="#vocabulary" className="label hover:text-[var(--ink)]">
              Vocabulary
            </a>
            <a href="#pipeline" className="label hidden hover:text-[var(--ink)] sm:inline">
              Pipeline
            </a>
            <Link href="/dashboard" className="label label-signal hover:opacity-70">
              Console →
            </Link>
          </div>
        </div>
      </nav>

      {/* ── Hero ───────────────────────────────────────── */}
      <section className="mx-auto max-w-[1400px] px-6 lg:px-10">
        <div className="grid grid-cols-1 gap-10 py-14 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] lg:gap-14 lg:py-20">
          {/* Left: the claim */}
          <div className="rise flex flex-col justify-center">
            <div className="mb-7 flex items-center gap-3">
              <span className="label">Runs in this tab</span>
              <div className="h-px w-10 bg-[var(--rule-strong)]" />
              <span className="label">No upload</span>
            </div>

            <h1 className="display text-[clamp(2.5rem,5.4vw,4.5rem)] text-[var(--ink)]">
              Hold a hand
              <br />
              to your camera.
              <br />
              <span className="text-[var(--ink-3)]">This page reads it.</span>
            </h1>

            <p className="mt-7 max-w-lg text-[15px] leading-relaxed text-[var(--ink-2)]">
              Not a video. Not a demo reel. The model loads into your browser and
              classifies {MEASURED.gestures} hand gestures from{" "}
              {MEASURED.landmarks} tracked joints — {MEASURED.modelSizeMb} MB of
              weights, running on your machine. No frame is uploaded, because
              there is nowhere to upload it to.
            </p>

            <div className="mt-9 flex flex-wrap items-center gap-3">
              <Link href="/dashboard" className="btn btn-solid">
                Open console
              </Link>
              <a
                href="#vocabulary"
                className="btn"
              >
                See all {MEASURED.gestures} gestures
              </a>
            </div>

            {/* Measured facts, not adjectives */}
            <div className="mt-12 grid grid-cols-2 gap-x-8 gap-y-7 border-t border-[var(--rule)] pt-8 sm:grid-cols-4">
              <Stat label="Test accuracy" value={MEASURED.testAccuracy} unit="%" size="sm" />
              <Stat label="Model size" value={MEASURED.modelSizeMb} unit="MB" size="sm" />
              <Stat label="Classifier" value={MEASURED.classifierMs} unit="ms" size="sm" />
              <Stat label="Trained on" value={MEASURED.trainingSamples} size="sm" />
            </div>
          </div>

          {/* Right: the instrument */}
          <div className="rise" style={{ animationDelay: "90ms" }}>
            <div className="panel overflow-hidden">
              <div className="flex items-center justify-between border-b border-[var(--rule)] px-4 py-2.5">
                <StatusFlag live={live} label={live ? "Acquiring" : "Camera off"} />
                <span className="label">
                  {live ? `${telemetry?.fps ?? 0} fps` : "—"}
                </span>
              </div>

              <div className="brackets">
                <LiveViewport
                  onTelemetry={handleTelemetry}
                  onStatusChange={handleStatus}
                  className="aspect-[16/11] w-full"
                />
              </div>

              {/* Live readout strip */}
              <div className="grid grid-cols-3 divide-x divide-[var(--rule)] border-t border-[var(--rule)]">
                <div className="px-4 py-3">
                  <span className="label">Gesture</span>
                  <p
                    className="readout mt-1 truncate text-sm"
                    style={{ color: detected ? "var(--signal)" : "var(--ink-3)" }}
                  >
                    {detected ? result?.gestureName : "no hand"}
                  </p>
                </div>
                <div className="px-4 py-3">
                  <span className="label">Confidence</span>
                  <p
                    className="readout mt-1 text-sm"
                    style={{ color: detected ? "var(--ink)" : "var(--ink-3)" }}
                  >
                    {detected ? (result?.confidence ?? 0).toFixed(3) : "—"}
                  </p>
                </div>
                <div className="px-4 py-3">
                  <span className="label">Frame time</span>
                  <p className="readout mt-1 text-sm text-[var(--ink)]">
                    {avgLatency ? `${avgLatency} ms` : "—"}
                  </p>
                </div>
              </div>

              <div className="border-t border-[var(--rule)] px-4 py-3">
                <div className="mb-2 flex items-center justify-between">
                  <span className="label">Frame time, last 60</span>
                  <span className="label">ceiling 60 ms</span>
                </div>
                <Sparkline values={latency} max={60} height={30} />
              </div>
            </div>

            <p className="mt-3 text-xs leading-relaxed text-[var(--ink-3)]">
              The feed is desaturated deliberately — the skeleton is what the
              model sees, so the skeleton is what gets drawn. The dashed box is
              the hand extent the classifier reasons over.
            </p>
          </div>
        </div>
      </section>

      {/* ── Vocabulary ─────────────────────────────────── */}
      <section
        id="vocabulary"
        className="border-t border-[var(--rule)] bg-[var(--field-1)]"
      >
        <div className="mx-auto max-w-[1400px] px-6 py-16 lg:px-10 lg:py-24">
          <div className="mb-12 flex flex-col justify-between gap-6 md:flex-row md:items-end">
            <div className="max-w-2xl">
              <span className="label">Plate I — Recognised vocabulary</span>
              <h2 className="display mt-4 text-[clamp(1.9rem,3.6vw,2.9rem)] text-[var(--ink)]">
                Every gesture, drawn from
                <br />
                the hands that taught it.
              </h2>
              <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-[var(--ink-2)]">
                These are not icons. Each plate is a real captured hand — the
                medoid of its class, the single closest observation to that
                gesture&apos;s centre across {MEASURED.trainingSamples} labelled
                samples. An average of hands would be an anatomical impossibility,
                so nothing here is averaged.
              </p>
            </div>
            <div className="shrink-0 md:w-64">
              <FieldRow name="Classes" value={MEASURED.gestures} />
              <FieldRow name="Points per plate" value={MEASURED.landmarks} />
              <FieldRow name="Held-out accuracy" value={`${MEASURED.testAccuracy}%`} />
            </div>
          </div>

          {atlas ? (
            <div className="grid grid-cols-2 gap-px border border-[var(--rule)] bg-[var(--rule)] sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
              {atlas.classes.map((cls) => (
                <figure
                  key={cls.label}
                  className="group flex flex-col bg-[var(--field-1)] p-5 transition-colors hover:bg-[var(--field-2)]"
                >
                  <HandPlate
                    landmarks={cls.landmarks}
                    connections={atlas.connections}
                    className="h-32 w-full transition-transform duration-300 group-hover:scale-[1.05]"
                  />
                  <figcaption className="mt-4 flex items-baseline justify-between gap-2">
                    <span className="mono text-[11px] text-[var(--ink)]">
                      {cls.label}
                    </span>
                    <span className="readout text-[10px] text-[var(--ink-4)]">
                      {cls.sampleCount.toLocaleString()}
                    </span>
                  </figcaption>
                </figure>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-px border border-[var(--rule)] bg-[var(--rule)] sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
              {Array.from({ length: 18 }).map((_, i) => (
                <div key={i} className="h-[172px] animate-pulse bg-[var(--field-1)]" />
              ))}
            </div>
          )}

          <p className="mt-6 text-xs text-[var(--ink-3)]">
            Plus a second model covering {MEASURED.aslLetters} ASL fingerspelling
            letters at {MEASURED.aslAccuracy}% — selectable in the console.
          </p>
        </div>
      </section>

      {/* ── Pipeline ───────────────────────────────────── */}
      <section id="pipeline" className="border-t border-[var(--rule)]">
        <div className="mx-auto max-w-[1400px] px-6 py-16 lg:px-10 lg:py-24">
          <span className="label">Signal chain</span>
          <h2 className="display mt-4 max-w-2xl text-[clamp(1.9rem,3.6vw,2.9rem)] text-[var(--ink)]">
            It classifies skeletons,
            <br />
            not pixels.
          </h2>
          <p className="mt-5 max-w-2xl text-[15px] leading-relaxed text-[var(--ink-2)]">
            The image is discarded the moment {MEASURED.landmarks} joint
            coordinates come out of it. Everything downstream sees only numbers,
            which is why the model is {MEASURED.modelSizeMb} MB instead of
            hundreds, and why lighting and skin tone stop mattering.
          </p>

          <ol className="mt-12 grid grid-cols-1 gap-px border border-[var(--rule)] bg-[var(--rule)] sm:grid-cols-2 lg:grid-cols-6">
            {PIPELINE.map((s) => (
              <li key={s.step} className="flex flex-col gap-2 bg-[var(--field)] p-5">
                <span className="readout text-[10px] text-[var(--signal)]">{s.step}</span>
                <span className="display text-base text-[var(--ink)]">{s.name}</span>
                <span className="mono text-[11px] leading-relaxed text-[var(--ink-3)]">
                  {s.detail}
                </span>
              </li>
            ))}
          </ol>

          <div className="mt-14 grid grid-cols-1 gap-10 lg:grid-cols-3">
            <div>
              <h3 className="display text-lg text-[var(--ink)]">Privacy by construction</h3>
              <p className="mt-3 text-sm leading-relaxed text-[var(--ink-2)]">
                There is no inference server. The ONNX weights load over HTTP once,
                then every frame is processed by your CPU or GPU. Turning off your
                network mid-session changes nothing.
              </p>
            </div>
            <div>
              <h3 className="display text-lg text-[var(--ink)]">It says when it doesn&apos;t know</h3>
              <p className="mt-3 text-sm leading-relaxed text-[var(--ink-2)]">
                Outputs are temperature-calibrated, so a confidence of 0.7 means
                roughly seven-in-ten. Anything under the rejection threshold is
                reported as no match rather than guessed.
              </p>
            </div>
            <div>
              <h3 className="display text-lg text-[var(--ink)]">Teachable past the vocabulary</h3>
              <p className="mt-3 text-sm leading-relaxed text-[var(--ink-2)]">
                A fixed model is always finite. Demonstrate a gesture it has never
                seen a handful of times and the console stores it locally as a
                few-shot class — no retraining, no upload.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Spec ───────────────────────────────────────── */}
      <section className="border-t border-[var(--rule)] bg-[var(--field-1)]">
        <div className="mx-auto max-w-[1400px] px-6 py-16 lg:px-10 lg:py-20">
          <div className="grid grid-cols-1 gap-10 lg:grid-cols-[1fr_1.4fr]">
            <div>
              <span className="label">Specification</span>
              <h2 className="display mt-4 text-[clamp(1.7rem,3vw,2.4rem)] text-[var(--ink)]">
                Measured, on held-out data.
              </h2>
              <p className="mt-4 max-w-sm text-sm leading-relaxed text-[var(--ink-2)]">
                Test figures come from a split the model never trained or
                validated on. The frame-time figure above is whatever your machine
                is doing right now.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-x-12 sm:grid-cols-2">
              <div>
                <FieldRow name="Architecture" value="Transformer encoder" />
                <FieldRow name="Parameters" value={MEASURED.params} />
                <FieldRow name="Input" value={`${MEASURED.windowFrames} × ${MEASURED.features}`} />
                <FieldRow name="Export" value="ONNX · int8-ready" />
              </div>
              <div>
                <FieldRow name="Gestures" value={`${MEASURED.gestures} classes`} />
                <FieldRow name="ASL letters" value={`${MEASURED.aslLetters} classes`} />
                <FieldRow name="Runtime" value="onnxruntime-web" />
                <FieldRow name="Backend" value="WebGPU, WASM fallback" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ─────────────────────────────────────── */}
      <footer className="border-t border-[var(--rule)]">
        <div className="mx-auto flex max-w-[1400px] flex-col gap-4 px-6 py-10 text-[var(--ink-3)] sm:flex-row sm:items-center sm:justify-between lg:px-10">
          <span className="label">Dextera · on-device gesture recognition</span>
          <span className="label">
            Gesture data: HaGRID · ASL alphabet · see DATASET_LICENSES
          </span>
        </div>
      </footer>
    </div>
  );
}
