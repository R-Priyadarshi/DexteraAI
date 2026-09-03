"use client";

import { useEffect, useState } from "react";
import { asset } from "@/lib/base-path";

export interface Point3 {
  0: number;
  1: number;
  2: number;
  length: number;
}

export type AtlasLandmarks = number[][];

export interface GestureClass {
  label: string;
  index: number;
  handedness: string;
  sampleCount: number;
  landmarks: AtlasLandmarks;
}

export interface GestureAtlas {
  source: string;
  note: string;
  connections: number[][];
  classes: GestureClass[];
}

/** MediaPipe's 21-point topology, used when the atlas hasn't loaded yet. */
export const FALLBACK_CONNECTIONS: number[][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

/** Landmark indices that terminate a digit — drawn slightly larger. */
const TIPS = new Set([4, 8, 12, 16, 20]);

interface HandPlateProps {
  landmarks: AtlasLandmarks;
  connections?: number[][];
  /** Draw in the live signal colour rather than resting ink. */
  live?: boolean;
  /** Render joint index numbers, for the anatomical/atlas views. */
  annotate?: boolean;
  strokeWidth?: number;
  className?: string;
}

/**
 * Draws a hand from 21 landmarks as a wireframe plate.
 *
 * Coordinates are expected wrist-centred and unit-scaled (see
 * scripts/export_gesture_atlas.py). y follows image convention — larger is
 * lower — which SVG shares, so fingers above the wrist render above it.
 */
export function HandPlate({
  landmarks,
  connections = FALLBACK_CONNECTIONS,
  live = false,
  annotate = false,
  strokeWidth = 0.022,
  className = "",
}: HandPlateProps) {
  if (!landmarks?.length) return null;

  const stroke = live ? "var(--signal)" : "var(--ink-2)";
  const jointFill = live ? "var(--signal)" : "var(--ink-2)";

  // Fit the pose to its frame. Poses arrive scaled by wrist distance, which
  // leaves each hand occupying a different slice of the box — a splayed palm
  // and a closed fist would render at wildly different sizes. Framing on the
  // actual bounding box makes the set read as a consistent plate series.
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of landmarks) {
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[1] > maxY) maxY = p[1];
  }
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);
  const span = Math.max(spanX, spanY);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;

  // Map into a -1..1 box, keeping aspect, with room for the joint dots.
  const k = 1.82 / span;
  const fit = (p: number[]) => [(p[0] - cx) * k, (p[1] - cy) * k];
  const pts = landmarks.map(fit);

  // Stroke weight is given in the original coordinate space, so rescale it
  // with the pose or thin hands would draw hairlines and fists would blob.
  const sw = strokeWidth * k;

  return (
    <svg
      viewBox="-1 -1 2 2"
      className={className}
      fill="none"
      aria-hidden="true"
    >
      {/* Bones */}
      {connections.map(([a, b], i) => {
        const p = pts[a];
        const q = pts[b];
        if (!p || !q) return null;
        return (
          <line
            key={`b${i}`}
            x1={p[0]}
            y1={p[1]}
            x2={q[0]}
            y2={q[1]}
            stroke={stroke}
            strokeWidth={sw}
            strokeLinecap="round"
          />
        );
      })}

      {/* Joints */}
      {pts.map((p, i) => (
        <circle
          key={`j${i}`}
          cx={p[0]}
          cy={p[1]}
          r={i === 0 ? sw * 2.1 : TIPS.has(i) ? sw * 1.8 : sw * 1.15}
          fill={i === 0 || TIPS.has(i) ? jointFill : "var(--field)"}
          stroke={jointFill}
          strokeWidth={sw * 0.75}
        />
      ))}

      {annotate &&
        pts.map((p, i) => (
          <text
            key={`t${i}`}
            x={p[0] + 0.07}
            y={p[1] - 0.05}
            fontSize="0.1"
            fill="var(--ink-4)"
            fontFamily="var(--font-mono)"
          >
            {i}
          </text>
        ))}
    </svg>
  );
}

/** Loads the gesture atlas exported from the training data. */
export function useGestureAtlas() {
  const [atlas, setAtlas] = useState<GestureAtlas | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(asset("/data/gesture-atlas.json"))
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.json();
      })
      .then((data: GestureAtlas) => {
        if (!cancelled) setAtlas(data);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { atlas, error };
}
