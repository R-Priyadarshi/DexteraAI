"use client";

import { type Landmark } from "./gesture-engine";

/**
 * Landmark normalization, mirroring `core/landmarks/normalizer.py` exactly.
 *
 * This has to match the Python implementation transform for transform, because
 * the model is trained on Python's output and served Python's weights. It was
 * missing entirely: the browser extracted features straight from MediaPipe's
 * raw output, so 63 of the 86 feature dimensions — the flattened coordinate
 * block — arrived as absolute image positions in [0, 1] where the model had
 * learned wrist-centred, unit-scaled, rotation-aligned coordinates.
 *
 * The result was a model receiving out-of-distribution input for 73% of its
 * feature vector, which presented as confident-looking gestures scoring barely
 * above the rejection threshold and landing on the wrong label.
 *
 * `NormalizationMode.FULL` is what the datasets use, so that is what this
 * implements: centre on the wrist, scale to unit extent, then rotate the
 * wrist→middle-knuckle axis onto +Y.
 */

const WRIST = 0;
const MIDDLE_MCP = 9;

/**
 * Normalize one hand's landmarks.
 *
 * Returns a new array; the input is left untouched, because callers also draw
 * the raw landmarks over the video and mutating them in place would drag the
 * skeleton to the origin.
 */
export function normalizeLandmarks(landmarks: Landmark[]): Landmark[] {
    if (landmarks.length === 0) return [];

    // 1. Centre on the wrist, so absolute position in frame stops mattering.
    const origin = landmarks[WRIST];
    const centred = landmarks.map((p) => ({
        x: p.x - origin.x,
        y: p.y - origin.y,
        z: p.z - origin.z,
    }));

    // 2. Scale to unit extent, so distance from the camera stops mattering.
    let maxDist = 0;
    for (const p of centred) {
        const d = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (d > maxDist) maxDist = d;
    }
    const scaled =
        maxDist < 1e-6
            ? centred
            : centred.map((p) => ({
                  x: p.x / maxDist,
                  y: p.y / maxDist,
                  z: p.z / maxDist,
              }));

    // 3. Rotate the wrist→middle-knuckle axis onto +Y, so hand tilt stops
    //    mattering. Only the XY plane is rotated; z is depth and carries no
    //    in-plane rotation.
    const dx = scaled[MIDDLE_MCP].x - scaled[WRIST].x;
    const dy = scaled[MIDDLE_MCP].y - scaled[WRIST].y;
    const norm = Math.sqrt(dx * dx + dy * dy);
    if (norm < 1e-6) return scaled;

    // Matching the Python rotation matrix [[cos, sin, 0], [-sin, cos, 0], ...]
    // with cos = dy/norm and sin = dx/norm.
    const cos = dy / norm;
    const sin = dx / norm;

    return scaled.map((p) => ({
        x: cos * p.x + sin * p.y,
        y: -sin * p.x + cos * p.y,
        z: p.z,
    }));
}
