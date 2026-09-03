"use client";

/**
 * Geometry for drawing landmarks over a video rendered with `object-fit: cover`.
 *
 * Hand landmarks are normalised over the *source* video frame. The video is
 * displayed scaled to fill its box and cropped on whichever axis overflows, so
 * mapping normalised coordinates straight onto the overlay canvas is only
 * correct when the stream and the box happen to share an aspect ratio. A 4:3
 * webcam in a 16:9 panel puts the skeleton nowhere near the hand — misplaced
 * *and* mis-scaled, which reads as broken tracking rather than a drawing bug.
 *
 * Separated out and unit-tested because it is easy to get subtly wrong and
 * nearly impossible to check by eye: an overlay that is 10% off still looks
 * plausible on a moving hand.
 */

export interface CoverGeometry {
    /** Size the video actually occupies, before cropping. */
    drawnWidth: number;
    drawnHeight: number;
    /** Offset of the drawn video relative to the box. Negative when cropped. */
    offsetX: number;
    offsetY: number;
}

export function coverGeometry(
    videoWidth: number,
    videoHeight: number,
    boxWidth: number,
    boxHeight: number
): CoverGeometry {
    // Degenerate inputs happen for real: `videoWidth` is 0 until the stream's
    // metadata arrives, and dividing by it would poison every coordinate with
    // NaN for the first frames.
    if (videoWidth <= 0 || videoHeight <= 0 || boxWidth <= 0 || boxHeight <= 0) {
        return { drawnWidth: boxWidth, drawnHeight: boxHeight, offsetX: 0, offsetY: 0 };
    }

    const videoAspect = videoWidth / videoHeight;
    const boxAspect = boxWidth / boxHeight;

    let drawnWidth: number;
    let drawnHeight: number;

    if (videoAspect > boxAspect) {
        // Wider than the box: fills the height, cropped left and right.
        drawnHeight = boxHeight;
        drawnWidth = drawnHeight * videoAspect;
    } else {
        // Taller than the box: fills the width, cropped top and bottom.
        drawnWidth = boxWidth;
        drawnHeight = drawnWidth / videoAspect;
    }

    return {
        drawnWidth,
        drawnHeight,
        offsetX: (boxWidth - drawnWidth) / 2,
        offsetY: (boxHeight - drawnHeight) / 2,
    };
}

/**
 * Project a normalised landmark onto the overlay canvas.
 *
 * `mirrored` accounts for the video being flipped in CSS so the user sees
 * themselves as in a mirror. The crop is symmetric, so flipping within the
 * drawn area and flipping about the box centre are equivalent.
 */
export function projectLandmark(
    point: { x: number; y: number },
    geometry: CoverGeometry,
    mirrored = true
): { x: number; y: number } {
    const nx = mirrored ? 1 - point.x : point.x;
    return {
        x: geometry.offsetX + nx * geometry.drawnWidth,
        y: geometry.offsetY + point.y * geometry.drawnHeight,
    };
}
