"use client";

/**
 * The on-screen cursor for hands-free pointing.
 *
 * Rendered as a fixed overlay above everything and explicitly
 * `pointer-events: none`, so it can sit over the interface it is operating
 * without intercepting the clicks it dispatches.
 *
 * The dwell ring is the whole interaction: without a visible, continuously
 * filling indicator the user has no way to know whether a click is coming, how
 * long is left, or that holding still is what triggers it.
 */

interface PointerOverlayProps {
    x: number;
    y: number;
    dwellProgress: number;
    active: boolean;
    /** Set briefly right after a click, to confirm it landed. */
    flash: boolean;
}

const RADIUS = 22;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export function PointerOverlay({ x, y, dwellProgress, active, flash }: PointerOverlayProps) {
    if (!active) return null;

    return (
        <div
            className="pointer-events-none fixed z-[200]"
            style={{
                left: x,
                top: y,
                transform: "translate(-50%, -50%)",
                // No CSS transition: the position is already smoothed in the
                // engine, and transitioning it again adds lag the user feels as
                // the cursor lagging behind their hand.
            }}
            aria-hidden="true"
        >
            <svg width={RADIUS * 2 + 8} height={RADIUS * 2 + 8} style={{ display: "block" }}>
                <g transform={`translate(${RADIUS + 4}, ${RADIUS + 4})`}>
                    {/* Track */}
                    <circle
                        r={RADIUS}
                        fill="none"
                        stroke="rgba(237,235,230,0.22)"
                        strokeWidth={2}
                    />
                    {/* Dwell progress, drawn from 12 o'clock */}
                    <circle
                        r={RADIUS}
                        fill="none"
                        stroke={flash ? "#edebe6" : "#ffb627"}
                        strokeWidth={3}
                        strokeLinecap="round"
                        strokeDasharray={CIRCUMFERENCE}
                        strokeDashoffset={CIRCUMFERENCE * (1 - dwellProgress)}
                        transform="rotate(-90)"
                    />
                    {/* Centre dot marks the exact hit point, which the ring does not */}
                    <circle r={flash ? 6 : 3} fill={flash ? "#edebe6" : "#ffb627"} />
                </g>
            </svg>
        </div>
    );
}
