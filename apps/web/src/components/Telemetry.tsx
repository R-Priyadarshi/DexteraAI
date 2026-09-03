"use client";

/**
 * Readout primitives.
 *
 * Everything here renders a measured quantity. Nothing renders a decoration —
 * if a value isn't being measured, the component shows a dash rather than a
 * plausible-looking number.
 */

interface StatProps {
  label: string;
  value: string | number | null | undefined;
  unit?: string;
  live?: boolean;
  size?: "sm" | "md" | "lg";
}

export function Stat({ label, value, unit, live = false, size = "md" }: StatProps) {
  const shown = value === null || value === undefined || value === "" ? "—" : value;
  const sizes = {
    sm: "text-lg",
    md: "text-2xl",
    lg: "text-4xl",
  } as const;

  return (
    <div className="flex flex-col gap-1.5">
      <span className="label">{label}</span>
      <span
        className={`readout ${sizes[size]} leading-none`}
        style={{ color: live ? "var(--signal)" : "var(--ink)" }}
      >
        {shown}
        {unit && (
          <span className="ml-1 text-[0.5em] text-[var(--ink-3)]">{unit}</span>
        )}
      </span>
    </div>
  );
}

interface SparklineProps {
  values: number[];
  /** Fixed ceiling in the same unit as `values`; keeps the scale readable. */
  max?: number;
  height?: number;
  className?: string;
}

/** Latency trace. Bars are real per-frame measurements, newest on the right. */
export function Sparkline({ values, max, height = 34, className = "" }: SparklineProps) {
  const slots = 60;
  const padded = [...values].slice(-slots);

  // `max` is the nominal budget, not a hard ceiling. A slow machine would peg
  // every bar at full height and the trace would stop saying anything, so the
  // scale grows past nominal while over-budget samples stay flagged red.
  const observed = padded.length > 0 ? Math.max(...padded) : 0;
  const nominal = max ?? 1;
  const ceiling = Math.max(nominal, observed);

  return (
    <div
      className={`flex items-end gap-px ${className}`}
      style={{ height }}
      aria-hidden="true"
    >
      {Array.from({ length: slots }).map((_, i) => {
        const v = padded[i - (slots - padded.length)];
        // Empty slots keep a 2px footprint so the axis reads as an idle trace
        // rather than a blank panel.
        const h = v === undefined ? 2 : Math.max(2, Math.min(1, v / ceiling) * height);
        return (
          <div
            key={i}
            className="flex-1"
            style={{
              height: h,
              background:
                v === undefined
                  ? "var(--rule-2)"
                  : v > nominal
                    ? "var(--alert)"
                    : "var(--signal-2)",
            }}
          />
        );
      })}
    </div>
  );
}

interface MeterProps {
  label: string;
  value: number;
  /** Below this, the reading is treated as a rejection rather than a detection. */
  threshold?: number;
  mono?: boolean;
}

export function ConfidenceMeter({ label, value, threshold = 0.3 }: MeterProps) {
  const pct = Math.round(value * 100);
  const passing = value >= threshold;

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <span
          className="mono text-xs"
          style={{ color: passing ? "var(--ink)" : "var(--ink-3)" }}
        >
          {label}
        </span>
        <span
          className="readout text-xs"
          style={{ color: passing ? "var(--signal)" : "var(--ink-3)" }}
        >
          {value.toFixed(3)}
        </span>
      </div>
      <div className="meter">
        <div
          className={`meter-fill ${passing ? "" : "meter-fill-dim"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

interface FieldRowProps {
  name: string;
  value: React.ReactNode;
}

/** A single key/value line in a spec block. */
export function FieldRow({ name, value }: FieldRowProps) {
  return (
    <div className="flex items-baseline justify-between gap-6 border-b border-[var(--rule-2)] py-2.5">
      <span className="label">{name}</span>
      <span className="readout text-xs text-[var(--ink)]">{value}</span>
    </div>
  );
}

/** Live/idle status pill used in headers. */
export function StatusFlag({ live, label }: { live: boolean; label?: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className={live ? "live-dot" : "idle-dot"} />
      <span className={`label ${live ? "label-signal" : ""}`}>
        {label ?? (live ? "Acquiring" : "Idle")}
      </span>
    </div>
  );
}
