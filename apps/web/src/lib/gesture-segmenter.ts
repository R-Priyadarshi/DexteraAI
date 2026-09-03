/**
 * Temporal segmentation for a continuous classification stream.
 *
 * The classifier scores every frame independently, so holding a pose produces
 * the same label 30 times a second. Anything bound to that label — a slide
 * advance, a media key, a click — would fire 30 times too. What consumers
 * actually need is discrete events: this gesture *started*, is *being held*,
 * has *ended*.
 *
 * The state machine here converts one into the other, with three guards that
 * each address a distinct failure mode:
 *
 *   - **Entry debounce.** A label must win `onsetFrames` consecutive frames
 *     before it is accepted, so a single misclassified frame mid-gesture
 *     cannot emit a spurious onset.
 *   - **Exit hysteresis.** Once active, a gesture survives `offsetFrames` of
 *     disagreement before it ends. Landmark tracking drops frames when a hand
 *     rotates or is briefly occluded; without this, one dropped frame ends the
 *     gesture and the next one starts it again, producing a burst of events
 *     from a hand that never moved.
 *   - **Refractory period.** After an offset, no new onset is admitted for
 *     `refractoryMs`. This bounds the event rate regardless of what the
 *     classifier does, which matters because the actions downstream are not
 *     idempotent.
 *
 * Entry and exit deliberately use different thresholds. A symmetric threshold
 * sits exactly on the boundary the signal is dithering across, which is what
 * produces chatter in the first place.
 */

export type GesturePhase = "idle" | "onset" | "hold" | "offset";

export interface SegmentEvent {
  phase: GesturePhase;
  /** Label the segment refers to; null while idle. */
  gestureName: string | null;
  gestureId: number;
  /** Confidence of the frame that produced this event. */
  confidence: number;
  /** Milliseconds since this segment's onset. Zero for onset and idle. */
  heldMs: number;
  /** Monotonically increasing id, unique per segment. Zero while idle. */
  segmentId: number;
}

export interface SegmenterConfig {
  /** Consecutive agreeing frames required to open a segment. */
  onsetFrames: number;
  /** Consecutive disagreeing frames required to close one. */
  offsetFrames: number;
  /** Minimum confidence for a frame to count toward an onset. */
  enterConfidence: number;
  /** Confidence below which an active segment starts closing. */
  exitConfidence: number;
  /** Dead time after an offset before another onset may open. */
  refractoryMs: number;
  /** Labels that can never open a segment (background / null classes). */
  ignored: string[];
}

export const DEFAULT_SEGMENTER_CONFIG: SegmenterConfig = {
  onsetFrames: 4,
  offsetFrames: 6,
  enterConfidence: 0.7,
  // Deliberately below `enterConfidence` — see the hysteresis note above.
  exitConfidence: 0.45,
  refractoryMs: 350,
  ignored: ["none", "no hand", "unknown"],
};

interface FrameInput {
  gestureName: string;
  gestureId: number;
  confidence: number;
  /** True when the frame was rejected by the open-set threshold. */
  rejected?: boolean;
}

export class GestureSegmenter {
  private config: SegmenterConfig;

  private activeName: string | null = null;
  private activeId = -1;
  private segmentId = 0;
  private onsetAt = 0;

  /** Label currently accumulating evidence toward an onset. */
  private candidate: string | null = null;
  private candidateId = -1;
  private candidateFrames = 0;

  /** Frames of disagreement accumulated against the active segment. */
  private dissentFrames = 0;

  private refractoryUntil = 0;

  constructor(config: Partial<SegmenterConfig> = {}) {
    this.config = { ...DEFAULT_SEGMENTER_CONFIG, ...config };
  }

  configure(config: Partial<SegmenterConfig>): void {
    this.config = { ...this.config, ...config };
  }

  getConfig(): SegmenterConfig {
    return { ...this.config };
  }

  /** True while a segment is open. */
  isActive(): boolean {
    return this.activeName !== null;
  }

  /**
   * Feed one classified frame; get back the phase it produced.
   *
   * `now` is injectable so the state machine can be tested deterministically
   * rather than against a real clock.
   */
  push(frame: FrameInput, now: number = performance.now()): SegmentEvent {
    const usable =
      !frame.rejected &&
      frame.gestureId >= 0 &&
      !this.config.ignored.includes(frame.gestureName);

    // ── An segment is open: decide whether it survives this frame ──────────
    if (this.activeName !== null) {
      const agrees =
        usable &&
        frame.gestureName === this.activeName &&
        frame.confidence >= this.config.exitConfidence;

      if (agrees) {
        this.dissentFrames = 0;
        return this.event("hold", frame.confidence, now);
      }

      this.dissentFrames++;
      if (this.dissentFrames < this.config.offsetFrames) {
        // Still within hysteresis — treat the drop-out as noise, not an end.
        return this.event("hold", frame.confidence, now);
      }

      const closing = this.event("offset", frame.confidence, now);
      this.activeName = null;
      this.activeId = -1;
      this.dissentFrames = 0;
      this.candidate = null;
      this.candidateFrames = 0;
      this.refractoryUntil = now + this.config.refractoryMs;
      return closing;
    }

    // ── Idle: accumulate evidence toward opening one ───────────────────────
    if (now < this.refractoryUntil || !usable || frame.confidence < this.config.enterConfidence) {
      // A frame that fails any entry condition breaks the run. Requiring
      // *consecutive* agreement is what makes a lone bad frame harmless.
      this.candidate = null;
      this.candidateFrames = 0;
      return this.idle();
    }

    if (frame.gestureName === this.candidate) {
      this.candidateFrames++;
    } else {
      this.candidate = frame.gestureName;
      this.candidateId = frame.gestureId;
      this.candidateFrames = 1;
    }

    if (this.candidateFrames >= this.config.onsetFrames) {
      this.activeName = this.candidate;
      this.activeId = this.candidateId;
      this.segmentId++;
      this.onsetAt = now;
      this.dissentFrames = 0;
      this.candidate = null;
      this.candidateFrames = 0;
      return this.event("onset", frame.confidence, now);
    }

    return this.idle();
  }

  /**
   * Abandon any open segment without emitting an offset.
   *
   * Used when tracking is lost outright, where the honest statement is that we
   * no longer know what the hand is doing — not that the gesture ended.
   */
  reset(): void {
    this.activeName = null;
    this.activeId = -1;
    this.candidate = null;
    this.candidateFrames = 0;
    this.dissentFrames = 0;
    this.onsetAt = 0;
  }

  private event(phase: GesturePhase, confidence: number, now: number): SegmentEvent {
    return {
      phase,
      gestureName: this.activeName,
      gestureId: this.activeId,
      confidence,
      heldMs: phase === "onset" ? 0 : Math.max(0, now - this.onsetAt),
      segmentId: this.segmentId,
    };
  }

  private idle(): SegmentEvent {
    return {
      phase: "idle",
      gestureName: null,
      gestureId: -1,
      confidence: 0,
      heldMs: 0,
      segmentId: 0,
    };
  }
}
