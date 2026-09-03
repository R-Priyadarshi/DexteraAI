import { describe, expect, it } from "vitest";
import {
  PREFERRED_SIGNATURE,
  SIGNATURE_LENGTH,
  signatureFor,
} from "../../components/BiometricGuard";

/** The vocabulary actually shipped in models/hagrid/labels.json. */
const HAGRID = [
  "call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm",
  "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three",
  "three2", "two_up", "two_up_inverted",
];

const ASL = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));

describe("signatureFor", () => {
  it("returns the configured number of steps", () => {
    expect(signatureFor(HAGRID)).toHaveLength(SIGNATURE_LENGTH);
    expect(signatureFor(ASL)).toHaveLength(SIGNATURE_LENGTH);
  });

  it("only ever names gestures the loaded model can produce", () => {
    // The failure this replaced: the guard displayed one set of gestures while
    // requiring another, so following its instructions never unlocked it.
    for (const vocabulary of [HAGRID, ASL]) {
      for (const step of signatureFor(vocabulary)) {
        expect(vocabulary).toContain(step);
      }
    }
  });

  it("prefers the curated gestures when the bundle has them", () => {
    expect(signatureFor(HAGRID)).toEqual(["peace", "fist", "like"]);
  });

  it("falls back to the vocabulary for a bundle sharing no preferred labels", () => {
    // The ASL alphabet has none of the general-gesture names.
    expect(signatureFor(ASL)).toEqual(["A", "B", "C"]);
  });

  it("uses no repeated step", () => {
    // A repeat would let one held pose satisfy two steps in a row.
    for (const vocabulary of [HAGRID, ASL]) {
      const steps = signatureFor(vocabulary);
      expect(new Set(steps).size).toBe(steps.length);
    }
  });

  it("falls back rather than returning a short signature", () => {
    // Only two preferred labels present: taking just those would silently
    // shorten the signature and weaken the guard.
    const sparse = ["peace", "fist", "zzz", "yyy"];
    expect(signatureFor(sparse)).toHaveLength(SIGNATURE_LENGTH);
  });

  it("does not throw on an empty or tiny vocabulary", () => {
    // Reachable before a bundle finishes loading.
    expect(signatureFor([])).toEqual([]);
    expect(signatureFor(["only"])).toEqual(["only"]);
  });

  it("keeps the preferred list free of duplicates", () => {
    expect(new Set(PREFERRED_SIGNATURE).size).toBe(PREFERRED_SIGNATURE.length);
  });
});
