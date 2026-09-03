import { describe, expect, it } from "vitest";

import { AIR_DIGIT_NAMES, AIR_DIGITS, AIR_LETTER_NAMES, AIR_LETTERS, AIR_SETS, type AirSetName } from "../air-letters";
import { DollarRecognizer, ORIENTED_OPTIONS, SHAPE_OPTIONS, type Point } from "../dollar-recognizer";
import { handDrawn, type StrokeOptions } from "./stroke-fixtures";

/** One recogniser per set — never both at once, which is the point. */
function recognizerFor(set: AirSetName): DollarRecognizer {
    const recognizer = new DollarRecognizer(ORIENTED_OPTIONS);
    for (const [name, points] of Object.entries(AIR_SETS[set])) {
        recognizer.learn(name, points);
    }
    return recognizer;
}

/** How far the right template beats a rival, for one written stroke. */
function margin(
    options: typeof ORIENTED_OPTIONS,
    correct: Point[],
    rival: Point[],
    written: Point[],
): number {
    const a = new DollarRecognizer(options);
    a.learn("correct", correct);
    const b = new DollarRecognizer(options);
    b.learn("rival", rival);
    return a.recognize(written)!.score - b.recognize(written)!.score;
}

describe("the air alphabet", () => {
    it("covers A-Z and 0-9 with no gaps", () => {
        expect(AIR_LETTER_NAMES).toHaveLength(26);
        expect(AIR_DIGIT_NAMES).toHaveLength(10);
        for (const c of "ABCDEFGHIJKLMNOPQRSTUVWXYZ") {
            expect(AIR_LETTERS[c], `missing letter ${c}`).toBeDefined();
        }
        for (const c of "0123456789") {
            expect(AIR_DIGITS[c], `missing digit ${c}`).toBeDefined();
        }
    });

    it("gives every form at least two points", () => {
        for (const set of Object.values(AIR_SETS)) {
            for (const [name, points] of Object.entries(set)) {
                expect(points.length, `${name} is degenerate`).toBeGreaterThan(1);
            }
        }
    });

    it.each(AIR_LETTER_NAMES)("reads a hand-written %s", (name) => {
        const written = handDrawn(AIR_LETTERS[name], {
            rotate: 0.08,
            scale: 2.5,
            offset: { x: 40, y: -15 },
            seed: 5,
        });
        expect(recognizerFor("letters").recognize(written)?.name).toBe(name);
    });

    it.each(AIR_DIGIT_NAMES)("reads a hand-written digit %s", (name) => {
        const written = handDrawn(AIR_DIGITS[name], {
            rotate: 0.08,
            scale: 2.5,
            offset: { x: 40, y: -15 },
            seed: 5,
        });
        expect(recognizerFor("digits").recognize(written)?.name).toBe(name);
    });

    it.each(["letters", "digits"] as AirSetName[])(
        "reads %s written smaller, further away and tilted the other way",
        (set) => {
            const recognizer = recognizerFor(set);
            for (const [name, form] of Object.entries(AIR_SETS[set])) {
                const written = handDrawn(form, {
                    rotate: -0.1,
                    scale: 0.4,
                    offset: { x: -120, y: 300 },
                    jitter: 0.04,
                    seed: 23,
                });
                expect(recognizer.recognize(written)?.name, `${name} misread`).toBe(name);
            }
        },
    );
});

describe("why letters and digits are separate sets", () => {
    it("recognises every symbol when the sets are kept apart", () => {
        for (const set of ["letters", "digits"] as AirSetName[]) {
            const recognizer = recognizerFor(set);
            for (const [name, form] of Object.entries(AIR_SETS[set])) {
                expect(recognizer.recognize(handDrawn(form, { seed: 8 }))?.name).toBe(name);
            }
        }
    });

    it("keeps more symbols readable than merging them, once the hand is shaky", () => {
        // Measured across 8 noise seeds rather than assumed, and the numbers
        // corrected an earlier claim that the sets collide outright:
        //
        //   tremor 0.02   merged 100%    separate 100%
        //   tremor 0.05   merged 100%    separate 100%
        //   tremor 0.09   merged  99.0%  separate 100%
        //   tremor 0.14   merged  96.9%  separate  99.0%
        //
        // So the split buys nothing for a steady hand and roughly a third of
        // the errors back for an unsteady one. It stays because mode is
        // something the writer always knows and the recogniser would otherwise
        // have to guess, and because guessing gets worse exactly when the
        // person is already struggling.
        const JITTER = 0.14;

        const merged = new DollarRecognizer(ORIENTED_OPTIONS);
        for (const [name, form] of Object.entries({ ...AIR_LETTERS, ...AIR_DIGITS })) {
            merged.learn(name, form);
        }

        let mergedHits = 0;
        let separateHits = 0;
        let total = 0;
        for (let seed = 1; seed <= 8; seed++) {
            for (const set of ["letters", "digits"] as AirSetName[]) {
                const scoped = recognizerFor(set);
                for (const [name, form] of Object.entries(AIR_SETS[set])) {
                    const written = handDrawn(form, { jitter: JITTER, seed, rotate: 0.06 });
                    total++;
                    if (merged.recognize(written)?.name === name) mergedHits++;
                    if (scoped.recognize(written)?.name === name) separateHits++;
                }
            }
        }

        expect(total).toBe(288);
        expect(separateHits).toBeGreaterThan(mergedHits);
    });
});

describe("orientation", () => {
    // Each pair is the same stroke turned around, which is why the alphabet is
    // matched with rotation invariance off.
    const TURNED_PAIRS: [string, string][] = [
        ["M", "W"],
        ["N", "Z"],
        ["C", "U"],
    ];

    it.each(TURNED_PAIRS)("tells %s from %s", (a, b) => {
        const recognizer = recognizerFor("letters");
        expect(recognizer.recognize(handDrawn(AIR_LETTERS[a], { seed: 31 }))?.name).toBe(a);
        expect(recognizer.recognize(handDrawn(AIR_LETTERS[b], { seed: 31 }))?.name).toBe(b);
    });

    it("tells 6 from 9", () => {
        const recognizer = recognizerFor("digits");
        expect(recognizer.recognize(handDrawn(AIR_DIGITS["6"], { seed: 31 }))?.name).toBe("6");
        expect(recognizer.recognize(handDrawn(AIR_DIGITS["9"], { seed: 31 }))?.name).toBe("9");
    });

    it.each(TURNED_PAIRS)("keeps %s and %s further apart than rotation invariance would", (a, b) => {
        // The honest form of this claim. Normalising rotation away does not
        // always flip these outright — with only two templates in play the ±45°
        // search may not reach — but it does erode the margin that keeps them
        // apart, and the margin is what survives real handwriting.
        const written = handDrawn(AIR_LETTERS[a], { seed: 12 });
        const oriented = margin(ORIENTED_OPTIONS, AIR_LETTERS[a], AIR_LETTERS[b], written);
        const invariant = margin(SHAPE_OPTIONS, AIR_LETTERS[a], AIR_LETTERS[b], written);

        expect(oriented).toBeGreaterThan(invariant);
        expect(oriented).toBeGreaterThan(0);
    });
});
