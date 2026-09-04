import { describe, expect, it } from "vitest";

import type { GestureCandidate } from "../gesture-engine";
import { SpellingBuffer } from "../spelling-buffer";
import { WordSuggester } from "../word-suggester";

/** Ranked candidates, given as "letter:confidence" strings for brevity. */
function ranked(...spec: string[]): GestureCandidate[] {
    return spec.map((s) => {
        const [gestureName, confidence] = s.split(":");
        return { gestureName, gestureId: -1, confidence: Number(confidence) };
    });
}

function spell(buffer: SpellingBuffer, ...letters: GestureCandidate[][]): void {
    for (const candidates of letters) buffer.commit(candidates);
}

describe("SpellingBuffer", () => {
    it("starts empty", () => {
        const buffer = new SpellingBuffer();
        expect(buffer.isEmpty).toBe(true);
        expect(buffer.text).toBe("");
    });

    it("spells with the model's first choice", () => {
        const buffer = new SpellingBuffer();
        spell(buffer, ranked("C:0.9", "O:0.05"), ranked("A:0.8", "S:0.1"), ranked("T:0.95"));
        expect(buffer.word).toBe("CAT");
    });

    it("ignores a commit with no candidates", () => {
        // A frame where the model abstained must not push an empty slot, which
        // would silently shift every later correction by one position.
        const buffer = new SpellingBuffer();
        buffer.commit([]);
        expect(buffer.letters).toHaveLength(0);
    });

    it("swaps a letter for one of its runners-up", () => {
        const buffer = new SpellingBuffer();
        spell(buffer, ranked("W:0.5", "V:0.4", "U:0.1"), ranked("E:0.9"));
        expect(buffer.word).toBe("WE");

        // W, V and U are the handshape family the model confuses most; picking
        // between them is the whole point of keeping the candidates.
        buffer.choose(0, 1);
        expect(buffer.word).toBe("VE");
    });

    it("refuses a candidate index that does not exist", () => {
        const buffer = new SpellingBuffer();
        spell(buffer, ranked("A:0.9"));
        buffer.choose(0, 5);
        buffer.choose(9, 0);
        expect(buffer.word).toBe("A");
    });

    it("backspaces letters, then words", () => {
        const buffer = new SpellingBuffer();
        spell(buffer, ranked("H:0.9"), ranked("I:0.9"));
        buffer.space();
        spell(buffer, ranked("Y:0.9"), ranked("O:0.9"));
        expect(buffer.text).toBe("HI YO");

        buffer.backspace();
        expect(buffer.text).toBe("HI Y");
        buffer.backspace();
        expect(buffer.text).toBe("HI");

        // Once the current word is empty, backspace reopens the previous one
        // rather than doing nothing.
        buffer.backspace();
        expect(buffer.word).toBe("HI");
        buffer.backspace();
        expect(buffer.word).toBe("H");
    });

    it("does not start a word with a space", () => {
        const buffer = new SpellingBuffer();
        buffer.space();
        expect(buffer.text).toBe("");
    });

    it("replaces the word in progress when a suggestion is accepted", () => {
        const buffer = new SpellingBuffer();
        spell(buffer, ranked("H:0.9"), ranked("E:0.9"), ranked("L:0.9"));
        buffer.accept("hello");
        expect(buffer.text).toBe("hello");
        expect(buffer.letters).toHaveLength(0);
    });

    it("stops accepting letters at its length cap", () => {
        const buffer = new SpellingBuffer();
        for (let i = 0; i < 100; i++) buffer.commit(ranked("A:0.9"));
        expect(buffer.letters.length).toBeLessThanOrEqual(24);
    });
});

describe("WordSuggester", () => {
    const dictionary = ["cat", "cats", "cot", "car", "hello", "help", "held", "world", "vet", "wet"];

    it("completes a word from its opening letters", () => {
        const suggester = new WordSuggester(dictionary);
        const buffer = new SpellingBuffer();
        spell(buffer, ranked("h:0.9"), ranked("e:0.9"), ranked("l:0.9"));

        const words = suggester.suggest(buffer.letters).map((s) => s.word);
        expect(words).toContain("hello");
        expect(words).toContain("held");
        expect(words).not.toContain("cat");
    });

    it("finds the right word through a misread letter", () => {
        // The point of the whole design. The model's first choice spells "wet",
        // which is a real word, so nothing looks wrong — but "vet" is reachable
        // because V was the runner-up, and both are offered.
        const suggester = new WordSuggester(dictionary);
        const buffer = new SpellingBuffer();
        spell(buffer, ranked("w:0.55", "v:0.40"), ranked("e:0.95"), ranked("t:0.95"));

        const words = suggester.suggest(buffer.letters).map((s) => s.word);
        expect(words).toContain("wet");
        expect(words).toContain("vet");
        // Ordered by how much confidence each reading needs to borrow.
        expect(words.indexOf("wet")).toBeLessThan(words.indexOf("vet"));
    });

    it("ranks an exact-length word above a longer completion", () => {
        const suggester = new WordSuggester(dictionary);
        const buffer = new SpellingBuffer();
        spell(buffer, ranked("c:0.9"), ranked("a:0.9"), ranked("t:0.9"));

        const [first] = suggester.suggest(buffer.letters);
        expect(first.word).toBe("cat");
        expect(first.exact).toBe(true);
    });

    it("returns nothing when no word can start that way", () => {
        const suggester = new WordSuggester(dictionary);
        const buffer = new SpellingBuffer();
        spell(buffer, ranked("z:0.9"), ranked("q:0.9"));
        expect(suggester.suggest(buffer.letters)).toEqual([]);
    });

    it("returns nothing for an empty buffer", () => {
        expect(new WordSuggester(dictionary).suggest([])).toEqual([]);
    });

    it("never suggests a word shorter than what has been spelled", () => {
        const suggester = new WordSuggester(dictionary);
        const buffer = new SpellingBuffer();
        spell(buffer, ranked("c:0.9"), ranked("a:0.9"), ranked("t:0.9"), ranked("s:0.9"));

        const words = suggester.suggest(buffer.letters).map((s) => s.word);
        expect(words).toContain("cats");
        expect(words).not.toContain("cat");
    });

    it("handles a real-sized dictionary quickly", () => {
        // 60k words ship with the app; suggestion runs on every committed
        // letter, so it has to stay well inside a frame.
        const many = Array.from({ length: 60000 }, (_, i) => {
            const a = String.fromCharCode(97 + (i % 26));
            const b = String.fromCharCode(97 + ((i >> 5) % 26));
            return `${a}${b}${i.toString(36)}`;
        });
        const suggester = new WordSuggester(many);
        const buffer = new SpellingBuffer();
        spell(buffer, ranked("a:0.6", "b:0.3"), ranked("c:0.9"));

        const started = performance.now();
        suggester.suggest(buffer.letters);
        expect(performance.now() - started).toBeLessThan(16);
    });
});
