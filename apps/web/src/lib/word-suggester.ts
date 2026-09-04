/**
 * Real words consistent with what was spelled, searched across the whole
 * candidate set rather than the letters that happened to win.
 *
 * This is where keeping candidates pays off. Per-letter accuracy on an unseen
 * person is 90.7%, so a five-letter word comes out clean 0.907^5 = 61% of the
 * time. But the correct letter is in the top three 99.5% of the time, so the
 * correct *word* is reachable from those candidates 0.995^5 = 97.5% of the
 * time. The gap between 61% and 97.5% is entirely a matter of looking.
 *
 * Searching every combination would be 3^n. Instead each dictionary word is
 * tested against the slots directly, which is linear in the dictionary and
 * needs no expansion — and the dictionary is bucketed by first letter, so only
 * the buckets the first slot allows are ever scanned.
 */

import type { LetterSlot } from "./spelling-buffer";

export interface Suggestion {
    word: string;
    /**
     * Product of the confidences of the letters this reading requires. A word
     * spelled exactly as the model's first choices scores highest; one needing
     * a third-choice letter scores lower but is still offered.
     */
    score: number;
    /** True when the word is exactly as long as what has been spelled. */
    exact: boolean;
}

export class WordSuggester {
    /** Words bucketed by first letter, so a query scans only what it can start with. */
    private readonly byInitial = new Map<string, string[]>();

    constructor(words: Iterable<string>) {
        for (const word of words) {
            if (!word) continue;
            const bucket = this.byInitial.get(word[0]);
            if (bucket) bucket.push(word);
            else this.byInitial.set(word[0], [word]);
        }
    }

    /**
     * Fetch a newline-separated word list.
     *
     * Deliberately not bundled: it is half a megabyte of text that only the
     * spelling surface needs, so it loads when that surface is first opened
     * rather than on every page view.
     */
    static async load(url: string): Promise<WordSuggester> {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`word list ${url}: ${response.status}`);
        }
        const text = await response.text();
        return new WordSuggester(text.split("\n").map((w) => w.trim().toLowerCase()));
    }

    get size(): number {
        let total = 0;
        for (const bucket of this.byInitial.values()) total += bucket.length;
        return total;
    }

    /**
     * Words that could be what the user is spelling, best first.
     *
     * Both completions and corrections: a word longer than the letters so far
     * is offered as a completion, one of the same length as a correction. Exact
     * matches rank above completions of equal confidence, since a finished word
     * is the more likely reading of a finished spelling.
     */
    suggest(slots: readonly LetterSlot[], limit = 5): Suggestion[] {
        if (slots.length === 0) return [];

        // Confidence per allowed letter, per position. Lower-cased once here
        // rather than per dictionary word.
        const allowed = slots.map((slot) => {
            const map = new Map<string, number>();
            for (const candidate of slot.candidates) {
                const letter = candidate.gestureName.toLowerCase();
                // A letter can appear twice if the vocabulary has case or digit
                // variants; keep the more confident reading.
                map.set(letter, Math.max(map.get(letter) ?? 0, candidate.confidence));
            }
            return map;
        });

        const found: Suggestion[] = [];
        for (const [initial, weight] of allowed[0]) {
            for (const word of this.byInitial.get(initial) ?? []) {
                if (word.length < slots.length) continue;

                let score = weight;
                let ok = true;
                for (let i = 1; i < slots.length; i++) {
                    const letterScore = allowed[i].get(word[i]);
                    if (letterScore === undefined) {
                        ok = false;
                        break;
                    }
                    score *= letterScore;
                }
                if (ok) {
                    found.push({ word, score, exact: word.length === slots.length });
                }
            }
        }

        found.sort((a, b) => {
            if (a.exact !== b.exact) return a.exact ? -1 : 1;
            if (b.score !== a.score) return b.score - a.score;
            // Shorter completions first: they are closer to what was spelled.
            return a.word.length - b.word.length;
        });
        return found.slice(0, limit);
    }
}
