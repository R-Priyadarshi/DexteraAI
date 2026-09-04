/**
 * A word being fingerspelled, kept as candidates rather than letters.
 *
 * The reason this is not just a string: on this project's own measurement the
 * fingerspelling model gets the letter right 90.7% of the time on a person it
 * has never seen, but the right letter is in its top three 99.5% of the time.
 * Committing only the winner throws that away, and it throws it away
 * irreversibly — by the time the user sees "HELLL" the information that L was a
 * close call with O is gone.
 *
 * So each position keeps its whole candidate list, and correcting a letter is
 * picking from three rather than spelling it again. It is also what lets
 * `word-suggester` search across positions: five letters at 90.7% each is a
 * 61% chance of a clean word, while the true word being reachable from the
 * top-three at every position is 97.5%.
 *
 * Nothing here knows about frames or gestures. The caller decides when a letter
 * has been held long enough to mean it, and calls `commit` once.
 */

import type { GestureCandidate } from "./gesture-engine";

export interface LetterSlot {
    /** Ranked candidates for this position, most confident first. */
    candidates: GestureCandidate[];
    /** Index into `candidates` currently shown. Changed by `choose`. */
    chosen: number;
}

/** Longest word the buffer will hold before refusing more letters. */
const MAX_LETTERS = 24;

export class SpellingBuffer {
    private slots: LetterSlot[] = [];
    private words: string[] = [];

    /** Append a letter from the model's ranked candidates. */
    commit(candidates: readonly GestureCandidate[]): void {
        if (candidates.length === 0 || this.slots.length >= MAX_LETTERS) return;
        this.slots.push({ candidates: [...candidates], chosen: 0 });
    }

    /** Remove the last letter, or the last word once the current one is empty. */
    backspace(): void {
        if (this.slots.length > 0) {
            this.slots.pop();
            return;
        }
        // Pull the previous word back apart, so a mistake two words ago is
        // reachable without clearing everything. Its candidates are gone, so it
        // returns as fixed letters.
        const previous = this.words.pop();
        if (previous) {
            this.slots = [...previous].map((letter) => ({
                candidates: [{ gestureName: letter, gestureId: -1, confidence: 1 }],
                chosen: 0,
            }));
        }
    }

    /** Pick a different candidate for one position. */
    choose(slotIndex: number, candidateIndex: number): void {
        const slot = this.slots[slotIndex];
        if (!slot || candidateIndex < 0 || candidateIndex >= slot.candidates.length) return;
        slot.chosen = candidateIndex;
    }

    /** End the current word. */
    space(): void {
        if (this.slots.length === 0) return;
        this.words.push(this.word);
        this.slots = [];
    }

    /** Replace the word being spelled — how a suggestion is accepted. */
    accept(word: string): void {
        this.words.push(word);
        this.slots = [];
    }

    clear(): void {
        this.slots = [];
        this.words = [];
    }

    /** The word currently being spelled. */
    get word(): string {
        return this.slots.map((s) => s.candidates[s.chosen].gestureName).join("");
    }

    /** Everything typed, including the word in progress. */
    get text(): string {
        return [...this.words, this.word].filter(Boolean).join(" ");
    }

    get letters(): readonly LetterSlot[] {
        return this.slots;
    }

    get isEmpty(): boolean {
        return this.slots.length === 0 && this.words.length === 0;
    }
}
