/**
 * Presentation control.
 *
 * Slide navigation is a good fit for gesture control precisely because it is
 * idempotent-ish and low-stakes: a mis-advanced slide costs a gesture to undo,
 * unlike a mis-fired system action.
 *
 * The deck is driven through `dextera_slide` events rather than by calling into
 * `SpatialDeck` directly, so any surface that listens for those events — the
 * built-in deck, an embedded reveal.js, a future native shell — is controlled
 * by the same plugin without modification.
 */

import { type GestureResult } from "@/lib/gesture-engine";
import { type DexteraPlugin } from "@/lib/plugin-engine";

type SlideCommand = "next" | "prev" | "first" | "last";

/**
 * Gestures this plugin responds to, by label.
 *
 * Kept separate from `ActionRegistry`'s user-editable bindings on purpose: this
 * is the plugin's own opinion about presentation control, and a user who wants
 * something different rebinds in the Mapper rather than editing a plugin.
 */
const SLIDE_GESTURES: Record<string, SlideCommand> = {
    two_up: "next",
    one: "prev",
    palm: "first",
    fist: "last",
};

function emit(command: SlideCommand) {
    if (typeof window === "undefined") return;
    window.dispatchEvent(new CustomEvent("dextera_slide", { detail: command }));
}

export const PresentationPlugin: DexteraPlugin = {
    id: "presentation-manager",
    name: "Presentation",
    version: "2.0.0",
    author: "DexteraAI",

    onGesture: (result: GestureResult) => {
        // Only act on segment onsets. A held pose produces ~30 results a
        // second; acting on each would run the deck to its last slide instantly.
        if (result.phase !== "onset" || result.rejected) return;

        const command = SLIDE_GESTURES[result.gestureName];
        if (command) {
            emit(command);
            return;
        }

        // Swipes are a natural second route to next/prev. The direction is
        // mirrored to the user's frame of reference, matching the velocity
        // convention in `gesture-engine.ts`.
        if (result.spatialIntent === "swipe_left" || result.spatialIntent === "hyper_left") {
            emit("next");
        } else if (
            result.spatialIntent === "swipe_right" ||
            result.spatialIntent === "hyper_right"
        ) {
            emit("prev");
        }
    },
};
