"use client";

import { useEffect, useState } from "react";

/**
 * A slide surface driven entirely by gestures, via the `dextera_slide` event
 * that the deck actions in `action-registry.ts` dispatch.
 *
 * It doubles as the instructions for itself: the deck explains which gesture
 * moves it, so the only way to read all of it is to actually drive it with
 * your hand. That makes it a demonstration rather than a mock-up.
 */
const SLIDES = [
    {
        kicker: "01 — Try it",
        title: "This deck has no buttons.",
        body: "Open Mapper, bind a gesture to \u201cNext Deck Slide\u201d, then unlock the guard and perform it. The deck advances with no key or click involved.",
    },
    {
        kicker: "02 — What just happened",
        title: "Gesture → event → action.",
        body: "The classifier emitted a label. The action registry matched it to a bound action, which dispatched a slide event this panel listens for. Nothing on that path is specific to slides.",
    },
    {
        kicker: "03 — Writing your own",
        title: "A plugin is one interface.",
        body: "Implement DexteraPlugin with an id and an onGesture handler, register it, and any recognised gesture becomes yours to act on.",
    },
    {
        kicker: "04 — Beyond the vocabulary",
        title: "Teach it something new.",
        body: "Open Studio and demonstrate a gesture the model has never seen a few times. It is stored locally as a few-shot class — no retraining, no upload.",
    },
];

export function SpatialDeck() {
    const [current, setCurrent] = useState(0);
    const [changing, setChanging] = useState(false);

    useEffect(() => {
        const handleSlide = (e: Event) => {
            const direction = (e as CustomEvent<string>).detail;
            setChanging(true);
            setCurrent((prev) => {
                if (direction === "next") return Math.min(prev + 1, SLIDES.length - 1);
                if (direction === "prev") return Math.max(prev - 1, 0);
                if (direction === "first") return 0;
                if (direction === "last") return SLIDES.length - 1;
                return prev;
            });
            const t = setTimeout(() => setChanging(false), 260);
            return () => clearTimeout(t);
        };

        window.addEventListener("dextera_slide", handleSlide);
        return () => window.removeEventListener("dextera_slide", handleSlide);
    }, []);

    const slide = SLIDES[current];

    return (
        <div className="relative">
            <div className="mb-5 flex items-center justify-between border-b border-[var(--rule)] pb-3">
                <span className="label">Gesture-driven deck</span>
                <span className="readout text-[10px] text-[var(--ink-3)]">
                    {String(current + 1).padStart(2, "0")} / {String(SLIDES.length).padStart(2, "0")}
                </span>
            </div>

            <div
                className="grid grid-cols-1 gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]"
                style={{
                    opacity: changing ? 0.35 : 1,
                    transition: "opacity 200ms ease",
                }}
            >
                <div>
                    <span className="label label-signal">{slide.kicker}</span>
                    <h3 className="display mt-3 text-2xl text-[var(--ink)]">{slide.title}</h3>
                </div>
                <p className="max-w-xl text-sm leading-relaxed text-[var(--ink-2)]">
                    {slide.body}
                </p>
            </div>

            {/* Position indicator: segments, not dots */}
            <div className="mt-6 flex gap-1">
                {SLIDES.map((_, i) => (
                    <div
                        key={i}
                        className="h-[2px] flex-1 transition-colors duration-300"
                        style={{
                            background: i === current ? "var(--signal)" : "var(--rule)",
                        }}
                    />
                ))}
            </div>
        </div>
    );
}
