"use client";

import { useState, useEffect } from "react";

const SLIDES = [
    { id: 1, title: "DexteraAI_Core", content: "Transformer-based Temporal Classification", color: "from-blue-500/20" },
    { id: 2, title: "Neural_Calibration", content: "Dynamic Environmental Noise Measurement", color: "from-purple-500/20" },
    { id: 3, title: "Multimodal_Fusion", content: "Probabilistic Intent Sync: Voice + Gesture", color: "from-emerald-500/20" },
    { id: 4, title: "Industrial_Scale", content: "Edge-Native Performance (< 15ms Latency)", color: "from-amber-500/20" },
];

export function SpatialDeck() {
    const [currentSlide, setCurrentSlide] = useState(0);
    const [isAnimating, setIsAnimating] = useState(false);

    useEffect(() => {
        const handleSlide = (e: any) => {
            if (isAnimating) return;
            const direction = e.detail;
            console.log(`DexteraAI: SpatialDeck received slide instruction: [${direction}]`);
            
            setIsAnimating(true);
            if (direction === "next") {
                setCurrentSlide((prev) => Math.min(prev + 1, SLIDES.length - 1));
            } else if (direction === "prev") {
                setCurrentSlide((prev) => Math.max(prev - 1, 0));
            } else if (direction === "first") {
                setCurrentSlide(0);
            } else if (direction === "last") {
                setCurrentSlide(SLIDES.length - 1);
            }
            setTimeout(() => setIsAnimating(false), 600);
        };

        window.addEventListener("dextera_slide", handleSlide);
        return () => window.removeEventListener("dextera_slide", handleSlide);
    }, [isAnimating]);

    return (
        <div className="spatial-card relative h-[300px] w-full overflow-hidden rounded-[2rem] border-white/[0.03] bg-black/40">
            <div className={`absolute inset-0 bg-gradient-to-br ${SLIDES[currentSlide].color} to-transparent opacity-30 transition-all duration-1000`} />
            
            <div className="absolute top-8 left-10 flex items-center gap-3">
                <div className="h-1 w-6 bg-blue-500 rounded-full" />
                <span className="hud-label">Project_Ultra_Briefing</span>
            </div>

            <div className="flex h-full flex-col items-center justify-center p-12 text-center">
                <div className={`transition-all duration-700 ${isAnimating ? "opacity-0 scale-95 translate-y-4" : "opacity-100 scale-100 translate-y-0"}`}>
                    <h3 className="text-[10px] font-mono tracking-[0.4em] text-white/40 uppercase mb-4">Module_{currentSlide + 1}</h3>
                    <h2 className="text-3xl font-light tracking-tight text-white mb-2 italic">
                        {SLIDES[currentSlide].title}
                    </h2>
                    <p className="text-sm text-[#86868b] max-w-xs uppercase tracking-widest leading-loose">
                        {SLIDES[currentSlide].content}
                    </p>
                </div>
            </div>

            {/* Pagination HUD */}
            <div className="absolute bottom-8 inset-x-0 flex justify-center gap-2">
                {SLIDES.map((_, i) => (
                    <div 
                        key={i} 
                        className={`h-1 transition-all duration-500 rounded-full ${i === currentSlide ? "w-8 bg-blue-500" : "w-2 bg-white/10"}`} 
                    />
                ))}
            </div>
        </div>
    );
}
