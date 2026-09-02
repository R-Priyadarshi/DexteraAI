import Link from "next/link";

export default function LandingPage() {
    return (
        <div className="relative min-h-screen overflow-hidden bg-[#050505]">
            {/* Cinematic Background Layer */}
            <div className="absolute inset-0 neural-grid opacity-20 pointer-events-none" />
            <div className="absolute top-[-10%] left-[-10%] right-[-10%] bottom-[-10%] neural-glow animate-pulse-soft pointer-events-none" />

            {/* Decorative Orbs */}
            <div className="absolute top-[20%] left-[10%] h-96 w-96 rounded-full bg-blue-500/10 blur-[120px] animate-float" />
            <div className="absolute bottom-[10%] right-[10%] h-[30rem] w-[30rem] rounded-full bg-slate-800/20 blur-[150px] animate-float" style={{ animationDelay: "-5s" }} />

            <nav className="fixed top-0 z-50 flex w-full items-center justify-between px-8 py-10 lg:px-20">
                <div className="flex items-center gap-3">
                    <div className="h-[2px] w-6 bg-blue-500" />
                    <span className="text-sm font-medium tracking-[0.3em] text-[#f5f5f7] uppercase">DexteraAI</span>
                </div>
                <div className="hidden lg:flex items-center gap-12">
                    {["Intelligence", "Security", "Ecosystem"].map((item) => (
                        <span key={item} className="text-[10px] font-bold tracking-[0.2em] text-[#86868b] uppercase cursor-pointer hover:text-white transition-colors">{item}</span>
                    ))}
                </div>
            </nav>

            <main className="relative z-10 flex flex-col items-center px-6 pt-40 lg:pt-56">
                <div className="flex flex-col items-center text-center space-y-8 w-full max-w-none px-6 lg:px-20">
                    <div className="spatial-panel rounded-full px-5 py-1.5 animate-in fade-in slide-in-from-bottom-4 duration-1000">
                        <span className="text-[10px] font-bold tracking-[0.3em] text-blue-500 uppercase">Industrial Standard v0.1</span>
                    </div>

                    <h1 className="text-5xl lg:text-8xl font-light tracking-tighter leading-[1.1] reveal-text animate-in fade-in slide-in-from-bottom-8 duration-1000 delay-200">
                        Hand-Biometric <br />
                        <span className="font-medium">Intelligence.</span>
                    </h1>

                    <p className="max-w-2xl text-lg lg:text-xl font-light text-[#86868b] leading-relaxed animate-in fade-in slide-in-from-bottom-12 duration-1000 delay-500">
                        Redefining human-machine interaction through zero-latency spatial compute.
                        100% On-device. 100% Private. Built for the trillion-dollar age.
                    </p>

                    <div className="pt-12 animate-in fade-in slide-in-from-bottom-16 duration-1000 delay-700">
                        <Link href="/dashboard">
                            <button className="group relative flex items-center gap-8 overflow-hidden rounded-full bg-white px-10 py-5 transition-all hover:scale-[1.02] active:scale-[0.98]">
                                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-black/[0.05] to-transparent animate-shimmer opacity-0 group-hover:opacity-100" />
                                <span className="text-sm font-semibold tracking-widest text-black uppercase">Enter Command Center</span>
                                <div className="h-px w-6 bg-black/20 group-hover:w-10 transition-all duration-500" />
                            </button>
                        </Link>
                    </div>
                </div>

                {/* Feature Triptych */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-10 mt-40 lg:mt-60 w-full pb-20 px-10 lg:px-20 animate-in fade-in slide-in-from-bottom-20 duration-1000 delay-1000">
                    <div className="flex flex-col space-y-4 group p-8 rounded-3xl border border-white/5 bg-white/[0.02] hover:bg-white/[0.04] transition-all cursor-crosshair">
                        <span className="text-xs font-bold tracking-widest text-blue-500 uppercase">01. Intelligence</span>
                        <h3 className="text-2xl font-light text-white">Under 5ms.</h3>
                        <p className="text-sm text-[#86868b] leading-relaxed">Proprietary Transformer-based architecture optimized for edge GPU execution.</p>
                        <div className="h-px w-full bg-white/5 mt-4 overflow-hidden">
                            <div className="h-full w-2/3 bg-blue-500 animate-shimmer" />
                        </div>
                    </div>
                    <div className="flex flex-col space-y-4 group p-8 rounded-3xl border border-white/5 bg-white/[0.02] hover:bg-white/[0.04] transition-all cursor-crosshair">
                        <span className="text-xs font-bold tracking-widest text-blue-500 uppercase">02. Security</span>
                        <h3 className="text-2xl font-light text-white">Edge Compute.</h3>
                        <p className="text-sm text-[#86868b] leading-relaxed">No data leaves the device. Biometric signatures are processed and discarded in RAM.</p>
                        <div className="flex gap-2 mt-4">
                            <div className="h-1 w-1 rounded-full bg-green-500" />
                            <span className="text-[8px] font-mono text-white/30 tracking-widest">ENCRYPTION_SAFE</span>
                        </div>
                    </div>
                    <div className="flex flex-col space-y-4 group p-8 rounded-3xl border border-white/5 bg-white/[0.02] hover:bg-white/[0.04] transition-all cursor-crosshair">
                        <span className="text-xs font-bold tracking-widest text-blue-500 uppercase">03. Ecosystem</span>
                        <h3 className="text-2xl font-light text-white">Universal.</h3>
                        <p className="text-sm text-[#86868b] leading-relaxed">A seamless bridge between physical gestures and digital ecosystems.</p>
                        <span className="text-[10px] text-blue-500/60 font-mono mt-4">v0.1.0_INDUSTRIAL</span>
                    </div>
                </div>
            </main>

            {/* Footer Branding */}
            <footer className="relative z-10 flex w-full border-t border-white/[0.03] px-8 py-10 lg:px-20 items-center justify-between">
                <span className="text-[9px] font-mono tracking-widest text-[#86868b] uppercase">Dextera Neural Systems Lab</span>
                <span className="text-[9px] font-mono tracking-widest text-[#86868b] uppercase">Est. 2026 — Secure Biometry</span>
            </footer>
        </div>
    );
}
