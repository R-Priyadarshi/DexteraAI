"use client";

export type VoiceIntent = "confirm" | "abort" | "launch" | "stealth" | "reset";

interface IntentDefinition {
    intent: VoiceIntent;
    keywords: Record<string, number>; // keyword -> weight
}

/**
 * VoiceEngine — Industrial-grade weighted intent processor.
 * 
 * Uses fuzzy keyword weighting to identify intent from transcriptions.
 * Supports synonyms and phrase-based confidence scoring.
 */
export class VoiceEngine {
    private static instance: VoiceEngine;
    private recognition: any | null = null;
    /** Whether the browser's recogniser is currently running. */
    private isListening = false;

    /**
     * Whether we *want* to be listening.
     *
     * Distinct from `isListening`, which reflects the recogniser's actual
     * state. The Web Speech API ends a session on every pause in speech, so
     * continuous listening means restarting on `onend` — and the restart
     * decision has to consult intent, not state. Checking `isListening` there
     * could never work: `onend` sets it false immediately above the check, so
     * recognition stopped for good after the first silence.
     */
    private shouldListen = false;
    private onIntentDetected: (intent: VoiceIntent, confidence: number) => void = () => { };
    private onStatusChange: (status: "idle" | "listening" | "processing" | "error") => void = () => { };

    private readonly INTENT_LIBRARY: IntentDefinition[] = [
        {
            intent: "confirm",
            keywords: { "confirm": 1.0, "yes": 0.8, "proceed": 0.9, "do it": 0.7, "okay": 0.6, "correct": 0.5 }
        },
        {
            intent: "abort",
            keywords: { "abort": 1.0, "cancel": 0.9, "stop": 1.0, "emergency": 0.8, "halt": 0.9, "terminate": 1.0 }
        },
        {
            intent: "launch",
            keywords: { "launch": 1.0, "start": 0.8, "execute": 0.9, "begin": 0.7, "run": 0.6, "initiate": 1.0 }
        },
        {
            intent: "stealth",
            keywords: { "stealth": 1.0, "hide": 0.9, "vanish": 0.8, "cloak": 0.9, "invisible": 0.7, "disappear": 0.8 }
        },
        {
            intent: "reset",
            keywords: { "reset": 1.0, "clear": 0.8, "wipe": 0.9, "restart": 0.7, "recalibrate": 1.0 }
        }
    ];

    private constructor() {
        this.setupRecognition();
    }

    public static getInstance(): VoiceEngine {
        if (!VoiceEngine.instance) {
            VoiceEngine.instance = new VoiceEngine();
        }
        return VoiceEngine.instance;
    }

    private setupRecognition() {
        if (typeof window === "undefined") return;

        const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.warn("DexteraAI: Web Speech API not supported.");
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = "en-US";

        this.recognition.onstart = () => {
            this.isListening = true;
            this.onStatusChange("listening");
        };

        this.recognition.onend = () => {
            this.isListening = false;
            if (!this.shouldListen) {
                this.onStatusChange("idle");
                return;
            }
            // Restarting synchronously inside onend throws InvalidStateError in
            // some browsers, which would kill listening permanently.
            setTimeout(() => {
                if (!this.shouldListen) return;
                try {
                    this.recognition.start();
                } catch {
                    // Already starting, or the engine is shutting down. Either
                    // way the next onend will try again.
                }
            }, 150);
        };

        this.recognition.onresult = (event: any) => {
            const transcript = Array.from(event.results)
                .map((result: any) => result[0].transcript)
                .join(" ")
                .toLowerCase();

            this.onStatusChange("processing");
            this.parseIntent(transcript);
        };

        this.recognition.onerror = (event: any) => {
            // `no-speech` and `aborted` are routine in continuous listening —
            // reporting them as errors makes an idle microphone look broken.
            const benign = event?.error === "no-speech" || event?.error === "aborted";
            if (!benign) this.onStatusChange("error");
        };
    }

    private parseIntent(text: string) {
        let bestIntent: VoiceIntent | null = null;
        let maxScore = 0;

        for (const def of this.INTENT_LIBRARY) {
            let currentScore = 0;
            for (const [kw, weight] of Object.entries(def.keywords)) {
                if (text.includes(kw)) {
                    currentScore += weight;
                }
            }

            if (currentScore > maxScore) {
                maxScore = currentScore;
                bestIntent = def.intent;
            }
        }

        // Confidence heuristic: Normalize by max possible weight in the best intent
        if (bestIntent && maxScore >= 0.6) {
            const confidence = Math.min(1.0, maxScore);
            this.onIntentDetected(bestIntent, confidence);
        }
    }

    public start(
        callbacks: {
            onIntent: (intent: VoiceIntent, confidence: number) => void;
            onStatus: (status: "idle" | "listening" | "processing" | "error") => void;
        }
    ) {
        if (!this.recognition) return;
        this.onIntentDetected = callbacks.onIntent;
        this.onStatusChange = callbacks.onStatus;

        this.shouldListen = true;
        try {
            this.recognition.start();
        } catch {
            // start() throws if a session is already running, which is
            // harmless — `shouldListen` is set, so onend will keep it alive.
        }
    }

    public stop() {
        // Cleared before stopping, so the onend handler sees the intent and
        // does not immediately restart what we just stopped.
        this.shouldListen = false;
        this.isListening = false;
        this.recognition?.stop();
        this.onStatusChange("idle");
    }
}

export const voiceEngine = VoiceEngine.getInstance();
