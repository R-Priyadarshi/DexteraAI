/**
 * TacticalAudio Engine
 * Industrial-grade synthetic soundscape for neural synchronization.
 * Uses Web Audio API for zero-latency, non-hardcoded feedback.
 */

export class TacticalAudio {
    private static instance: TacticalAudio;
    private ctx: AudioContext | null = null;
    private masterGain: GainNode | null = null;
    private neuralHum: OscillatorNode | null = null;
    private humGain: GainNode | null = null;

    private constructor() {}

    static getInstance(): TacticalAudio {
        if (!TacticalAudio.instance) {
            TacticalAudio.instance = new TacticalAudio();
        }
        return TacticalAudio.instance;
    }

    private init() {
        if (this.ctx) return;
        this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
        this.masterGain = this.ctx.createGain();
        this.masterGain.gain.value = 0.5;
        this.masterGain.connect(this.ctx.destination);
    }

    /**
     * Start the continuous "Neural Hum" (Background Sync state)
     */
    startNeuralHum() {
        this.init();
        if (!this.ctx || !this.masterGain) return;

        if (this.neuralHum) this.stopNeuralHum();

        this.neuralHum = this.ctx.createOscillator();
        this.humGain = this.ctx.createGain();
        
        // Industrial low-frequency hum (55Hz - A1)
        this.neuralHum.type = 'sine';
        this.neuralHum.frequency.setValueAtTime(55, this.ctx.currentTime);
        
        this.humGain.gain.setValueAtTime(0, this.ctx.currentTime);
        this.humGain.gain.linearRampToValueAtTime(0.05, this.ctx.currentTime + 2);

        this.neuralHum.connect(this.humGain);
        this.humGain.connect(this.masterGain);
        this.neuralHum.start();
    }

    stopNeuralHum() {
        if (this.humGain && this.ctx) {
            this.humGain.gain.linearRampToValueAtTime(0, this.ctx.currentTime + 0.5);
            setTimeout(() => {
                this.neuralHum?.stop();
                this.neuralHum = null;
            }, 500);
        }
    }

    /**
     * Play a tactical confirmation ping
     */
    ping(type: 'success' | 'warning' | 'alert' | 'neutral' = 'neutral') {
        this.init();
        if (!this.ctx || !this.masterGain) return;

        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.connect(gain);
        gain.connect(this.masterGain);

        const now = this.ctx.currentTime;

        switch (type) {
            case 'success':
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(880, now); // A5
                osc.frequency.exponentialRampToValueAtTime(1760, now + 0.1);
                gain.gain.setValueAtTime(0.2, now);
                gain.gain.exponentialRampToValueAtTime(0.01, now + 0.3);
                osc.start(now);
                osc.stop(now + 0.3);
                break;
            case 'alert':
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(220, now);
                osc.frequency.linearRampToValueAtTime(110, now + 0.2);
                gain.gain.setValueAtTime(0.3, now);
                gain.gain.linearRampToValueAtTime(0, now + 0.2);
                osc.start(now);
                osc.stop(now + 0.2);
                break;
            case 'neutral':
                osc.type = 'sine';
                osc.frequency.setValueAtTime(440, now);
                gain.gain.setValueAtTime(0.1, now);
                gain.gain.exponentialRampToValueAtTime(0.01, now + 0.1);
                osc.start(now);
                osc.stop(now + 0.1);
                break;
        }
    }

    /**
     * High-precision "Bitstream" click for data events
     */
    bitClick() {
        this.init();
        if (!this.ctx || !this.masterGain) return;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'square';
        osc.frequency.setValueAtTime(2000, this.ctx.currentTime);
        gain.gain.setValueAtTime(0.02, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.02);
        osc.connect(gain);
        gain.connect(this.masterGain);
        osc.start();
        osc.stop(this.ctx.currentTime + 0.02);
    }
}

export const tacticalAudio = TacticalAudio.getInstance();
