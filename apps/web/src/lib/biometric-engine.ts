/**
 * BiometricEngine
 * Industrial-grade hand biometric calibration.
 * Captures user-specific hand landark geometry to tune heuristics.
 */

import { Landmark } from "./gesture-engine";

export interface HandSignature {
    palmSize: number;       // Normalized palm radius
    fingerRatios: number[]; // Ratios of finger lengths to palmSize
    fistThreshold: number;  // Calibrated distance for a "closed" state
}

export class BiometricEngine {
    private static instance: BiometricEngine;
    private signature: HandSignature | null = null;
    private calibrationSamples: Landmark[][] = [];
    private readonly SAMPLES_NEEDED = 30;

    private constructor() {
        this.loadSignature();
    }

    static getInstance(): BiometricEngine {
        if (!BiometricEngine.instance) {
            BiometricEngine.instance = new BiometricEngine();
        }
        return BiometricEngine.instance;
    }

    private loadSignature() {
        if (typeof window === "undefined") return;
        const saved = localStorage.getItem("dextera_biometric_sig");
        if (!saved) return;
        try {
            this.signature = JSON.parse(saved);
        } catch (err) {
            // This constructor runs at module import, so an unguarded parse
            // turns one corrupt localStorage entry into a blank console with a
            // stack trace — the sibling stores all guard theirs. Discard the
            // bad value so the user can simply recalibrate.
            console.error("BiometricEngine: discarding corrupt signature", err);
            localStorage.removeItem("dextera_biometric_sig");
            this.signature = null;
        }
    }

    public isCalibrated(): boolean {
        return this.signature !== null;
    }

    /**
     * Captures a sample during the calibration wizard.
     */
    public captureSample(landmarks: Landmark[]) {
        this.calibrationSamples.push(landmarks);
    }

    /**
     * Finalizes calibration and generates the HandSignature.
     */
    public finalizeCalibration(): HandSignature {
        if (this.calibrationSamples.length === 0) throw new Error("No calibration data.");

        // Average out palm size (0 to 5, 9, 13, 17)
        const palmSize = this.calculateAveragePalmSize();
        const fingerRatios = this.calculateFingerRatios(palmSize);
        
        this.signature = {
            palmSize,
            fingerRatios,
            fistThreshold: palmSize * 0.4 // Heuristic: fingers closer than 40% of palm size = closed
        };

        localStorage.setItem("dextera_biometric_sig", JSON.stringify(this.signature));
        this.calibrationSamples = [];
        return this.signature;
    }

    public getSignature(): HandSignature | null {
        return this.signature;
    }

    private calculateAveragePalmSize(): number {
        return this.calibrationSamples.reduce((acc, sample) => {
            const wrist = sample[0];
            const indexBase = sample[5];
            const dist = Math.sqrt(Math.pow(indexBase.x - wrist.x, 2) + Math.pow(indexBase.y - wrist.y, 2));
            return acc + dist;
        }, 0) / this.calibrationSamples.length;
    }

    private calculateFingerRatios(palmSize: number): number[] {
        const ratios = [0, 0, 0, 0, 0];
        const fingertips = [4, 8, 12, 16, 20];
        const bases = [1, 5, 9, 13, 17];

        this.calibrationSamples.forEach(sample => {
            fingertips.forEach((tipIdx, i) => {
                const base = sample[bases[i]];
                const tip = sample[tipIdx];
                const len = Math.sqrt(Math.pow(tip.x - base.x, 2) + Math.pow(tip.y - base.y, 2));
                ratios[i] += len / palmSize;
            });
        });

        return ratios.map(r => r / this.calibrationSamples.length);
    }
}

export const biometricEngine = BiometricEngine.getInstance();
