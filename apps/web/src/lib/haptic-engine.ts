import { tacticalAudio } from "./tactical-audio";

export type HapticPattern = "light" | "success" | "error" | "sonar_ping" | "kinetic_warning" | "command_success" | "lock";

/**
 * HapticEngine — Industrial Tactile Feedback System.
 * 
 * Provides high-fidelity biometric feedback via physical vibration (Web Vibrate API)
 * and unified Tactical Audio synchronization.
 */
export class HapticEngine {
    private static instance: HapticEngine;
    private isEnabled = true;

    private constructor() { }

    public static getInstance(): HapticEngine {
        if (!HapticEngine.instance) {
            HapticEngine.instance = new HapticEngine();
        }
        return HapticEngine.instance;
    }

    /**
     * Triggers a tactile and audio pulse.
     */
    public pulse(pattern: HapticPattern = "light") {
        if (!this.isEnabled || typeof window === "undefined") return;

        // 1. Physical Haptic (Precision Vibration)
        if ("vibrate" in navigator) {
            switch (pattern) {
                case "light": navigator.vibrate(5); break;
                case "success": navigator.vibrate([15, 20, 15]); break;
                case "error": navigator.vibrate([40, 40, 40]); break;
                case "sonar_ping": navigator.vibrate(2); break;
                case "kinetic_warning": navigator.vibrate([5, 5, 5, 5, 5]); break;
                case "command_success": navigator.vibrate([10, 40, 20, 60]); break;
                case "lock": navigator.vibrate([100, 50, 100]); break;
            }
        }

        // 2. Tactical Audio Synchronization
        switch (pattern) {
            case "success":
            case "command_success":
                tacticalAudio.ping('success');
                break;
            case "error":
            case "lock":
                tacticalAudio.ping('alert');
                break;
            default:
                tacticalAudio.ping('neutral');
                break;
        }

        // 3. Global System Event
        window.dispatchEvent(new CustomEvent("dextera-haptic", { detail: { pattern } }));
    }

    public setEnabled(enabled: boolean) {
        this.isEnabled = enabled;
    }
}

export const hapticEngine = HapticEngine.getInstance();
