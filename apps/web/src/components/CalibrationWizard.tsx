/**
 * CalibrationWizard
 * Industrial 3-step biometric setup.
 */

import React, { useState, useEffect, useRef } from 'react';
import { biometricEngine } from '@/lib/biometric-engine';
import { Landmark } from '@/lib/gesture-engine';
import { hapticEngine } from '@/lib/haptic-engine';
import { tacticalAudio } from '@/lib/tactical-audio';

interface Props {
    landmarks: Landmark[] | null;
    onComplete: () => void;
    onClose: () => void;
}

export const CalibrationWizard: React.FC<Props> = ({ landmarks, onComplete, onClose }) => {
    const [step, setStep] = useState(1);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState("Waiting for a hand");

    useEffect(() => {
        if (step === 2 && landmarks) {
            biometricEngine.captureSample(landmarks);
            setProgress(prev => {
                const next = prev + (100 / 50); // 50 samples
                if (next >= 100) {
                    setStep(3);
                    hapticEngine.pulse("success");
                    return 100;
                }
                return next;
            });
        }
    }, [landmarks, step]);

    const startSampling = () => {
        setStep(2);
        setStatus("Capturing");
        tacticalAudio.ping('neutral');
    };

    const finalize = () => {
        biometricEngine.finalizeCalibration();
        tacticalAudio.ping('success');
        hapticEngine.pulse("command_success");
        onComplete();
    };

    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center bg-[#020203]/90 backdrop-blur-[2px] animate-in fade-in duration-500">
            <div className="w-full max-w-xl panel p-12 rounded-[2px] border border-white/10 relative overflow-hidden">
                <div className="absolute top-0 left-0 h-px w-full bg-[var(--signal)]" />
                
                <header className="mb-12">
                    <div className="flex justify-between items-center mb-6">
                        <span className="label !text-[8px] text-[var(--signal)]">Calibration_Core</span>
                        <button onClick={onClose} className="text-[var(--ink-3)] hover:text-[var(--ink)] transition-colors text-[10px] font-mono tracking-widest uppercase">Abort</button>
                    </div>
                    <h2 className="text-4xl font-extralight tracking-tighter text-[var(--ink)] italic">
                        {step === 1 && "Initialization"}
                        {step === 2 && "Capturing Hand_Signature"}
                        {step === 3 && "Authorization Ready"}
                    </h2>
                </header>

                <div className="space-y-10">
                    <div className="relative h-1 w-full bg-white/5 rounded-full overflow-hidden">
                        <div 
                            className="h-full bg-[var(--signal)] transition-all duration-300"
                            style={{ width: `${step === 1 ? 33 : (step === 2 ? 66 : 100)}%` }}
                        />
                    </div>

                    <div className="text-[11px] font-mono leading-relaxed text-[var(--ink-3)] uppercase tracking-widest">
                        {step === 1 && (
                            <p>Position your hand clearly within the camera viewport. Maintain a neutral open palm posture to establish the spatial baseline.</p>
                        )}
                        {step === 2 && (
                            <p>Engine is analyzing geometric variance. Slowly oscillate your fingers to calibrate individual extension ratios.</p>
                        )}
                        {step === 3 && (
                            <p>Hand signature successfully mapped to the industrial core. Authorization token generated and stored locally.</p>
                        )}
                    </div>

                    <div className="pt-6">
                        {step === 1 && (
                            <button 
                                onClick={startSampling}
                                className="w-full py-5 rounded-[2px] bg-[var(--signal-4)] border border-[var(--signal)] text-[var(--signal)] text-[10px] font-black uppercase tracking-[0.5em] hover:bg-[var(--signal-4)] transition-all active:scale-95"
                            >
                                Start_Baseline_Scan
                            </button>
                        )}
                        {step === 2 && (
                            <div className="flex flex-col gap-4">
                                <div className="flex justify-between text-[8px] label">
                                    <span>Sampling_Density</span>
                                    <span>{progress.toFixed(0)}%</span>
                                </div>
                                <div className="h-12 w-full flex gap-1 items-end px-2 pb-2 bg-white/[0.02] rounded-[2px] overflow-hidden">
                                    {Array.from({ length: 40 }).map((_, i) => (
                                        <div 
                                            key={i}
                                            className={`flex-1 bg-[var(--signal-4)] rounded-t-sm transition-all duration-500`}
                                            style={{ height: `${Math.random() * 80 + 20}%`, opacity: progress > (i * 2.5) ? 1 : 0.1 }}
                                        />
                                    ))}
                                </div>
                            </div>
                        )}
                        {step === 3 && (
                            <button 
                                onClick={finalize}
                                className="w-full py-5 rounded-[2px] bg-[var(--signal-4)] border border-[var(--signal)] text-[var(--signal)] text-[10px] font-black uppercase tracking-[0.5em] hover:bg-[var(--signal-3)] transition-all active:scale-95"
                            >
                                Save_Signature_&_Authorize
                            </button>
                        )}
                    </div>
                </div>

                <div className="mt-12 pt-8 border-t border-white/5 flex gap-4 items-center">
                    <div className={`h-2 w-2 rounded-full  ${landmarks ? "bg-[var(--signal)]" : "bg-red-500"}`} />
                    <span className="text-[8px] font-mono text-[var(--ink-3)] uppercase tracking-[0.3em]">
                        {landmarks ? "Optical_Sync: LOCKED" : "Optical_Sync: SEARCHING"}
                    </span>
                </div>
            </div>
        </div>
    );
};
