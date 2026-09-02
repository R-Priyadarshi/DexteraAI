"use client";

import { type Landmark } from "./gesture-engine";

export interface CustomGesture {
    id: string;
    name: string;
    samples: Landmark[][]; // Array of landmark arrays (frames)
    actionId?: string;
    createdAt: number;
}

export class GestureStore {
    private static STORAGE_KEY = "dextera_custom_gestures";
    private gestures: CustomGesture[] = [];

    constructor() {
        this.load();
    }

    private load() {
        if (typeof window === "undefined") return;
        const saved = localStorage.getItem(GestureStore.STORAGE_KEY);
        if (saved) {
            try {
                this.gestures = JSON.parse(saved);
            } catch (e) {
                console.error("DexteraAI: Failed to load custom gestures", e);
                this.gestures = [];
            }
        }
    }

    private save() {
        if (typeof window === "undefined") return;
        localStorage.setItem(GestureStore.STORAGE_KEY, JSON.stringify(this.gestures));
    }

    public addGesture(name: string, samples: Landmark[][]) {
        const newGesture: CustomGesture = {
            id: `custom_${Date.now()}`,
            name,
            samples,
            createdAt: Date.now()
        };
        this.gestures.push(newGesture);
        this.save();
        return newGesture;
    }

    public getGestures(): CustomGesture[] {
        return this.gestures;
    }

    public deleteGesture(id: string) {
        this.gestures = this.gestures.filter(g => g.id !== id);
        this.save();
    }

    public clear() {
        this.gestures = [];
        this.save();
    }
}

export const gestureStore = new GestureStore();
