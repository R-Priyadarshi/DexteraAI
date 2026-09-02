/**
 * PluginEngine — The foundational layer for DexteraAI gesture plugins.
 * 
 * Plugins allow third-party developers to extend DexteraAI's utility
 * without modifying the core engine.
 */

import { GestureResult } from "./gesture-engine";

export interface DexteraPlugin {
    id: string;
    name: string;
    version: string;
    author: string;
    onGesture: (result: GestureResult) => void;
    initialize?: () => Promise<void>;
    onShutdown?: () => Promise<void>;
}

export class PluginEngine {
    private static instance: PluginEngine;
    private plugins: Map<string, DexteraPlugin> = new Map();

    private constructor() { }

    public static getInstance(): PluginEngine {
        if (!PluginEngine.instance) {
            PluginEngine.instance = new PluginEngine();
        }
        return PluginEngine.instance;
    }

    public register(plugin: DexteraPlugin) {
        console.log(`DexteraAI: Registering plugin [${plugin.name} v${plugin.version}]`);
        if (plugin.initialize) plugin.initialize();
        this.plugins.set(plugin.id, plugin);
    }

    public broadcast(result: GestureResult) {
        this.plugins.forEach(plugin => {
            try {
                plugin.onGesture(result);
            } catch (err) {
                console.error(`DexteraAI: Plugin error [${plugin.id}]:`, err);
            }
        });
    }

    public getPlugin(id: string): DexteraPlugin | undefined {
        return this.plugins.get(id);
    }

    public listPlugins(): DexteraPlugin[] {
        return Array.from(this.plugins.values());
    }
}
