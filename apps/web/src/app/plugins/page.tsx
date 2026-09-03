"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { PluginEngine, type DexteraPlugin } from "@/lib/plugin-engine";
import { PresentationPlugin } from "@/lib/plugins/presentation";
import { ActionRegistry, type GestureAction } from "@/lib/action-registry";
import { FieldRow } from "@/components/Telemetry";

/**
 * What is actually loaded, not a storefront.
 *
 * This page previously listed three invented plugins behind a dead Install
 * button. It now reads the live PluginEngine and ActionRegistry, so it can only
 * ever show things that really exist in this build.
 */
export default function PluginsPage() {
    const [plugins, setPlugins] = useState<DexteraPlugin[]>([]);
    const [actions, setActions] = useState<GestureAction[]>([]);

    useEffect(() => {
        const engine = PluginEngine.getInstance();
        engine.register(PresentationPlugin);
        setPlugins(engine.listPlugins());

        setActions(ActionRegistry.getInstance().getAllActions());
    }, []);

    const byCategory = actions.reduce<Record<string, GestureAction[]>>((acc, a) => {
        (acc[a.category] ??= []).push(a);
        return acc;
    }, {});

    return (
        <div className="min-h-screen">
            <nav className="sticky top-0 z-50 border-b border-[var(--rule)] bg-[var(--field)]/95 backdrop-blur-[2px]">
                <div className="mx-auto flex max-w-[1200px] items-center justify-between px-6 py-4 lg:px-10">
                    <div className="flex items-baseline gap-3">
                        <Link href="/" className="display text-[15px] text-[var(--ink)] hover:opacity-70">
                            Dextera
                        </Link>
                        <span className="label">Extensions</span>
                    </div>
                    <Link href="/dashboard" className="label label-signal hover:opacity-70">
                        Console →
                    </Link>
                </div>
            </nav>

            <main className="mx-auto max-w-[1200px] px-6 py-14 lg:px-10 lg:py-20">
                <span className="label">Loaded in this build</span>
                <h1 className="display mt-4 max-w-2xl text-[clamp(2rem,4vw,3.2rem)] text-[var(--ink)]">
                    Extensions are code,
                    <br />
                    not a marketplace.
                </h1>
                <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-[var(--ink-2)]">
                    There is nothing to install and no account to sign into. A plugin is a
                    TypeScript object registered at startup; an action is a named function a
                    gesture can be bound to in the console&apos;s Mapper.
                </p>

                {/* Registered plugins */}
                <section className="mt-14">
                    <div className="mb-4 flex items-baseline justify-between border-b border-[var(--rule)] pb-3">
                        <span className="label">Registered plugins</span>
                        <span className="readout text-xs text-[var(--ink-3)]">{plugins.length}</span>
                    </div>

                    {plugins.length === 0 ? (
                        <p className="label">None registered</p>
                    ) : (
                        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                            {plugins.map((p) => (
                                <article key={p.id} className="panel p-5">
                                    <div className="flex items-baseline justify-between gap-3">
                                        <h2 className="display text-lg text-[var(--ink)]">{p.name}</h2>
                                        <span className="readout text-[10px] text-[var(--signal)]">
                                            v{p.version}
                                        </span>
                                    </div>
                                    <p className="mono mt-2 text-[11px] text-[var(--ink-3)]">{p.id}</p>
                                    <p className="mt-3 text-xs leading-relaxed text-[var(--ink-2)]">
                                        by {p.author}
                                    </p>
                                </article>
                            ))}
                        </div>
                    )}
                </section>

                {/* Bindable actions */}
                <section className="mt-14">
                    <div className="mb-4 flex items-baseline justify-between border-b border-[var(--rule)] pb-3">
                        <span className="label">Bindable actions</span>
                        <span className="readout text-xs text-[var(--ink-3)]">{actions.length}</span>
                    </div>

                    {actions.length === 0 ? (
                        <p className="label">Registry not available in this context</p>
                    ) : (
                        <div className="grid grid-cols-1 gap-10 md:grid-cols-2 lg:grid-cols-3">
                            {Object.entries(byCategory).map(([category, list]) => (
                                <div key={category}>
                                    <span className="label label-signal">{category}</span>
                                    <div className="mt-3">
                                        {list.map((a) => (
                                            <FieldRow key={a.id} name={a.name} value={a.id} />
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </section>

                {/* How to write one */}
                <section className="mt-14">
                    <div className="mb-4 border-b border-[var(--rule)] pb-3">
                        <span className="label">Writing a plugin</span>
                    </div>
                    <pre className="mono overflow-x-auto border border-[var(--rule)] bg-[var(--field-1)] p-5 text-[12px] leading-relaxed text-[var(--ink-2)]">
{`import type { DexteraPlugin } from "@/lib/plugin-engine";

export const MyPlugin: DexteraPlugin = {
  id: "my-plugin",
  name: "My Plugin",
  version: "1.0.0",
  author: "you",

  onGesture: (result) => {
    if (result.gestureName === "palm" && result.confidence > 0.8) {
      // your action here
    }
  },
};

// then, once at startup:
PluginEngine.getInstance().register(MyPlugin);`}
                    </pre>
                    <p className="mt-4 max-w-xl text-xs leading-relaxed text-[var(--ink-3)]">
                        Every recognised frame is broadcast to every registered plugin while the
                        console is unlocked. Gestures the base model does not know can be taught
                        in Studio first, then handled here like any other label.
                    </p>
                </section>
            </main>

            <footer className="border-t border-[var(--rule)]">
                <div className="mx-auto max-w-[1200px] px-6 py-10 lg:px-10">
                    <span className="label">Dextera · on-device gesture recognition</span>
                </div>
            </footer>
        </div>
    );
}
