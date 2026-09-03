"use client";

import { useEffect } from "react";

/**
 * Registers the offline service worker.
 *
 * Registration is deliberately deferred until after `load`: the worker's
 * install pre-caches the app shell, and doing that while the page is still
 * fetching its own critical resources makes the first visit measurably slower
 * to become interactive.
 *
 * Renders nothing.
 */
export function ServiceWorker() {
    useEffect(() => {
        if (typeof navigator === "undefined" || !("serviceWorker" in navigator)) return;

        // Service workers require a secure context. localhost counts as one, so
        // this still works in development.
        const register = () => {
            navigator.serviceWorker
                .register("/sw.js")
                .catch((err) => console.warn("Service worker registration failed", err));
        };

        if (document.readyState === "complete") register();
        else window.addEventListener("load", register, { once: true });
    }, []);

    return null;
}
