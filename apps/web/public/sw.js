/**
 * Service worker for offline operation.
 *
 * The product's whole claim is that recognition runs on-device with nothing
 * leaving the machine. Until now that was true of the *data* but not of the
 * *app*: every load still fetched the page, the ONNX runtime, and a multi-
 * megabyte model over the network. Caching those closes the gap — after one
 * visit the app runs with no connection at all, which is what "on-device"
 * should mean.
 *
 * Two strategies, because the assets have opposite requirements:
 *
 *   - **Model and runtime assets are cache-first.** They are large, immutable
 *     for a given build, and re-downloading them on every load is the single
 *     biggest cost of opening the app. A changed model ships under a new cache
 *     version.
 *   - **Everything else is network-first with a cache fallback.** Pages and
 *     scripts must not go stale behind a deploy, so the network wins whenever
 *     it is available and the cache exists only to survive its absence.
 */

const VERSION = "v1";
const SHELL_CACHE = `dextera-shell-${VERSION}`;
const ASSET_CACHE = `dextera-assets-${VERSION}`;

/** Pages worth having before the first offline load. */
const SHELL = ["/", "/dashboard", "/plugins", "/manifest.webmanifest", "/icon.svg"];

/** Large immutable assets: the ONNX runtime, WASM, MediaPipe, model bundles. */
const isImmutableAsset = (url) =>
  url.pathname.startsWith("/onnx/") ||
  url.pathname.endsWith(".wasm") ||
  url.pathname.endsWith(".onnx") ||
  url.pathname.endsWith(".onnx.data");

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) =>
      // One missing entry must not fail the whole install, or a single 404
      // leaves the app with no service worker at all.
      Promise.allSettled(SHELL.map((url) => cache.add(url)))
    )
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k !== SHELL_CACHE && k !== ASSET_CACHE)
            .map((k) => caches.delete(k))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Only GET is cacheable, and cross-origin requests are left entirely alone.
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (isImmutableAsset(url)) {
    event.respondWith(
      caches.open(ASSET_CACHE).then(async (cache) => {
        const hit = await cache.match(request);
        if (hit) return hit;
        const response = await fetch(request);
        // Range requests return 206, which cannot be stored; and an error
        // response cached here would persist a broken model.
        if (response.ok && response.status === 200) {
          cache.put(request, response.clone());
        }
        return response;
      })
    );
    return;
  }

  event.respondWith(
    (async () => {
      try {
        const response = await fetch(request);
        if (response.ok) {
          const cache = await caches.open(SHELL_CACHE);
          cache.put(request, response.clone());
        }
        return response;
      } catch {
        const hit = await caches.match(request);
        if (hit) return hit;
        // A navigation with nothing cached still needs *something* back, or
        // the browser shows its own offline error instead of the app shell.
        if (request.mode === "navigate") {
          const shell = await caches.match("/dashboard");
          if (shell) return shell;
        }
        throw new Error("Offline and not cached");
      }
    })()
  );
});
