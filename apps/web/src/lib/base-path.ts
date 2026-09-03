/**
 * Runtime asset URLs, corrected for the base path the site is served under.
 *
 * The engine loads its model bundles, WASM runtimes and the MediaPipe script by
 * URL at runtime rather than importing them, so Next's bundler never sees those
 * strings and its `basePath` rewriting does not reach them. That is fine at a
 * domain root and breaks entirely one level down: a GitHub Pages project site
 * lives at `/<repo>/`, where a literal `/onnx/...` resolves against the domain
 * root and 404s. The camera then runs with nothing behind it, which looks like
 * a broken model rather than a missing file.
 *
 * `NEXT_PUBLIC_BASE_PATH` is inlined at build time, so this costs nothing at
 * runtime and is empty for the common case of serving from a root.
 */
export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

/**
 * Prefix a root-relative asset path with the deployment's base path.
 *
 * Pass paths that start with "/". An absolute URL is returned unchanged, so a
 * bundle hosted on a CDN still works.
 */
export function asset(path: string): string {
    if (/^[a-z][a-z0-9+.-]*:/i.test(path) || path.startsWith("//")) {
        return path;
    }
    return `${BASE_PATH}${path}`;
}
