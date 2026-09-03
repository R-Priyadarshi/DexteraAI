# Deployment headers

The web app is a static export (`output: "export"`), so it can be served from
any static host. Two headers are worth setting, and Next.js **cannot** set them
for you in a static build — it drops `headers()` at export time and warns:

```
⚠ rewrites, redirects, and headers are not applied when exporting your
  application, detected (headers).
```

## What they do

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Together these make the page **cross-origin isolated**, which is the
precondition for `SharedArrayBuffer`, which is what ONNX Runtime Web needs for
its multithreaded WASM backend.

Without them the app still works. It runs the model single-threaded, so
inference is slower — noticeably on machines without WebGPU, where WASM is the
only backend available. The engine detects this at startup and logs which mode
it is in rather than requesting threads it cannot get.

`public/_headers` already covers **Netlify** and **Cloudflare Pages**; it ships
in the export automatically. Everything else needs configuring.

## nginx

```nginx
location / {
    add_header Cross-Origin-Opener-Policy   same-origin   always;
    add_header Cross-Origin-Embedder-Policy require-corp  always;
    add_header X-Content-Type-Options       nosniff       always;
    try_files $uri $uri.html $uri/index.html =404;
}

# A cached service worker pins clients to an old asset manifest indefinitely.
location = /sw.js {
    add_header Cache-Control "no-cache" always;
}
```

## Vercel — `vercel.json`

```json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "Cross-Origin-Opener-Policy", "value": "same-origin" },
        { "key": "Cross-Origin-Embedder-Policy", "value": "require-corp" }
      ]
    }
  ]
}
```

## Caddy

```
header {
    Cross-Origin-Opener-Policy   same-origin
    Cross-Origin-Embedder-Policy require-corp
}
```

## GitHub Pages

GitHub Pages cannot set custom headers at all, so cross-origin isolation is not
achievable there and the model will run single-threaded. That is a supported
configuration, just a slower one — if you need the threads, host elsewhere.

## Verifying

In the browser console on the deployed site:

```js
crossOriginIsolated   // must be true
```

The engine also logs its decision at startup, so the browser console tells you
which mode you actually got without having to remember the check.

## A caveat worth knowing

`require-corp` means every cross-origin subresource must opt in with
`Cross-Origin-Resource-Policy` or CORS. This app loads no third-party assets at
runtime — the fonts are self-hosted by `next/font`, and the model, WASM and
MediaPipe files are all same-origin — so enabling it is safe here. If you add a
third-party script or image later and it starts failing to load, this is why.
