# Removed: unmounted API router sketches

**This directory holds no code. It exists to record what was here and why it went.**

Nine router modules — `analytics.py`, `custom_gestures.py`, `integrations.py`,
`multimodal.py`, `notifications.py`, `plugins.py`, `rbac.py`, `retraining.py`
and `sync.py` — were sketches of possible server-side features. None was ever
included in `create_app()`, so none was ever reachable over HTTP. They were
first quarantined here, then deleted.

They are in git history if you ever want them back:

```bash
git log --oneline --diff-filter=D -- backend/experimental/
git show <commit>^:backend/experimental/sync.py
```

## Why deleted rather than kept

Unreachable code with security defects is still a liability. It gets mounted by
someone in a hurry, it trips security scanners, and it makes the API surface
look larger than it is. Each of these had a blocking problem:

| File | Problem |
|---|---|
| `sync.py` | **Path traversal.** Request `filename` passed straight to `os.path.join`, so `../../` escaped the sync directory on both upload and download. |
| `custom_gestures.py` | **Path traversal.** `name` interpolated into a file path for create and delete, allowing writes and deletes outside the intended directory. |
| `integrations.py` | **SSRF.** Unauthenticated endpoint POSTed to an arbitrary caller-supplied URL. Also imported `requests`, an undeclared dependency, so it failed at import time. |
| `rbac.py` | Implied access control, enforced nothing. Module-level dict of users and roles, no authentication, no other route consulting it. Worse than no RBAC, because it looked like protection. |
| `analytics.py` | Returned hardcoded fake numbers (`active_users: 5`, `gestures_recognized: 1200`). |
| `retraining.py` | `retrain_model()` was `time.sleep(5)`. Trained nothing. |
| `plugins.py`, `notifications.py`, `multimodal.py` | In-memory registries with no persistence, no auth, and no consumer. |

Common to all: no authentication, no rate limiting, no input validation, no
tests, and process-local mutable state that would not survive a restart or work
across more than one worker.

## What the app actually serves

| Endpoint | Source |
|---|---|
| `GET /api/health` | `backend/apps/api/routes.py` |
| `POST /api/predict` | `backend/apps/api/routes.py` |
| `WS /api/ws/stream` | `backend/apps/api/routes.py` |

## Before adding any of this back

DexteraAI runs inference **on-device**, and the API server is an optional
convenience for demos and integration tests — not the delivery path. A feature
that can run in the browser should.

Two of these have since been answered without a server at all:

- **Custom gesture sync** → gesture packs. Studio exports taught gestures as a
  JSON file of landmark coordinates that imports on another machine. No
  account, no upload, no endpoint.
- **Plugins** → `apps/web/src/lib/plugin-engine.ts`. A plugin is a TypeScript
  object registered at startup, not a record in a server registry.

If something genuinely belongs server-side, write it fresh against the current
codebase with authentication, input validation and tests, rather than reviving
a sketch that had none.
