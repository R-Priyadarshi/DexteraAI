# Experimental / Unmounted API Routers

**Status: not part of the product. Not wired into the running app. Do not mount as-is.**

These modules were written as sketches of possible server-side features. None of them
were ever included in `create_app()` (`backend/apps/api/main.py`), so none of them have
ever been reachable over HTTP. They were moved here out of `backend/apps/api/` so that
the live API surface is exactly what the app actually serves:

| Live endpoint | Source |
|---|---|
| `GET /api/health` | `backend/apps/api/routes.py` |
| `POST /api/predict` | `backend/apps/api/routes.py` |
| `WS /api/ws/stream` | `backend/apps/api/routes.py` |

DexteraAI runs inference **on-device**. The API server is optional and exists for demos
and integration testing, so the product does not need most of what is sketched here.

## Why these are not simply enabled

Each has blocking problems that must be fixed before it could be exposed to a network:

| File | Problem |
|---|---|
| `sync.py` | **Path traversal.** `filename` from the request is passed straight to `os.path.join`, so `../../` escapes the sync directory on both upload and download. |
| `custom_gestures.py` | **Path traversal.** The `name` parameter is interpolated into a file path for create and delete, allowing writes and deletes outside the intended directory. |
| `integrations.py` | **SSRF.** Unauthenticated endpoint POSTs to an arbitrary caller-supplied URL. Also imports `requests`, which is not a declared dependency, so it fails at import time. |
| `rbac.py` | Implies access control but enforces nothing. Users and roles live in a module-level dict, there is no authentication, and no other route consults it. Worse than no RBAC, because it looks like protection. |
| `analytics.py` | Returns hardcoded fake numbers (`active_users: 5`, `gestures_recognized: 1200`). |
| `retraining.py` | `retrain_model()` is a `time.sleep(5)` stub that trains nothing. |
| `plugins.py`, `notifications.py`, `multimodal.py` | In-memory registries with no persistence, no auth, and no consumer. |

Common to all of them: no authentication, no rate limiting, no input validation, no
tests, and process-local mutable state that would not survive a restart or work across
more than one worker.

## If you want one of these for real

1. Decide it belongs server-side at all. On-device is the default for this project, and
   a feature that can run in the browser should.
2. Fix the security issue in the table above first.
3. Add authentication and validated request/response models in `backend/apps/api/schemas.py`.
4. Add tests under `tests/`.
5. Mount it in `create_app()` and document it in `docs/api-reference.md`.

Until then, treat this directory as design notes, not shippable code.
