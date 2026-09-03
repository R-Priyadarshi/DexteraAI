# Desktop bridge

A gesture recognised in the browser can only act inside its own tab. That is
the correct default for a web page and the wrong one for this product: the
point of gesture control is to drive the media player, the slides, and the
window you are actually looking at.

The bridge is a small local daemon that closes that gap without giving up the
on-device posture. It binds to loopback, and nothing it handles leaves the
machine.

## Running it

```bash
python -m bridge.server
```

It prints a token on startup. Open the console's **Desktop** panel, paste the
token, and connect. The token is also written to `~/.cache/dextera/bridge-token`
with owner-only permissions.

```bash
python -m bridge.server --port 8765 --allow-origin https://your-deployment
```

## Why it needs a token

A localhost WebSocket that injects keystrokes is dangerous in a way a localhost
HTTP endpoint is not. **Browsers do not apply the same-origin policy to
WebSocket connections**: any page in any tab can open `ws://127.0.0.1:8765`
with no preflight and no user interaction. An unauthenticated version of this
daemon would let any website you happen to have open type into your machine.

Three independent controls, all of which must pass:

| Control | Stops | Does not stop |
| --- | --- | --- |
| Loopback binding | Anything off-machine | Any local process or page |
| `Origin` allowlist | Arbitrary websites — the browser sets this header and a page cannot forge it | A native process, which sets any header it likes |
| Shared token | A local process that has not read the token file | Anything running as your user that reads the file |

Beyond authentication, the protocol is constrained by design: clients send
action **ids** from `bridge/actions.py`, never key codes and never command
strings. The worst a fully authenticated malicious client can do is the set of
things listed in that file — press play/pause, change volume, page through
slides. Adding a capability means adding it there deliberately; it cannot be
reached by crafting a message.

Requests are also rate-limited to 10 per second. The segmenter emits at most a
few events a second, so anything near that bound is a malfunctioning or hostile
client, and injecting input at machine speed would make the desktop unusable.

## Threat model, stated plainly

The token protects against *other software on the machine* and *other pages in
the browser*. It does not protect against a process running as your user, which
can read the token file — such a process could already inject input directly
and does not need the bridge. If you do not want gesture control over your
desktop, do not run the daemon; nothing starts it automatically.

## Platform support

Input injection uses `pynput`. This is developed and tested on **X11**. Wayland
restricts synthetic input by design, and `pynput` cannot inject there without a
compositor-specific portal or a `uinput`-based helper — the daemon will start
but actions will silently have no effect. macOS requires granting Accessibility
permission to the terminal running the daemon.

## Adding an action

Add a `BridgeAction` to `_ACTIONS` in `bridge/actions.py`. It appears in the
console's Desktop panel automatically, since the action list is sent to the
client on connect rather than duplicated in the front end.
