"""OS-level actions the bridge is allowed to perform.

This module is the entire authority on what a gesture can do to the machine.
The wire protocol carries action *identifiers* from this registry and nothing
else — never a key code, never a command string. That is the difference between
a bridge that performs a fixed set of desktop actions and one that is a remote
code execution primitive with extra steps.

Adding a capability means adding it here, deliberately, with a name the user
sees in the console. It cannot be reached by crafting a message.

Actions are declared as data and only resolved to real input calls when one is
executed. Importing this module therefore touches no input device and does not
require `pynput` to be installed — which matters because importing it used to
do both: `pynput` opens an X connection at import, so on a headless machine the
import raised `ImportError: failed to acquire X connection`, taking the whole
test suite down with it. A registry you cannot inspect without a display is
also a registry you cannot test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

# What an action does, as data. Resolved against pynput at execution time.
ActionKind = Literal["tap", "chord", "scroll", "click"]


@dataclass(frozen=True, slots=True)
class BridgeAction:
    """One thing a gesture may do to the operating system.

    `kind` and `target` describe the action rather than performing it, so the
    registry can be listed, validated and tested without a display.
    """

    id: str
    name: str
    category: str
    description: str
    kind: ActionKind
    target: Any


class InputUnavailableError(RuntimeError):
    """Raised when input injection cannot be performed on this machine."""


# Controllers are created once, on first use, and cached here. Module-level
# instantiation would open an X connection at import time.
_controllers: dict[str, Any] = {}


def _get_controllers() -> tuple[Any, Any, Any, Any]:
    """Return (keyboard, mouse, Key, Button), importing pynput on first use.

    Deferred because `pynput` is an optional dependency (the `bridge` extra)
    and because it acquires a display connection eagerly — neither of which
    should be a precondition for importing this module.
    """
    if "keyboard" not in _controllers:
        try:
            from pynput.keyboard import Controller as KeyboardController
            from pynput.keyboard import Key
            from pynput.mouse import Button
            from pynput.mouse import Controller as MouseController
        except ImportError as exc:
            # pynput raises ImportError for two quite different reasons: the
            # package is absent, or it is present but cannot reach a display
            # server (it opens an X connection at import). Reporting "install
            # pynput" to someone on a headless box would send them in circles.
            import importlib.util

            installed = importlib.util.find_spec("pynput") is not None
            if installed:
                raise InputUnavailableError(
                    f"pynput cannot reach a display server: {exc}. "
                    "Input injection needs an X session; Wayland is not supported."
                ) from exc
            raise InputUnavailableError(
                'Input injection needs pynput. Install it with: pip install -e ".[bridge]"'
            ) from exc

        try:
            _controllers["keyboard"] = KeyboardController()
            _controllers["mouse"] = MouseController()
        except Exception as exc:
            # No display, or a platform pynput cannot drive. Wayland reaches
            # here too, which is why bridge/README.md says so plainly.
            raise InputUnavailableError(
                f"Input injection is unavailable on this display server: {exc}"
            ) from exc

        _controllers["Key"] = Key
        _controllers["Button"] = Button

    return (
        _controllers["keyboard"],
        _controllers["mouse"],
        _controllers["Key"],
        _controllers["Button"],
    )


def is_available() -> bool:
    """Whether this machine can actually inject input."""
    try:
        _get_controllers()
    except InputUnavailableError:
        return False
    return True


def _perform(action: BridgeAction) -> None:
    """Resolve an action's declared intent against pynput and run it."""
    # `key_enum` and `button_enum` are pynput's Key and Button classes; named
    # in lower case because they are locals, not class definitions.
    keyboard, mouse, key_enum, button_enum = _get_controllers()

    if action.kind == "tap":
        key = getattr(key_enum, action.target)
        keyboard.press(key)
        keyboard.release(key)

    elif action.kind == "chord":
        # Press modifiers, tap the final key, release in reverse order. Reverse
        # release matters: a modifier left stuck down because an exception
        # unwound the presses makes the keyboard unusable until the user
        # notices and presses it themselves.
        *modifier_names, final_name = action.target
        modifiers = [getattr(key_enum, n) for n in modifier_names]
        final = getattr(key_enum, final_name)
        pressed: list[Any] = []
        try:
            for modifier in modifiers:
                keyboard.press(modifier)
                pressed.append(modifier)
            keyboard.press(final)
            keyboard.release(final)
        finally:
            for modifier in reversed(pressed):
                keyboard.release(modifier)

    elif action.kind == "scroll":
        mouse.scroll(0, action.target)

    elif action.kind == "click":
        mouse.click(getattr(button_enum, action.target))


# The registry. Media keys are the highest-value bindings because they work
# against whatever application currently owns playback, with no per-app
# integration.
_ACTIONS: tuple[BridgeAction, ...] = (
    BridgeAction(
        "media_play_pause", "Play / pause", "media",
        "Toggles playback in whichever app owns media keys.",
        "tap", "media_play_pause",
    ),
    BridgeAction(
        "media_next", "Next track", "media",
        "Skips to the next track.", "tap", "media_next",
    ),
    BridgeAction(
        "media_previous", "Previous track", "media",
        "Returns to the previous track.", "tap", "media_previous",
    ),
    BridgeAction(
        "volume_up", "Volume up", "media",
        "Raises system volume.", "tap", "media_volume_up",
    ),
    BridgeAction(
        "volume_down", "Volume down", "media",
        "Lowers system volume.", "tap", "media_volume_down",
    ),
    BridgeAction(
        "volume_mute", "Mute", "media",
        "Toggles mute.", "tap", "media_volume_mute",
    ),
    BridgeAction(
        "slide_next", "Next slide", "presentation",
        "Sends Page Down, which advances most presentation software.",
        "tap", "page_down",
    ),
    BridgeAction(
        "slide_previous", "Previous slide", "presentation",
        "Sends Page Up.", "tap", "page_up",
    ),
    BridgeAction(
        "scroll_up", "Scroll up", "navigation",
        "Scrolls the focused window up.", "scroll", 3,
    ),
    BridgeAction(
        "scroll_down", "Scroll down", "navigation",
        "Scrolls the focused window down.", "scroll", -3,
    ),
    BridgeAction(
        "switch_window", "Switch window", "navigation",
        "Alt-Tab to the previously focused window.", "chord", ("alt", "tab"),
    ),
    BridgeAction(
        "click_left", "Click", "pointer",
        "Left mouse click at the current cursor position.", "click", "left",
    ),
)

ACTIONS: dict[str, BridgeAction] = {action.id: action for action in _ACTIONS}


def describe() -> list[dict[str, str]]:
    """Serialisable action list, for the console to display and bind against."""
    return [
        {
            "id": a.id,
            "name": a.name,
            "category": a.category,
            "description": a.description,
        }
        for a in _ACTIONS
    ]


def run(action_id: str) -> bool:
    """Execute an allowlisted action. Returns False if the id is unknown."""
    action = ACTIONS.get(action_id)
    if action is None:
        # Not an error worth raising: an unknown id is exactly what a
        # mismatched client version or a probing page produces, and neither
        # should be able to stop the bridge.
        logger.warning(f"Rejected unknown action id: {action_id!r}")
        return False

    try:
        _perform(action)
    except InputUnavailableError as exc:
        logger.error(f"Cannot perform {action_id}: {exc}")
        return False
    except Exception as exc:
        logger.error(f"Action {action_id} failed: {exc}")
        return False
    return True
