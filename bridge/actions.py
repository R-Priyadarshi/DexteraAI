"""OS-level actions the bridge is allowed to perform.

This module is the entire authority on what a gesture can do to the machine.
The wire protocol carries action *identifiers* from this registry and nothing
else — never a key code, never a command string. That is the difference between
a bridge that performs a fixed set of desktop actions and one that is a remote
code execution primitive with extra steps.

Adding a capability means adding it here, deliberately, with a name the user
sees in the console. It cannot be reached by crafting a message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

if TYPE_CHECKING:
    from collections.abc import Callable

_keyboard = KeyboardController()
_mouse = MouseController()


@dataclass(frozen=True, slots=True)
class BridgeAction:
    """One thing a gesture may do to the operating system."""

    id: str
    name: str
    category: str
    description: str
    run: Callable[[], None]


def _tap(key: object) -> Callable[[], None]:
    def action() -> None:
        _keyboard.press(key)  # type: ignore[arg-type]
        _keyboard.release(key)  # type: ignore[arg-type]

    return action


def _chord(*keys: object) -> Callable[[], None]:
    """Press modifiers, tap the final key, release in reverse order.

    Reverse-order release matters: leaving a modifier stuck down because an
    exception unwound the presses would make the keyboard unusable until the
    user noticed and pressed it themselves.
    """

    def action() -> None:
        *modifiers, final = keys
        pressed: list[object] = []
        try:
            for modifier in modifiers:
                _keyboard.press(modifier)  # type: ignore[arg-type]
                pressed.append(modifier)
            _keyboard.press(final)  # type: ignore[arg-type]
            _keyboard.release(final)  # type: ignore[arg-type]
        finally:
            for modifier in reversed(pressed):
                _keyboard.release(modifier)  # type: ignore[arg-type]

    return action


def _scroll(dy: int) -> Callable[[], None]:
    def action() -> None:
        _mouse.scroll(0, dy)

    return action


def _click(button: Button) -> Callable[[], None]:
    def action() -> None:
        _mouse.click(button)

    return action


# The registry. Media keys are the highest-value bindings because they work
# against whatever application currently owns playback, with no per-app
# integration.
_ACTIONS: tuple[BridgeAction, ...] = (
    BridgeAction(
        "media_play_pause",
        "Play / pause",
        "media",
        "Toggles playback in whichever app owns media keys.",
        _tap(Key.media_play_pause),
    ),
    BridgeAction(
        "media_next",
        "Next track",
        "media",
        "Skips to the next track.",
        _tap(Key.media_next),
    ),
    BridgeAction(
        "media_previous",
        "Previous track",
        "media",
        "Returns to the previous track.",
        _tap(Key.media_previous),
    ),
    BridgeAction(
        "volume_up", "Volume up", "media", "Raises system volume.", _tap(Key.media_volume_up)
    ),
    BridgeAction(
        "volume_down",
        "Volume down",
        "media",
        "Lowers system volume.",
        _tap(Key.media_volume_down),
    ),
    BridgeAction(
        "volume_mute", "Mute", "media", "Toggles mute.", _tap(Key.media_volume_mute)
    ),
    BridgeAction(
        "slide_next",
        "Next slide",
        "presentation",
        "Sends Page Down, which advances most presentation software.",
        _tap(Key.page_down),
    ),
    BridgeAction(
        "slide_previous",
        "Previous slide",
        "presentation",
        "Sends Page Up.",
        _tap(Key.page_up),
    ),
    BridgeAction(
        "scroll_up", "Scroll up", "navigation", "Scrolls the focused window up.", _scroll(3)
    ),
    BridgeAction(
        "scroll_down",
        "Scroll down",
        "navigation",
        "Scrolls the focused window down.",
        _scroll(-3),
    ),
    BridgeAction(
        "switch_window",
        "Switch window",
        "navigation",
        "Alt-Tab to the previously focused window.",
        _chord(Key.alt, Key.tab),
    ),
    BridgeAction(
        "click_left",
        "Click",
        "pointer",
        "Left mouse click at the current cursor position.",
        _click(Button.left),
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
        action.run()
    except Exception as exc:
        logger.error(f"Action {action_id} failed: {exc}")
        return False
    return True
