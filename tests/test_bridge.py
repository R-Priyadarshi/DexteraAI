"""Tests for the local desktop bridge.

The bridge injects keystrokes into the user's session, so the tests that matter
most here are the ones covering what it refuses to do.
"""

from __future__ import annotations

import os
import stat

import pytest

from bridge import actions
from bridge.server import Bridge, load_or_create_token


class TestActionRegistry:
    def test_every_action_is_described(self) -> None:
        described = {a["id"] for a in actions.describe()}
        assert described == set(actions.ACTIONS)

    def test_unknown_action_is_refused(self) -> None:
        # The wire protocol carries ids from this registry and nothing else.
        # Anything unrecognised must be a no-op returning False, not an error
        # that takes the bridge down and not, obviously, an execution path.
        assert actions.run("definitely_not_an_action") is False
        assert actions.run("") is False
        assert actions.run("__import__('os').system('id')") is False

    def test_action_ids_are_plain_identifiers(self) -> None:
        # Ids reach a dict lookup only, but keeping them to a boring shape
        # means a future consumer that interpolates one cannot be surprised.
        for action_id in actions.ACTIONS:
            assert action_id.replace("_", "").isalnum(), action_id

    def test_descriptions_are_user_facing(self) -> None:
        for action in actions.ACTIONS.values():
            assert action.name and action.description
            assert action.category


class TestToken:
    def test_token_file_is_owner_only(self, tmp_path) -> None:
        path = tmp_path / "token"
        token = load_or_create_token(path)

        assert len(token) >= 32
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600, f"expected 0600, got {mode:o}"

    def test_token_is_stable_across_reads(self, tmp_path) -> None:
        path = tmp_path / "token"
        assert load_or_create_token(path) == load_or_create_token(path)

    def test_world_readable_token_is_rejected_not_repaired(self, tmp_path) -> None:
        # A token that was world-readable must be assumed already read.
        # Silently tightening the mode would keep using a secret that isn't one.
        path = tmp_path / "token"
        path.write_text("previously-leaked-token")
        os.chmod(path, 0o644)

        with pytest.raises(PermissionError, match="readable by other users"):
            load_or_create_token(path)

    def test_tokens_differ_between_installs(self, tmp_path) -> None:
        a = load_or_create_token(tmp_path / "a")
        b = load_or_create_token(tmp_path / "b")
        assert a != b


class TestRateLimit:
    def test_burst_is_capped(self) -> None:
        # The segmenter emits at most a few events a second, so a client at
        # this rate is malfunctioning or hostile; injecting input at machine
        # speed would make the desktop unusable.
        bridge = Bridge(token="t", allowed_origins=set())
        allowed = sum(0 if bridge._rate_limited() else 1 for _ in range(50))
        assert allowed <= 10


class TestOriginPolicy:
    class _Request:
        def __init__(self, origin: str | None) -> None:
            self.headers = {} if origin is None else {"Origin": origin}

    class _Connection:
        def __init__(self, origin: str | None) -> None:
            self.request = TestOriginPolicy._Request(origin)

    @pytest.fixture
    def bridge(self) -> Bridge:
        return Bridge(token="t", allowed_origins={"http://localhost:3000"})

    def test_allowed_origin_passes(self, bridge: Bridge) -> None:
        assert bridge._origin_allowed(self._Connection("http://localhost:3000"))

    def test_arbitrary_website_is_refused(self, bridge: Bridge) -> None:
        # Browsers do not apply the same-origin policy to WebSocket
        # connections, so without this check any open tab could drive the
        # bridge. The browser sets Origin and a page cannot forge it.
        assert not bridge._origin_allowed(self._Connection("https://evil.example"))

    def test_near_miss_origins_are_refused(self, bridge: Bridge) -> None:
        for origin in (
            "http://localhost:3000.evil.example",
            "https://localhost:3000",
            "http://localhost:3001",
            "http://notlocalhost:3000",
        ):
            assert not bridge._origin_allowed(self._Connection(origin)), origin

    def test_missing_origin_falls_through_to_the_token(self, bridge: Bridge) -> None:
        # A client with no Origin is not a browser. Native clients set any
        # header they like, so the header proves nothing about them and the
        # token is the control that actually binds them.
        assert bridge._origin_allowed(self._Connection(None))
