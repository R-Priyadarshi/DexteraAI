from typing import Any


class MultimodalInput:
    """Unified representation for gesture, voice, and text inputs."""

    def __init__(
        self,
        gesture: Any | None = None,
        voice: Any | None = None,
        text: str | None = None,
        sensor: dict[str, Any] | None = None,
    ) -> None:
        self.gesture = gesture
        self.voice = voice
        self.text = text
        self.sensor = sensor or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "gesture": self.gesture,
            "voice": self.voice,
            "text": self.text,
            "sensor": self.sensor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultimodalInput":
        return cls(
            gesture=data.get("gesture"),
            voice=data.get("voice"),
            text=data.get("text"),
            sensor=data.get("sensor", {}),
        )
