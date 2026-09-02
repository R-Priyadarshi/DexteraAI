import time

from fastapi import APIRouter

router = APIRouter()

metrics = {
    "active_users": 5,
    "gestures_recognized": 1200,
    "errors": 2,
    "uptime": time.time(),
}

logs = [
    {"timestamp": time.time(), "event": "System started"},
    {"timestamp": time.time(), "event": "Gesture recognized: wave"},
]

@router.get("/analytics/metrics")
def get_metrics() -> dict:
    return metrics

@router.get("/analytics/logs")
def get_logs() -> list[dict]:
    return logs
