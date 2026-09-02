from datetime import datetime

from fastapi import APIRouter, HTTPException

router = APIRouter()

notifications = []

@router.post("/notify")
def send_notification(message: str, channel: str = "dashboard") -> dict:
    try:
        notif = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "channel": channel,
        }
        notifications.append(notif)
        # Here, add logic for email/webhook/etc.
        return {"status": "sent", "notification": notif}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/notify/list")
def list_notifications() -> list[dict]:
    return notifications
