import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()
GESTURE_DIR = "checkpoints/custom_gestures/"
os.makedirs(GESTURE_DIR, exist_ok=True)

@router.post("/gestures/create")
def create_gesture(name: str, data: Any) -> dict:
    try:
        gesture_path = os.path.join(GESTURE_DIR, f"{name}.json")
        with open(gesture_path, "w") as f:
            json.dump(data, f)
        return {"status": "created", "name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/gestures/list")
def list_gestures() -> list[str]:
    return [f[:-5] for f in os.listdir(GESTURE_DIR) if f.endswith(".json")]

@router.delete("/gestures/delete/{name}")
def delete_gesture(name: str) -> dict:
    gesture_path = os.path.join(GESTURE_DIR, f"{name}.json")
    if not os.path.exists(gesture_path):
        raise HTTPException(status_code=404, detail="Gesture not found.")
    os.remove(gesture_path)
    return {"status": "deleted", "name": name}
