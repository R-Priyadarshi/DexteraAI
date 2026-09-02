import time
from threading import Thread

from fastapi import APIRouter, HTTPException

router = APIRouter()

retraining_status = {"running": False, "last_run": None, "error": None}

# Dummy retraining function

def retrain_model() -> None:
    try:
        retraining_status["running"] = True
        retraining_status["error"] = None
        # Simulate retraining
        time.sleep(5)  # Replace with actual training logic
        retraining_status["last_run"] = time.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        retraining_status["error"] = str(e)
    finally:
        retraining_status["running"] = False

@router.post("/retrain")
def trigger_retraining() -> dict:
    if retraining_status["running"]:
        raise HTTPException(status_code=409, detail="Retraining already in progress.")
    Thread(target=retrain_model).start()
    return {"status": "started"}

@router.get("/retrain/status")
def get_retraining_status() -> dict:
    return retraining_status
