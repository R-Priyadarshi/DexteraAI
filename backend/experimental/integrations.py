import requests
from fastapi import APIRouter, HTTPException

router = APIRouter()

integrations = {}

@router.post("/integrations/register")
def register_integration(name: str, url: str) -> dict:
    integrations[name] = url
    return {"status": "registered", "name": name, "url": url}

@router.post("/integrations/trigger/{name}")
def trigger_integration(name: str, payload: dict) -> dict:
    if name not in integrations:
        raise HTTPException(status_code=404, detail="Integration not found.")
    try:
        resp = requests.post(integrations[name], json=payload)
        return {"status": "triggered", "response": resp.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/integrations/list")
def list_integrations() -> dict:
    return integrations
