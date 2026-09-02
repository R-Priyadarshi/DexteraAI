from fastapi import APIRouter, HTTPException, Request

from core.types_multimodal import MultimodalInput

router = APIRouter()

@router.post("/multimodal/ingest")
async def ingest_multimodal(request: Request) -> dict:
    try:
        data = await request.json()
        multimodal = MultimodalInput.from_dict(data)
        # Here, add processing, logging, or routing logic
        return {"status": "received", "data": multimodal.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
