import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

router = APIRouter()

SYNC_DIR = "checkpoints/edge_sync/"
os.makedirs(SYNC_DIR, exist_ok=True)

@router.post("/sync/upload")
def upload_model(file: UploadFile = None) -> dict:
    if file is None:
        file = File(...)
    try:
        file_path = os.path.join(SYNC_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return {"status": "uploaded", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/sync/download/{filename}")
def download_model(filename: str) -> FileResponse:
    file_path = os.path.join(SYNC_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)

@router.get("/sync/list")
def list_synced_models() -> list[str]:
    return [f for f in os.listdir(SYNC_DIR) if os.path.isfile(os.path.join(SYNC_DIR, f))]
