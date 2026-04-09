from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.services.ingestion import process_and_store_document
from fastapi import BackgroundTasks

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Form(...)  # Using Form since UploadFile requires multipart/form-data
):
    """
    Uploads a proprietary document (.pdf or .txt), chunks it, and saves it to the vector database.
    Requires a tenant_id to isolate the data.
    """
    try:
        num_chunks = await process_and_store_document(file, tenant_id, background_tasks=BackgroundTasks())
        return {
            "status": "success",
            "message": f"Successfully ingested {file.filename}",
            "chunks_created": num_chunks,
            "tenant_id": tenant_id
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")