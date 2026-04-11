from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from backend.services.ingestion import process_and_store_document
from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from backend.services.models import User
from backend.services.authutil import get_current_user, get_db
from backend.services import crud

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user), # Securely extracts the user from the JWT
    db: Session = Depends(get_db)                   # Injects the PostgreSQL session
):
    """
    Uploads a proprietary document, logs it in Postgres, and saves the chunks to ChromaDB.
    """
    try:
        # 1. Process the document using the secure user ID (ChromaDB & BM25)
        num_chunks = await process_and_store_document(file, current_user.id, background_tasks) # type: ignore
        
        # 2. Log the successful upload in PostgreSQL
        crud.create_document_record(db=db, filename=file.filename, tenant_id=current_user.id) # type: ignore
        
        return {
            "status": "success",
            "message": f"Successfully ingested {file.filename}. BM25 index rebuilding in background.",
            "chunks_created": num_chunks,
            "tenant_id": current_user.id
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/my-files")
def list_my_documents(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Returns a list of all files uploaded by the authenticated user."""
    docs = crud.get_user_documents(db, tenant_id=current_user.id) # type: ignore
    return {"documents": docs}