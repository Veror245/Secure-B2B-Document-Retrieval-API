from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from backend.services.retriever import query_documents
from backend.services.authutil import get_current_user
from backend.services.models import User

router = APIRouter(prefix="/query", tags=["Query"])

# Define the Pydantic schema for the JSON request body
class QueryRequest(BaseModel):
    query: str

# Define the Pydantic schemas for the structured JSON response
class Source(BaseModel):
    file: str
    page: int | str
    preview: str
    full_content: str

class QueryResponse(BaseModel):
    answer_markdown: str
    is_relevant: bool
    sources: List[Source]


@router.post("/", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest, 
    current_user: User = Depends(get_current_user) # Securely extracts the user from the JWT
):
    """
    Searches the authenticated user's uploaded documents and generates an answer using RAG.
    """
    try:
        # Pass the secure current_user.id directly into the retrieval engine
        result = await query_documents(request.query, current_user.id) # type: ignore
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")