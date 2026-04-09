from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.retriever import query_documents

router = APIRouter(prefix="/query", tags=["Query"])

# Define the Pydantic schema for the JSON request body
class QueryRequest(BaseModel):
    query: str
    tenant_id: str

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
async def ask_question(request: QueryRequest):
    """
    Queries the vector database based on the provided tenant_id and generates a structured markdown answer.
    """
    try:
        result = await query_documents(request.query, request.tenant_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval Error: {str(e)}")