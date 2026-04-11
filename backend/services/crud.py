from sqlalchemy.orm import Session
from backend.services.models import Document

def create_document_record(db: Session, filename: str, tenant_id: str):
    """Logs the uploaded document into PostgreSQL and links it to the user."""
    db_document = Document(
        filename=filename, 
        tenant_id=tenant_id, 
        status="processed"
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def get_user_documents(db: Session, tenant_id: str):
    """Retrieves all documents owned by a specific user."""
    return db.query(Document).filter(Document.tenant_id == tenant_id).all()