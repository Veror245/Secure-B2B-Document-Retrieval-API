from fastapi import FastAPI
from backend.routers import documents
from backend.routers import query, auth
from backend.services.database import engine, Base

app = FastAPI(title="Secure B2B Document Retrieval API")

Base.metadata.create_all(bind=engine)

app.include_router(documents.router)
app.include_router(auth.router)
app.include_router(query.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

