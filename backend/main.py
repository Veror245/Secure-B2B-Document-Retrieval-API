from fastapi import FastAPI
from backend.routers import documents
from backend.routers import query

app = FastAPI()
app.include_router(documents.router)
app.include_router(query.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

