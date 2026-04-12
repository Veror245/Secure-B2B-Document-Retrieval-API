from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.extension import _rate_limit_exceeded_handler
from backend.routers import documents
from backend.routers import query, auth
from backend.services.database import engine, Base
from backend.services.limiter import limiter

app = FastAPI(title="Secure B2B Document Retrieval API")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler) # type: ignore

Base.metadata.create_all(bind=engine)

app.include_router(documents.router)
app.include_router(auth.router)
app.include_router(query.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

