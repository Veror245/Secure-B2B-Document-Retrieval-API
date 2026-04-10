from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os

# We will read this from Docker environment variables later
# Fallback to SQLite just in case you run it locally without Docker
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

# SQLite needs this check, Postgres doesn't
connect_args = {"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()