import os
from datetime import datetime, timedelta, timezone
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from authlib.jose import jwt, errors
from sqlalchemy.orm import Session
from backend.services.database import SessionLocal
from backend.services.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Security Configurations
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-production-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Passlib configuration for bcrypt password hashing
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against the stored bcrypt hash."""
    # bcrypt requires bytes, so we encode the strings
    password_bytes = plain_password.encode('utf-8')
    hash_bytes = hashed_password.encode('utf-8')
    
    return bcrypt.checkpw(password_bytes, hash_bytes)

def get_password_hash(password: str) -> str:
    """Generates a secure bcrypt hash for a new password."""
    password_bytes = password.encode('utf-8')
    
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed_password_bytes = bcrypt.hashpw(password_bytes, salt)
    
    # Return as a string to easily store it in PostgreSQL
    return hashed_password_bytes.decode('utf-8')


def create_access_token(data: dict) -> str:
    """Generates a secure JWT token using Authlib"""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # JWT standard payload
    payload = {
        **to_encode,
        "exp": expire.timestamp(), # Expiration time
        "iat": now.timestamp(),    # Issued at time
    }
    
    # Authlib requires a header and returns bytes, so we decode to string
    header = {"alg": ALGORITHM}
    token = jwt.encode(header, payload, SECRET_KEY)
    
    return token.decode("utf-8")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Validates the Authlib JWT and returns the DB User object"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the token using Authlib
        payload = jwt.decode(token, SECRET_KEY)
        
        # 'sub' is the standard JWT field for the Subject (User ID)
        user_id: str = payload.get("sub") # type: ignore
        if user_id is None:
            raise credentials_exception
            
    except errors.JoseError:
        # Catches expired or tampered tokens
        raise credentials_exception
        
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
        
    return user

class RoleChecker:
    """RBAC Dependency: Checks if the current user has the required roles."""
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Depends(get_current_user)):
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required roles: {self.allowed_roles}"
            )
        return user