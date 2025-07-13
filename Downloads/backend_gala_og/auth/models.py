from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

class TokenPayload(BaseModel):
    """JWT token payload structure"""
    sub: str  # User ID
    name: Optional[str] = None
    email: EmailStr
    picture: Optional[str] = None
    exp: Optional[int] = None  # Expiration time
    iat: Optional[int] = None  # Issued at

class GoogleTokenRequest(BaseModel):
    """Request model for Google token validation"""
    token: str  # Google ID token

class User(BaseModel):
    """User model for database storage"""
    id: str  # Google user ID
    email: EmailStr
    name: Optional[str] = None
    picture: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class UserResponse(BaseModel):
    """User response model (excludes sensitive data)"""
    id: str
    email: EmailStr
    name: Optional[str] = None
    picture: Optional[str] = None
