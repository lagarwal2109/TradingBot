from fastapi import APIRouter, HTTPException, Cookie
from typing import Optional
from auth import verify_jwt_token

# Create router for user routes
user_router = APIRouter(prefix="/user", tags=["User"])

@user_router.get("/profile")
async def get_user_profile(auth_token: Optional[str] = Cookie(None)):
    """
    Get the user profile information
    """
    if not auth_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Verify JWT token
    user_data = verify_jwt_token(auth_token)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return {
        "user": {
            "id": user_data["sub"],
            "email": user_data["email"],
            "name": user_data.get("name", "")
            # picture field removed as requested
        }
    }
