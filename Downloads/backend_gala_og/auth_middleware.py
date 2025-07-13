from fastapi import Request, HTTPException, Depends, Cookie
from typing import Optional, Dict
from auth import verify_jwt_token
from redis_session import SessionManager
import logging

logger = logging.getLogger(__name__)

async def get_current_user(form_auth_token: Optional[str] = Cookie(None)) -> Dict:
    """Extract and verify the user from JWT token
    
    Args:
        auth_token: JWT token from cookie
        
    Returns:
        Dict: User data from JWT
        
    Raises:
        HTTPException: If token is missing or invalid
    """
    # Debug logging to help diagnose authentication issues
    logger.info(f"🔍 Authentication check - Cookie received: {'YES' if form_auth_token else 'NO'}")
    if form_auth_token:
        logger.info(f"🔍 Cookie value length: {len(form_auth_token)} characters")
        logger.info(f"🔍 Cookie starts with: {form_auth_token[:20]}...")
        print(form_auth_token)
    
    if not form_auth_token:
        logger.warning("🚨 Authentication failed: No form_auth_token cookie found")
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user_data = verify_jwt_token(form_auth_token)
    if not user_data:
        logger.warning(f"🚨 Authentication failed: Invalid or expired token (length: {len(form_auth_token)})")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    logger.info(f"✅ Authentication successful for user: {user_data.get('sub', 'unknown')}")
    
    try:
        # Update user's activity timestamp in Redis
        session_mgr = await SessionManager.get_instance()
        await session_mgr.update_activity(user_data["sub"])
    except Exception as e:
        # Log error but don't fail authentication if Redis is unavailable
        logger.error(f"Failed to update user activity: {str(e)}")
    
    return user_data

async def get_current_user_id(user_data: Dict = Depends(get_current_user)) -> str:
    """Extract user ID from user data
    
    Args:
        user_data: User data from JWT token
        
    Returns:
        str: User ID from the JWT sub field
    """
    return user_data["sub"]

# Optional: You can also add a middleware to track all requests
async def session_middleware(request: Request, call_next):
    """Middleware to track user sessions for all requests
    
    This middleware should be added to the FastAPI app:
    app.middleware("http")(session_middleware)
    """
    try:
        # Try to get auth token from cookie
        auth_token = request.cookies.get("auth_token")
        if auth_token:
            # Verify token and update activity
            user_data = verify_jwt_token(auth_token)
            if user_data:
                session_mgr = await SessionManager.get_instance()
                await session_mgr.update_activity(user_data["sub"])
    except Exception as e:
        # Log but continue with request
        logger.error(f"Error in session middleware: {str(e)}")
    
    # Continue with request
    response = await call_next(request)
    return response
