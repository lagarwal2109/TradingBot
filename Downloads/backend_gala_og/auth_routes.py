from fastapi import APIRouter, Response, HTTPException, Cookie
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from auth import verify_google_token, create_jwt_token, verify_jwt_token

# Models
class GoogleTokenRequest(BaseModel):
    token: str = None
    credential: str = None
    clientId: str = None
    select_by: str = None

# Create router for authentication routes
router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/google")
async def google_login(request: GoogleTokenRequest, response: Response):
    """
    Endpoint to verify Google token and create session
    """
    try:
        # Debug log the entire request
        print(f"Google login request received: {request}")
        
        # Extract token from either field
        token = None
        if request.token:
            token = request.token
        elif request.credential:
            token = request.credential
        
        if not token:
            print("No token found in request")
            print(f"Request fields: {request.dict()}")
            raise ValueError("No token provided in request")
            
        print(f"Using token: {token[:20]}...")
        
        # Verify Google token
        user_data = verify_google_token(token)
        
        # Create JWT token
        token = create_jwt_token(user_data)
        
        # Set token as HTTP-only cookie
        # Determine if we're in production or development
        is_prod = True  # Change to True in production
        
        cookie_options = {
            "key": "form_auth_token",  # Changed to unique name to avoid conflicts
            "value": token,
            "httponly": True,      # Prevent JavaScript access
            "max_age": 604800,     # 7 days in seconds
            "path": "/"           # Use root path to ensure cookie is sent with all requests
        }
        
        # Add production-specific settings
        if is_prod:
            cookie_options["domain"] = ".mayacode.io"  # Allow cookie for all subdomains
            cookie_options["secure"] = True            # HTTPS only
            cookie_options["samesite"] = "none"        # Allow cross-site requests
        
        print(f"🍪 Setting cookie with options: {cookie_options}")
        print(f"🍪 Cookie name: {cookie_options['key']}")
        print(f"🍪 Cookie value length: {len(cookie_options['value'])} characters")
        print(f"🍪 Cookie value starts with: {cookie_options['value'][:20]}...")
        
        response.set_cookie(**cookie_options)
        
        print("✅ Cookie set successfully")
        return {"success": True, "message": "Authentication successful"}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")

@router.get("/session")
async def check_session(form_auth_token: Optional[str] = Cookie(None)):
    """
    Endpoint to check if user is authenticated
    """
    print(f"🔍 SESSION CHECK: Cookie received: {'YES' if form_auth_token else 'NO'}")
    if form_auth_token:
        print(f"🔍 SESSION CHECK: Cookie length: {len(form_auth_token)}")
    
    if not form_auth_token:
        print("❌ SESSION CHECK: No cookie - not authenticated")
        return {"authenticated": False}
    
    # Verify JWT token
    user_data = verify_jwt_token(form_auth_token)
    if not user_data:
        print("❌ SESSION CHECK: Invalid token - not authenticated")
        return {"authenticated": False}
    
    print(f"✅ SESSION CHECK: Authenticated user: {user_data.get('email')}")
    return {
        "authenticated": True,
        "user": {
            "sub": user_data["sub"],
            "email": user_data["email"],
            "name": user_data.get("name")
            # picture field removed as requested
        }
    }

@router.post("/logout")
async def logout(response: Response):
    """
    Endpoint to log out user by clearing cookie
    """
    # Clear the auth token cookie
    # Determine if we're in production or development
    is_prod = True  # Change to True in production
    
    cookie_options = {
        "key": "form_auth_token",  # Changed to match the login cookie name
        "value": "",
        "httponly": True,
        "max_age": 0,  # Expire immediately
        "path": "/"     # Back to root path
    }
    
    # Add production-specific settings
    if is_prod:
        cookie_options["domain"] = ".mayacode.io"
        cookie_options["secure"] = True
        cookie_options["samesite"] = "none"
    
    response.set_cookie(**cookie_options)
    
    return {"success": True, "message": "Logged out successfully"}
