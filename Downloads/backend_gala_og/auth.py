import jwt
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
from google.oauth2 import id_token
from google.auth.transport import requests

# Constants
GOOGLE_CLIENT_ID = "1024955758405-mhm1m90dlo89mfer4ecdamrgop7k3250.apps.googleusercontent.com"
JWT_SECRET = "d8e6f7a8c1b2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8"  # Should be stored in env variables
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 1 * 12 * 60 * 60  # 7 days in seconds

def verify_google_token(token: str) -> Dict:
    """Verify Google OAuth token and extract user information"""
    try:
        print(f"Attempting to verify token: {token[:20]}...")
        
        # Verify the token
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), GOOGLE_CLIENT_ID)
            
        print(f"Token verified successfully, info: {idinfo.keys()}")

        # Check that the token is valid
        if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Invalid issuer.')

        # Return user information - handle possible missing fields
        user_data = {
            'sub': idinfo['sub'],  # Google user ID
            'email': idinfo.get('email', f"{idinfo['sub']}@unknown.com"),
            'name': idinfo.get('name', ''),
            'picture': idinfo.get('picture', '')
        }
        
        print(f"Successfully extracted user data: {user_data}")
        return user_data
    except Exception as e:
        # Token verification failed
        print(f"Token verification error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Invalid token: {str(e)}")

def create_jwt_token(user_data: Dict) -> str:
    """Create a JWT token for the user"""
    payload = {
        'sub': user_data['sub'],
        'email': user_data['email'],
        'name': user_data.get('name', ''),
        'picture': user_data.get('picture', ''),
        'exp': datetime.utcnow() + timedelta(seconds=JWT_EXPIRATION),
        'iat': datetime.utcnow()
    }
    
    # Create the JWT token
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def verify_jwt_token(token: str) -> Optional[Dict]:
    """Verify a JWT token and return user data"""
    try:
        # Decode the JWT token
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        # Check if token is expired
        if payload['exp'] < time.time():
            return None
            
        # Return user data
        return {
            'sub': payload['sub'],
            'email': payload['email'],
            'name': payload.get('name', ''),
        }
    except Exception as e:
        # Token verification failed
        return None
