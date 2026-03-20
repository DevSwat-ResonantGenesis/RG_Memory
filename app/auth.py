"""
Authentication dependencies for Memory Service
"""
import os
import secrets
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
from jose import jwt, JWTError

# Simple JWT validation (in production, this should be centralized)
security = HTTPBearer()

# JWT Secret - use same logic as auth service
def _get_jwt_secret():
    """Get JWT secret from environment or generate default"""
    secret = os.getenv("AUTH_JWT_SECRET_KEY")
    if not secret:
        # Use same default pattern as auth service
        secret = "dev-secret-change-me-" + secrets.token_hex(8)
    return secret

JWT_SECRET_KEY = _get_jwt_secret()
ALGORITHM = "HS256"

# Debug: Log the JWT secret
print(f"[DEBUG] Memory Service JWT Secret: {JWT_SECRET_KEY[:20]}...")

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    Validate JWT token and extract user identity.
    
    This should be replaced with proper auth service integration.
    """
    print(f"[DEBUG] Auth: Checking credentials...")
    
    if not credentials:
        print(f"[DEBUG] Auth: No credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        print(f"[DEBUG] Auth: Decoding token: {credentials.credentials[:20]}...")
        # Decode JWT token
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        print(f"[DEBUG] Auth: Token decoded successfully: {payload}")
        
        # Extract user_id from token
        user_id = payload.get("user_id")
        org_id = payload.get("org_id")
        
        if not user_id:
            print(f"[DEBUG] Auth: No user_id in token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user_id"
            )
        
        result = {
            "user_id": user_id,
            "org_id": org_id,
            "role": payload.get("role", "user"),
            "scopes": payload.get("scopes", [])
        }
        print(f"[DEBUG] Auth: Returning user: {result}")
        return result
        
    except JWTError as e:
        print(f"[DEBUG] Auth: JWT Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))):
    """
    Optional authentication - returns None if no token provided.
    """
    if not credentials:
        return None
    
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "user_id": payload.get("user_id"),
            "org_id": payload.get("org_id"),
            "role": payload.get("role", "user"),
            "scopes": payload.get("scopes", [])
        }
    except JWTError:
        return None
