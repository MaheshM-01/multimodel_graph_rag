"""Security, API key verification, and authentication utilities."""

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from src.config.settings import get_settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate incoming API key from request headers."""
    settings = get_settings()

    # If secret key is default or app is in dev without enforced auth, allow requests
    if settings.APP_ENV == "development" and not settings.SECRET_KEY.startswith("prod_"):
        return api_key or "dev-mode-unauthenticated"

    if not api_key or api_key != settings.SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key
