from typing import Optional
from fastapi import Request


def _get_user_id(request: Request) -> Optional[str]:
    """Extract user_id from request headers (set by gateway)."""
    return request.headers.get("x-user-id")


def _get_org_id(request: Request) -> Optional[str]:
    """Extract org_id from request headers (set by gateway)."""
    return request.headers.get("x-org-id")
