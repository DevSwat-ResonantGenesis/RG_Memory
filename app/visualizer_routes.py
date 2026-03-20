from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_session

router = APIRouter(prefix="/memory/visualizer", tags=["visualizer"])

STATIC_DIR = Path(__file__).parent / "static"



@router.get("/semantic-space", response_class=HTMLResponse)
async def get_semantic_space_visualizer(
    session: AsyncSession = Depends(get_session),
):
    """
    Serve the semantic space visualizer HTML.
    Requires authentication - user must be logged in to access their memory visualization.
    """
    visualizer_path = STATIC_DIR / "semantic_space_visualizer.html"
    
    if not visualizer_path.exists():
        raise HTTPException(status_code=404, detail="Visualizer not found")
    
    with open(visualizer_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)


@router.get("/memory-manager", response_class=HTMLResponse)
async def get_memory_manager(
    session: AsyncSession = Depends(get_session),
):
    """
    Serve the memory manager HTML.
    Requires authentication - user must be logged in to manage their memories.
    """
    manager_path = STATIC_DIR / "memory_manager.html"
    
    if not manager_path.exists():
        raise HTTPException(status_code=404, detail="Memory manager not found")
    
    with open(manager_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)


@router.get("/hash-sphere")
async def get_hash_sphere_visualizer():
    """
    Serve the Memory Visualizer Pro HTML (Hash Sphere).
    This is intended to be embedded in the frontend via iframe (same pattern as code-visualizer).
    """
    visualizer_path = STATIC_DIR / "memory_visualizer_pro.html"

    if not visualizer_path.exists():
        raise HTTPException(status_code=404, detail="Visualizer not found")

    with open(visualizer_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)


@router.get("/health")
async def visualizer_health():
    """Health check for visualizer routes."""
    return {"status": "ok", "service": "memory-visualizer"}
