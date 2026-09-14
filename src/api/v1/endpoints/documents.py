"""Endpoints for listing, previewing, and managing ingested Knowledge Base documents."""

import mimetypes
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.core.logging import logger
from src.worker import JOB_REGISTRY

router = APIRouter()

STORAGE_DIR = Path("./data/media_store")


class DocumentMeta(BaseModel):
    filename: str
    size_bytes: int
    size_human: str
    extension: str
    mime_type: str
    modified_at: float
    status: str
    view_url: str


def format_bytes(size: int) -> str:
    size_float = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_float < 1024.0:
            return f"{size_float:.1f} {unit}" if unit != "B" else f"{int(size_float)} B"
        size_float /= 1024.0
    return f"{size_float:.1f} TB"


@router.get("", response_model=list[DocumentMeta], tags=["Documents"])
async def list_documents() -> list[DocumentMeta]:
    """List all documents currently available in the Knowledge Base media store."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    docs: list[DocumentMeta] = []

    for file_path in sorted(STORAGE_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if file_path.is_file() and not file_path.name.startswith("."):
            st = file_path.stat()
            mime, _ = mimetypes.guess_type(file_path.name)
            mime_type = mime or "application/octet-stream"

            doc_status = "Indexed"
            for job in JOB_REGISTRY.values():
                if job.filename == file_path.name:
                    doc_status = "Completed" if job.status == "completed" else str(job.status).capitalize()
                    break

            docs.append(
                DocumentMeta(
                    filename=file_path.name,
                    size_bytes=st.st_size,
                    size_human=format_bytes(st.st_size),
                    extension=file_path.suffix.lower(),
                    mime_type=mime_type,
                    modified_at=st.st_mtime,
                    status=doc_status,
                    view_url=f"/api/v1/documents/{file_path.name}/view",
                )
            )

    return docs


@router.get("/{filename}/view", tags=["Documents"])
async def view_document(filename: str):
    """View/preview or stream a document from the Knowledge Base."""
    file_path = STORAGE_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found in Knowledge Base.",
        )

    mime, _ = mimetypes.guess_type(filename)
    return FileResponse(
        path=str(file_path),
        media_type=mime or "application/octet-stream",
        filename=filename,
        content_disposition_type="inline",
    )


@router.get("/{filename}/pages/{page_num}/preview", tags=["Documents"])
async def preview_document_page(filename: str, page_num: int):
    """Render and serve a specific page of a PDF or image document as a high-resolution PNG image."""
    from fastapi.responses import Response
    file_path = STORAGE_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found in Knowledge Base.",
        )

    # If it's an image file, return the image directly
    if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]:
        mime, _ = mimetypes.guess_type(filename)
        return FileResponse(path=str(file_path), media_type=mime or "image/png")

    # If it's a PDF, render the page using fitz (PyMuPDF)
    try:
        import fitz
        safe_stem = re.sub(r"[^\w\-]", "_", file_path.stem.strip()) or "doc"
        cache_dir = STORAGE_DIR / ".cache" / "pages" / safe_stem
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_img = cache_dir / f"page_{page_num}.png"

        if cached_img.exists() and cached_img.stat().st_size > 0:
            return FileResponse(path=str(cached_img), media_type="image/png")

        doc = fitz.open(str(file_path))
        if page_num < 1 or page_num > len(doc):
            doc.close()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid page number {page_num}. Document has {len(doc)} pages.",
            )

        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=130)
        pix.save(str(cached_img))
        doc.close()

        return FileResponse(path=str(cached_img), media_type="image/png")
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PyMuPDF (fitz) is required for PDF page rendering.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering page {page_num} for {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not render page: {str(e)}",
        )



@router.delete("/{filename}", tags=["Documents"])
async def delete_document(filename: str):
    """Delete a document from the Knowledge Base media store."""
    file_path = STORAGE_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found in Knowledge Base.",
        )

    try:
        file_path.unlink()
        logger.info(f"Deleted document from Knowledge Base: {filename}")

        # Remove from in-memory job registry if present
        to_delete_jobs = [jid for jid, job in JOB_REGISTRY.items() if job.filename == filename]
        for jid in to_delete_jobs:
            JOB_REGISTRY.pop(jid, None)

        return {
            "success": True,
            "message": f"Document '{filename}' successfully removed from Knowledge Base.",
        }
    except Exception as e:
        logger.error(f"Failed to delete document {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}",
        )
