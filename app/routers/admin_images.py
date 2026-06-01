from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from database import get_db
from auth import get_current_user
import models, schemas

router = APIRouter(prefix="/api/admin/images", tags=["admin-images"])


@router.get("/", response_model=list[schemas.CapturedImageResponse])
def list_images(
    barcode_id: int = None,
    barcode_value: str = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    q = db.query(models.CapturedImage)
    if barcode_id:
        q = q.filter(models.CapturedImage.barcode_id == barcode_id)
    if barcode_value:
        bc = db.query(models.Barcode).filter(models.Barcode.barcode_value == barcode_value).first()
        if bc:
            q = q.filter(models.CapturedImage.barcode_id == bc.id)
        else:
            return []
    return q.order_by(models.CapturedImage.captured_at.desc()).offset(skip).limit(limit).all()


@router.get("/{image_id}", response_model=schemas.CapturedImageResponse)
def get_image(image_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    img = db.query(models.CapturedImage).filter(models.CapturedImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Not found")
    return img


@router.delete("/{image_id}", status_code=204)
def delete_image(image_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    img = db.query(models.CapturedImage).filter(models.CapturedImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Not found")
    for p in [img.file_path, img.thumbnail_path]:
        if p:
            path = Path(p)
            if path.exists():
                path.unlink(missing_ok=True)
    db.delete(img)
    db.commit()
