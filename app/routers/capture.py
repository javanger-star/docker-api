import os
import threading
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from PIL import Image
from database import get_db
from services.yolo_service import run_detection, _get_model, TRAY_IMGSZ, PRODUCT_IMGSZ
import models, schemas

router = APIRouter(prefix="/api/capture", tags=["capture"])

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/uploads"))
THUMBNAIL_SIZE = (400, 225)  # 16:9


@router.post("/", response_model=schemas.CapturedImageResponse)
async def capture_image(
    barcode_value: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    bc = db.query(models.Barcode).filter(
        models.Barcode.barcode_value == barcode_value
    ).first()
    if not bc:
        raise HTTPException(status_code=404, detail="バーコードが登録されていません")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_bc = "".join(c if c.isalnum() or c in "-_" else "_" for c in barcode_value)
    filename = f"{safe_bc}_{ts}.jpg"

    save_dir = UPLOAD_DIR / safe_bc
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / filename

    content = await file.read()
    # EXIF回転を画素に焼き込んで保存（OpenCVはEXIFを無視するため必須）
    try:
        from PIL import ImageOps
        import io as _io
        pil = Image.open(_io.BytesIO(content))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        buf = _io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        file_path.write_bytes(buf.getvalue())
    except Exception:
        file_path.write_bytes(content)

    # Thumbnail
    thumb_path = save_dir / f"thumb_{filename}"
    try:
        with Image.open(file_path) as img:
            img.thumbnail(THUMBNAIL_SIZE)
            img.save(thumb_path, "JPEG")
    except Exception:
        thumb_path = None

    layout_path = bc.layout_path or (bc.yolo_model.layout_path if bc.yolo_model else None)
    tray_model_path = bc.tray_model.model_path if bc.tray_model else None

    # YOLO detection（製品モデルが設定されている場合のみ実行）
    detection_result = None
    if bc.yolo_model and bc.yolo_model.status == "active" and bc.yolo_model.model_path:
        try:
            detection_result = run_detection(
                product_model_path=bc.yolo_model.model_path,
                layout_path=layout_path,
                image_path=str(file_path),
                tray_model_path=tray_model_path,
            )
        except Exception as e:
            detection_result = {"error": str(e), "boxes": []}

    img_record = models.CapturedImage(
        barcode_id=bc.id,
        filename=filename,
        file_path=str(file_path),
        thumbnail_path=str(thumb_path) if thumb_path else None,
        detection_result=detection_result,
    )
    db.add(img_record)
    db.commit()
    db.refresh(img_record)
    return img_record


_prewarm_lock = threading.Lock()
_prewarming: set = set()

@router.get("/prewarm/{barcode_value}", status_code=200)
def prewarm(barcode_value: str, db: Session = Depends(get_db)):
    """バーコードスキャン直後に呼び出し、実解像度でモデルを事前ウォームアップする（非同期）。"""
    bc = db.query(models.Barcode).filter(
        models.Barcode.barcode_value == barcode_value
    ).first()
    if not bc:
        return {"status": "not_found"}

    tray_path   = bc.tray_model.model_path   if bc.tray_model  and bc.tray_model.model_path  else None
    prod_path   = bc.yolo_model.model_path   if bc.yolo_model  and bc.yolo_model.model_path  else None
    key = f"{tray_path}|{prod_path}"

    with _prewarm_lock:
        if key in _prewarming:
            return {"status": "already_warming"}
        _prewarming.add(key)

    import numpy as np

    def _do():
        try:
            dummy = Image.fromarray(
                np.zeros((720, 1280, 3), dtype=np.uint8)
            )
            if tray_path and Path(tray_path).exists():
                _get_model(tray_path).predict(
                    source=dummy, imgsz=TRAY_IMGSZ, conf=0.5, verbose=False
                )
            if prod_path and Path(prod_path).exists():
                _get_model(prod_path).predict(
                    source=dummy, imgsz=PRODUCT_IMGSZ, conf=0.5, verbose=False
                )
        except Exception:
            pass
        finally:
            with _prewarm_lock:
                _prewarming.discard(key)

    threading.Thread(target=_do, daemon=True, name="prewarm").start()
    return {"status": "warming"}


@router.get("/reference/{barcode_value}")
def get_reference_image(barcode_value: str, db: Session = Depends(get_db)):
    """Configured reference image (or first training image as fallback) shown before shooting."""
    bc = db.query(models.Barcode).filter(
        models.Barcode.barcode_value == barcode_value
    ).first()
    if not bc:
        raise HTTPException(status_code=404, detail="バーコードが登録されていません")

    no_cache = {"Cache-Control": "no-store"}

    if bc.reference_image_path and Path(bc.reference_image_path).exists():
        return FileResponse(bc.reference_image_path, media_type="image/jpeg", headers=no_cache)

    img = (
        db.query(models.TrainingImage)
        .filter(models.TrainingImage.barcode_name == bc.name)
        .order_by(models.TrainingImage.created_at.asc())
        .first()
    )
    if not img or not Path(img.image_path).exists():
        raise HTTPException(status_code=404, detail="参考画像が見つかりません")

    return FileResponse(img.image_path, media_type="image/jpeg", headers=no_cache)


@router.get("/image/{image_id}")
def get_image_file(image_id: int, db: Session = Depends(get_db)):
    img = db.query(models.CapturedImage).filter(models.CapturedImage.id == image_id).first()
    if not img or not Path(img.file_path).exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img.file_path, media_type="image/jpeg")


@router.get("/thumbnail/{image_id}")
def get_thumbnail(image_id: int, db: Session = Depends(get_db)):
    img = db.query(models.CapturedImage).filter(models.CapturedImage.id == image_id).first()
    if not img or not img.thumbnail_path or not Path(img.thumbnail_path).exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(img.thumbnail_path, media_type="image/jpeg")
