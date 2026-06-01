from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
import models, schemas

router = APIRouter(prefix="/api/barcode", tags=["barcode"])


@router.get("/{barcode_value}", response_model=schemas.BarcodeCheck)
def check_barcode(barcode_value: str, db: Session = Depends(get_db)):
    bc = db.query(models.Barcode).filter(
        models.Barcode.barcode_value == barcode_value
    ).first()

    if not bc:
        return schemas.BarcodeCheck(found=False, barcode_value=barcode_value)

    has_model = bool(
        bc.yolo_model and
        bc.yolo_model.status == "active" and
        bc.yolo_model.model_path
    )

    return schemas.BarcodeCheck(
        found=True,
        barcode_value=barcode_value,
        name=bc.name,
        has_model=has_model,
        model_id=bc.yolo_model_id if has_model else None,
    )
