import shutil
import os
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from database import get_db
from auth import get_current_user
import models, schemas

router = APIRouter(prefix="/api/admin/models", tags=["admin-models"])
YOLO_DIR = Path(os.getenv("YOLO_MODEL_DIR", "/yolo_models"))


@router.get("/", response_model=list[schemas.YoloModelResponse])
def list_models(db: Session = Depends(get_db), _=Depends(get_current_user)):
    return db.query(models.YoloModel).all()


@router.get("/{model_id}", response_model=schemas.YoloModelResponse)
def get_model(model_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    m = db.query(models.YoloModel).filter(models.YoloModel.id == model_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Not found")
    return m


@router.post("/", response_model=schemas.YoloModelResponse, status_code=201)
async def create_model(
    name: str = Form(...),
    description: str = Form(""),
    status: str = Form("active"),
    model_file: UploadFile = File(None),
    layout_file: UploadFile = File(None),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    m = models.YoloModel(name=name, description=description, status=status)
    db.add(m)
    db.commit()
    db.refresh(m)

    model_dir = YOLO_DIR / str(m.id)
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_file and model_file.filename:
        mp = model_dir / model_file.filename
        mp.write_bytes(await model_file.read())
        m.model_path = str(mp)

    if layout_file and layout_file.filename:
        lp = model_dir / "layout.txt"
        lp.write_bytes(await layout_file.read())
        m.layout_path = str(lp)

    db.commit()
    db.refresh(m)
    return m


@router.put("/{model_id}", response_model=schemas.YoloModelResponse)
async def update_model(
    model_id: int,
    name: str = Form(None),
    description: str = Form(None),
    status: str = Form(None),
    model_file: UploadFile = File(None),
    layout_file: UploadFile = File(None),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    m = db.query(models.YoloModel).filter(models.YoloModel.id == model_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Not found")

    if name is not None:
        m.name = name
    if description is not None:
        m.description = description
    if status is not None:
        m.status = status

    model_dir = YOLO_DIR / str(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_file and model_file.filename:
        mp = model_dir / model_file.filename
        mp.write_bytes(await model_file.read())
        m.model_path = str(mp)

    if layout_file and layout_file.filename:
        lp = model_dir / "layout.txt"
        lp.write_bytes(await layout_file.read())
        m.layout_path = str(lp)

    db.commit()
    db.refresh(m)
    return m


@router.delete("/{model_id}", status_code=204)
def delete_model(model_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    m = db.query(models.YoloModel).filter(models.YoloModel.id == model_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Not found")
    model_dir = YOLO_DIR / str(model_id)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    db.delete(m)
    db.commit()
