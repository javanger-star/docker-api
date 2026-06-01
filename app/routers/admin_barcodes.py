import io
import os
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from PIL import Image as PilImage, ImageOps
from database import get_db
from auth import get_current_user
from services.yolo_service import generate_layout_from_image, generate_layout_from_annotations, generate_layout_from_annotations_direct
import models, schemas


def _save_image_with_exif(path: Path, raw: bytes) -> None:
    """EXIF回転を画素に焼き込んで保存する"""
    try:
        pil = PilImage.open(io.BytesIO(raw))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        path.write_bytes(buf.getvalue())
    except Exception:
        path.write_bytes(raw)

router = APIRouter(prefix="/api/admin/barcodes", tags=["admin-barcodes"])

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/uploads"))


@router.get("/", response_model=list[schemas.BarcodeResponse])
def list_barcodes(skip: int = 0, limit: int = 500, db: Session = Depends(get_db), _=Depends(get_current_user)):
    return db.query(models.Barcode).offset(skip).limit(limit).all()


@router.get("/{barcode_id}", response_model=schemas.BarcodeResponse)
def get_barcode(barcode_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    bc = db.query(models.Barcode).filter(models.Barcode.id == barcode_id).first()
    if not bc:
        raise HTTPException(status_code=404, detail="Not found")
    return bc


@router.post("/", response_model=schemas.BarcodeResponse, status_code=201)
def create_barcode(body: schemas.BarcodeCreate, db: Session = Depends(get_db), _=Depends(get_current_user)):
    if db.query(models.Barcode).filter(models.Barcode.barcode_value == body.barcode_value).first():
        raise HTTPException(status_code=409, detail="バーコードはすでに登録されています")
    bc = models.Barcode(**body.model_dump())
    db.add(bc)
    db.commit()
    db.refresh(bc)
    return bc


@router.put("/{barcode_id}", response_model=schemas.BarcodeResponse)
def update_barcode(barcode_id: int, body: schemas.BarcodeUpdate, db: Session = Depends(get_db), _=Depends(get_current_user)):
    bc = db.query(models.Barcode).filter(models.Barcode.id == barcode_id).first()
    if not bc:
        raise HTTPException(status_code=404, detail="Not found")
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(bc, k, v)
    db.commit()
    db.refresh(bc)
    return bc


@router.delete("/{barcode_id}", status_code=204)
def delete_barcode(barcode_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    bc = db.query(models.Barcode).filter(models.Barcode.id == barcode_id).first()
    if not bc:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(bc)
    db.commit()


@router.post("/{barcode_id}/generate-layout", response_model=schemas.LayoutGenerateResult)
async def generate_layout(
    barcode_id: int,
    ref_image: UploadFile = File(...),
    conf: float = 0.05,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    """
    正常配置の基準写真を1枚アップロードしてlayout.txtを自動生成する。
    YOLOモデルが設定されている必要がある。
    トレイクラスが検出された場合はトレイ相対座標で保存（位置ずれに強い）。
    """
    bc = db.query(models.Barcode).filter(models.Barcode.id == barcode_id).first()
    if not bc:
        raise HTTPException(status_code=404, detail="Not found")
    if not bc.yolo_model or not bc.yolo_model.model_path:
        raise HTTPException(status_code=400, detail="製品検出モデルが設定されていません")

    # 基準画像を一時保存
    ref_dir = UPLOAD_DIR / "_ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path = ref_dir / f"ref_{barcode_id}_{ref_image.filename}"
    _save_image_with_exif(ref_path, await ref_image.read())

    tray_model_path = bc.tray_model.model_path if bc.tray_model else None

    try:
        items, is_tray_relative = generate_layout_from_image(
            product_model_path=bc.yolo_model.model_path,
            image_path=str(ref_path),
            conf=conf,
            tray_model_path=tray_model_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO推論エラー: {e}")

    if not items:
        raise HTTPException(status_code=422, detail="アイテムが検出されませんでした。閾値(conf)を下げるか、モデルを確認してください")

    # layout.txt を保存（座標系ヘッダーを先頭に書き込む）
    layout_dir = UPLOAD_DIR / "_layouts"
    layout_dir.mkdir(parents=True, exist_ok=True)
    layout_path = layout_dir / f"layout_{barcode_id}.txt"
    coord_header = "tray_relative" if is_tray_relative else "image_relative"
    with open(layout_path, "w") as f:
        f.write(f"# coord: {coord_header}\n")
        for item in items:
            f.write(f"{item['class_name']} {item['cx']:.6f} {item['cy']:.6f} {item['w']:.6f} {item['h']:.6f}\n")

    bc.layout_path = str(layout_path)
    db.commit()

    return schemas.LayoutGenerateResult(
        layout_path=str(layout_path),
        item_count=len(items),
        items=items,
    )


@router.post("/{barcode_id}/generate-layout-from-annotations", response_model=schemas.LayoutGenerateResult)
def generate_layout_from_training_annotations(
    barcode_id: int,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    """アノテーションデータ（学習ラベル）からレイアウトを生成する。
    基準写真のアップロード不要。
    トレイ隅クラス（tray_lu/ru/ld/rd）を使って歪みを補正した
    トレイ相対座標のレイアウトを生成する。
    """
    bc = db.query(models.Barcode).filter(models.Barcode.id == barcode_id).first()
    if not bc:
        raise HTTPException(status_code=404, detail="Not found")

    training_classes = (
        db.query(models.TrainingClass)
        .filter(models.TrainingClass.barcode_name == bc.name)
        .order_by(models.TrainingClass.class_index)
        .all()
    )
    if not training_classes:
        raise HTTPException(status_code=400,
                            detail="学習クラスが登録されていません。先にモデル学習を完了してください。")

    training_images = (
        db.query(models.TrainingImage)
        .filter(
            models.TrainingImage.barcode_name == bc.name,
            models.TrainingImage.label_path.isnot(None),
        )
        .all()
    )
    if not training_images:
        raise HTTPException(status_code=400,
                            detail="アノテーション済みの学習画像がありません。先にアノテーションを作成してください。")

    pairs = [(img.image_path, img.label_path) for img in training_images]
    tc_dicts = [{"class_index": c.class_index, "name": c.name} for c in training_classes]

    try:
        items = generate_layout_from_annotations(pairs, tc_dicts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"レイアウト生成エラー: {e}")

    if not items:
        raise HTTPException(
            status_code=422,
            detail="アノテーションからレイアウトを生成できませんでした。"
                   "アノテーションファイルにトレイ隅クラス（tray_lu/ru/ld/rd）と"
                   "製品クラスが含まれているか確認してください。",
        )

    layout_dir = UPLOAD_DIR / "_layouts"
    layout_dir.mkdir(parents=True, exist_ok=True)
    layout_path = layout_dir / f"layout_{barcode_id}.txt"
    with open(layout_path, "w") as f:
        f.write("# coord: tray_relative\n")
        for item in items:
            f.write(f"{item['class_name']} {item['cx']:.6f} {item['cy']:.6f}"
                    f" {item['w']:.6f} {item['h']:.6f}\n")

    bc.layout_path = str(layout_path)
    db.commit()

    return schemas.LayoutGenerateResult(
        layout_path=str(layout_path),
        item_count=len(items),
        items=items,
    )


@router.post("/{barcode_id}/generate-layout-direct", response_model=schemas.LayoutGenerateResult)
def generate_layout_direct(
    barcode_id: int,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    """アノテーションから画像相対座標で直接レイアウト生成（トレイモデル不要・固定カメラ向け）。"""
    bc = db.query(models.Barcode).filter(models.Barcode.id == barcode_id).first()
    if not bc:
        raise HTTPException(status_code=404, detail="Not found")

    training_classes = (
        db.query(models.TrainingClass)
        .filter(models.TrainingClass.barcode_name == bc.name)
        .order_by(models.TrainingClass.class_index)
        .all()
    )
    if not training_classes:
        raise HTTPException(status_code=400, detail="学習クラスが登録されていません。")

    training_images = (
        db.query(models.TrainingImage)
        .filter(
            models.TrainingImage.barcode_name == bc.name,
            models.TrainingImage.label_path.isnot(None),
        )
        .all()
    )
    if not training_images:
        raise HTTPException(status_code=400, detail="アノテーション済みの学習画像がありません。")

    pairs = [(img.image_path, img.label_path) for img in training_images]
    tc_dicts = [{"class_index": c.class_index, "name": c.name} for c in training_classes]

    try:
        items = generate_layout_from_annotations_direct(pairs, tc_dicts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"レイアウト生成エラー: {e}")

    if not items:
        raise HTTPException(status_code=422, detail="アノテーションから製品位置を取得できませんでした。")

    layout_dir = UPLOAD_DIR / "_layouts"
    layout_dir.mkdir(parents=True, exist_ok=True)
    layout_path = layout_dir / f"layout_{barcode_id}.txt"
    with open(layout_path, "w") as f:
        f.write("# coord: image_relative\n")
        for item in items:
            f.write(f"{item['class_name']} {item['cx']:.6f} {item['cy']:.6f}"
                    f" {item['w']:.6f} {item['h']:.6f}\n")

    bc.layout_path = str(layout_path)
    db.commit()

    return schemas.LayoutGenerateResult(
        layout_path=str(layout_path),
        item_count=len(items),
        items=items,
    )


@router.delete("/{barcode_id}/layout", status_code=204)
def delete_layout(barcode_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    bc = db.query(models.Barcode).filter(models.Barcode.id == barcode_id).first()
    if not bc:
        raise HTTPException(status_code=404, detail="Not found")
    if bc.layout_path:
        p = Path(bc.layout_path)
        p.unlink(missing_ok=True)
    bc.layout_path = None
    db.commit()


@router.post("/{barcode_id}/reference-image", status_code=200)
async def set_reference_image(
    barcode_id: int,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    """撮影前確認画面に表示するサンプル画像を登録する。"""
    bc = db.query(models.Barcode).filter(models.Barcode.id == barcode_id).first()
    if not bc:
        raise HTTPException(status_code=404, detail="Not found")

    ref_dir = UPLOAD_DIR / "_ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path = ref_dir / f"reference_{barcode_id}.jpg"
    _save_image_with_exif(ref_path, await image.read())

    bc.reference_image_path = str(ref_path)
    db.commit()
    return {"reference_image_path": str(ref_path)}


@router.delete("/{barcode_id}/reference-image", status_code=204)
def delete_reference_image(barcode_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    bc = db.query(models.Barcode).filter(models.Barcode.id == barcode_id).first()
    if not bc:
        raise HTTPException(status_code=404, detail="Not found")
    if bc.reference_image_path:
        Path(bc.reference_image_path).unlink(missing_ok=True)
    bc.reference_image_path = None
    db.commit()


