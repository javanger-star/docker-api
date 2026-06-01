import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import APIRouter, Body, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
import models


class LabelBox(BaseModel):
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

class LabelSave(BaseModel):
    boxes: List[LabelBox]

router = APIRouter(prefix="/api/admin/training", tags=["training"])
TRAINING_DIR = Path(os.getenv("TRAINING_DIR", "/training_data"))


def _img_dir(barcode_name: Optional[str]) -> Path:
    base = TRAINING_DIR / barcode_name if barcode_name else TRAINING_DIR
    return base / "images"

def _lbl_dir(barcode_name: Optional[str]) -> Path:
    base = TRAINING_DIR / barcode_name if barcode_name else TRAINING_DIR
    return base / "labels"

POINT_EXPORT_SIZE = 0.02  # YOLOエクスポート時に点アノテーションに割り当てるbboxサイズ

def _label_text_for_yolo(lbl_path: Path) -> str:
    """w=0 h=0（点アノテーション）を YOLO 用の小サイズbboxに変換して返す"""
    lines = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            w, h = float(parts[3]), float(parts[4])
            if w == 0.0 and h == 0.0:
                parts[3] = f"{POINT_EXPORT_SIZE:.6f}"
                parts[4] = f"{POINT_EXPORT_SIZE:.6f}"
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n" if lines else ""


# ─── フォルダ（バーコード名）一覧 ──────────────────────────────────

@router.get("/folders")
def list_folders(db: Session = Depends(get_db), _=Depends(get_current_user)):
    """バーコード名ごとの画像数・ラベル付き数を返す"""
    rows = db.query(models.TrainingImage).all()
    folders: dict = {}
    for img in rows:
        key = img.barcode_name or ""
        if key not in folders:
            folders[key] = {"barcode_name": key, "total": 0, "labeled": 0}
        folders[key]["total"] += 1
        if img.label_path and Path(img.label_path).exists():
            folders[key]["labeled"] += 1
    return sorted(folders.values(), key=lambda x: x["barcode_name"])


# ─── クラス管理 ──────────────────────────────────────────────

@router.get("/classes")
def get_classes(folder: Optional[str] = Query(None), db: Session = Depends(get_db), _=Depends(get_current_user)):
    """フォルダ指定時: そのフォルダのクラス一覧。未指定: 空リスト（旧グローバルクラスは返さない）"""
    if not folder:
        return []
    return db.query(models.TrainingClass).filter(
        models.TrainingClass.barcode_name == folder
    ).order_by(models.TrainingClass.class_index).all()


@router.post("/classes", status_code=201)
def add_class(
    name: str = Form(...),
    barcode_name: str = Form(None),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    name = name.strip()
    barcode_name = (barcode_name or "").strip() or None
    if not name:
        raise HTTPException(status_code=400, detail="クラス名を入力してください")
    # 同一フォルダ内での重複チェック
    if db.query(models.TrainingClass).filter(
        models.TrainingClass.name == name,
        models.TrainingClass.barcode_name == barcode_name,
    ).first():
        raise HTTPException(status_code=409, detail="クラス名はすでに存在します")
    # 同一スコープ内での連番
    max_idx = db.query(models.TrainingClass).filter(
        models.TrainingClass.barcode_name == barcode_name
    ).count()
    cls = models.TrainingClass(name=name, class_index=max_idx, barcode_name=barcode_name)
    db.add(cls)
    db.commit()
    db.refresh(cls)
    return cls


@router.delete("/classes/{class_id}", status_code=204)
def delete_class(class_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    cls = db.query(models.TrainingClass).filter(models.TrainingClass.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Not found")
    removed_idx = cls.class_index
    scope_filter = models.TrainingClass.barcode_name == cls.barcode_name
    db.delete(cls)
    db.commit()
    for c in db.query(models.TrainingClass).filter(
        scope_filter, models.TrainingClass.class_index > removed_idx
    ).all():
        c.class_index -= 1
    db.commit()


@router.delete("/classes/cleanup-global", status_code=200)
def cleanup_global_classes(db: Session = Depends(get_db), _=Depends(get_current_user)):
    """旧グローバルクラス（barcode_name IS NULL）を一括削除する"""
    deleted = db.query(models.TrainingClass).filter(
        models.TrainingClass.barcode_name == None
    ).delete()
    db.commit()
    return {"deleted": deleted}


# ─── 学習画像管理 ──────────────────────────────────────────────

@router.get("/images")
def list_images(
    folder: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    q = db.query(models.TrainingImage)
    if folder is not None:
        q = q.filter(models.TrainingImage.barcode_name == (folder or None))
    imgs = q.order_by(models.TrainingImage.created_at.desc()).all()
    return [
        {
            "id": img.id,
            "barcode_name": img.barcode_name or "",
            "filename": img.filename,
            "has_label": img.label_path is not None and Path(img.label_path).exists(),
            "created_at": img.created_at,
        }
        for img in imgs
    ]


@router.post("/images", status_code=201)
async def upload_image(
    image: UploadFile = File(...),
    label: UploadFile = File(None),
    barcode_name: str = Form(None),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    barcode_name = (barcode_name or "").strip() or None
    img_d = _img_dir(barcode_name)
    lbl_d = _lbl_dir(barcode_name)
    img_d.mkdir(parents=True, exist_ok=True)
    lbl_d.mkdir(parents=True, exist_ok=True)

    safe_name = Path(image.filename).name  # strip any directory component
    img_path = img_d / safe_name
    raw = await image.read()
    # EXIF回転を画素に適用してから保存（OpenCVはEXIFを無視するため必須）
    try:
        from PIL import Image as PilImage, ImageOps
        import io
        pil = PilImage.open(io.BytesIO(raw))
        pil = ImageOps.exif_transpose(pil)  # EXIF orientation を画素に反映
        pil = pil.convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=95)
        img_path.write_bytes(buf.getvalue())
    except Exception:
        img_path.write_bytes(raw)  # 失敗時はそのまま保存

    lbl_path = None
    if label and label.filename:
        lbl_path = lbl_d / (Path(image.filename).stem + ".txt")
        lbl_path.write_bytes(await label.read())

    rec = models.TrainingImage(
        barcode_name=barcode_name,
        filename=image.filename,
        image_path=str(img_path),
        label_path=str(lbl_path) if lbl_path else None,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


@router.post("/fix-exif", status_code=200)
def fix_exif_existing_images(db: Session = Depends(get_db), _=Depends(get_current_user)):
    """既存学習画像のEXIF回転を画素に焼き込む（アップロード時にEXIF処理が行われなかった画像の一括修正）"""
    from PIL import Image as PilImage, ImageOps
    rows = db.query(models.TrainingImage).filter(models.TrainingImage.image_path.isnot(None)).all()
    fixed, skipped, errors = 0, 0, []
    for row in rows:
        p = Path(row.image_path)
        if not p.exists():
            skipped += 1
            continue
        try:
            with PilImage.open(p) as img:
                orientation = img.getexif().get(274)  # 274 = Orientation tag
            if not orientation or orientation == 1:
                skipped += 1
                continue
            with PilImage.open(p) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=95)
                p.write_bytes(buf.getvalue())
            fixed += 1
        except Exception as e:
            errors.append({"path": str(p), "error": str(e)})
    return {"fixed": fixed, "skipped": skipped, "errors": errors}


@router.post("/upload-zip", status_code=201)
async def upload_roboflow_zip(
    zip_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    """
    ZIPファイル名をバーコード名・フォルダ名として自動登録する。
    Roboflow の YOLOv8 形式 export ZIP をアップロードして展開する。
    """
    if not zip_file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="ZIPファイルを選択してください")

    barcode_name = Path(zip_file.filename).stem  # 例: "TRAY_001.zip" → "TRAY_001"

    # バーコードを自動登録（なければ新規作成）
    barcode = db.query(models.Barcode).filter(
        models.Barcode.barcode_value == barcode_name
    ).first()
    barcode_created = False
    if not barcode:
        barcode = models.Barcode(barcode_value=barcode_name, name=barcode_name)
        db.add(barcode)
        db.commit()
        db.refresh(barcode)
        barcode_created = True

    content = await zip_file.read()
    img_d = _img_dir(barcode_name)
    lbl_d = _lbl_dir(barcode_name)
    img_d.mkdir(parents=True, exist_ok=True)
    lbl_d.mkdir(parents=True, exist_ok=True)

    added_imgs, class_names = 0, []

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = z.namelist()

        # data.yaml からクラス名を取得
        yaml_files = [n for n in names if n.endswith("data.yaml")]
        if yaml_files:
            raw = z.read(yaml_files[0])
            data_yaml = yaml.safe_load(raw)
            yaml_names = data_yaml.get("names", {})
            if isinstance(yaml_names, dict):
                class_names = [yaml_names[i] for i in sorted(yaml_names)]
            elif isinstance(yaml_names, list):
                class_names = yaml_names

        # 画像・ラベルを展開
        for name in names:
            p = Path(name)
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                data = z.read(name)
                dest = img_d / p.name
                # EXIF回転を画素に焼き込んで保存
                try:
                    from PIL import Image as PilImage, ImageOps
                    pil = PilImage.open(io.BytesIO(data))
                    pil = ImageOps.exif_transpose(pil).convert("RGB")
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=95)
                    dest.write_bytes(buf.getvalue())
                except Exception:
                    dest.write_bytes(data)

                lbl_path_str = None
                for lbl_name in [
                    name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt",
                    str(p.parent.parent / "labels" / p.stem) + ".txt",
                ]:
                    if lbl_name in names:
                        lbl_data = z.read(lbl_name)
                        lbl_file = lbl_d / (p.stem + ".txt")
                        lbl_file.write_bytes(lbl_data)
                        lbl_path_str = str(lbl_file)
                        break
                if lbl_path_str is None:
                    # 負例（アノテーションなし）: 空ラベルファイルを作成
                    lbl_file = lbl_d / (p.stem + ".txt")
                    lbl_file.write_bytes(b"")
                    lbl_path_str = str(lbl_file)

                # DB 登録（同じバーコード内で重複は上書き）
                existing = db.query(models.TrainingImage).filter(
                    models.TrainingImage.barcode_name == barcode_name,
                    models.TrainingImage.filename == p.name,
                ).first()
                if existing:
                    existing.image_path = str(dest)
                    if lbl_path_str:
                        existing.label_path = lbl_path_str
                else:
                    db.add(models.TrainingImage(
                        barcode_name=barcode_name,
                        filename=p.name,
                        image_path=str(dest),
                        label_path=lbl_path_str,
                    ))
                added_imgs += 1

    db.commit()

    # クラス名を DB に同期（ZIPにdata.yamlがある場合）
    if class_names:
        db.query(models.TrainingClass).delete()
        db.commit()
        for idx, cname in enumerate(class_names):
            db.add(models.TrainingClass(name=cname, class_index=idx))
        db.commit()

    return {
        "added": added_imgs,
        "classes": class_names,
        "barcode_name": barcode_name,
        "barcode_created": barcode_created,
    }


@router.delete("/images/{image_id}", status_code=204)
def delete_image(image_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    img = db.query(models.TrainingImage).filter(models.TrainingImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Not found")
    for p in [img.image_path, img.label_path]:
        if p:
            Path(p).unlink(missing_ok=True)
    db.delete(img)
    db.commit()


@router.get("/images/{image_id}/file")
def get_image_file(image_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    img = db.query(models.TrainingImage).filter(models.TrainingImage.id == image_id).first()
    if not img or not img.image_path or not Path(img.image_path).exists():
        raise HTTPException(status_code=404, detail="Not found")
    suffix = Path(img.image_path).suffix.lower()
    media = "image/png" if suffix == ".png" else "image/jpeg"
    return FileResponse(img.image_path, media_type=media)


@router.get("/images/{image_id}/thumbnail")
def get_image_thumbnail(image_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    import io
    from PIL import Image
    img = db.query(models.TrainingImage).filter(models.TrainingImage.id == image_id).first()
    if not img or not img.image_path or not Path(img.image_path).exists():
        raise HTTPException(status_code=404, detail="Not found")
    try:
        with Image.open(img.image_path) as pil:
            pil.thumbnail((160, 90))
            buf = io.BytesIO()
            pil.save(buf, "JPEG", quality=70)
        return Response(content=buf.getvalue(), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})
    except Exception:
        raise HTTPException(status_code=500, detail="Thumbnail generation failed")


@router.get("/images/{image_id}/label")
def get_label(image_id: int, db: Session = Depends(get_db), _=Depends(get_current_user)):
    img = db.query(models.TrainingImage).filter(models.TrainingImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Not found")
    if not img.label_path or not Path(img.label_path).exists():
        return {"boxes": []}
    boxes = []
    for line in Path(img.label_path).read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append({
                "class_id": int(parts[0]),
                "cx": float(parts[1]), "cy": float(parts[2]),
                "w": float(parts[3]),  "h": float(parts[4]),
            })
    return {"boxes": boxes}


@router.post("/images/{image_id}/label")
def save_label(
    image_id: int,
    body: LabelSave,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    img = db.query(models.TrainingImage).filter(models.TrainingImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Not found")
    lbl_d = _lbl_dir(img.barcode_name)
    lbl_d.mkdir(parents=True, exist_ok=True)
    lbl_path = lbl_d / (Path(img.filename).stem + ".txt")
    with open(lbl_path, "w") as f:
        for box in body.boxes:
            # w=0 h=0 は点アノテーション（隅クラス）を示すセンチネル値として保持
            f.write(f"{box.class_id} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}\n")
    img.label_path = str(lbl_path)
    db.commit()
    return {"saved": len(body.boxes)}


@router.delete("/images/{image_id}/label", status_code=204)
def clear_label(
    image_id: int,
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    img = db.query(models.TrainingImage).filter(models.TrainingImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Not found")
    if img.label_path:
        Path(img.label_path).unlink(missing_ok=True)
        img.label_path = None
        db.commit()


@router.get("/stats")
def get_stats(
    folder: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    q = db.query(models.TrainingImage)
    if folder is not None:
        q = q.filter(models.TrainingImage.barcode_name == (folder or None))
    imgs = q.all()
    total = len(imgs)
    labeled = sum(1 for img in imgs if img.label_path and Path(img.label_path).exists())
    if folder:
        classes = db.query(models.TrainingClass).filter(
            models.TrainingClass.barcode_name == folder
        ).order_by(models.TrainingClass.class_index).all()
    else:
        classes = []
    return {"total_images": total, "labeled_images": labeled, "classes": [c.name for c in classes]}


# ─── ダウンロード ──────────────────────────────────────────────

@router.get("/dataset.zip")
def download_dataset(
    folder: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _=Depends(get_current_user),
):
    if folder:
        classes = db.query(models.TrainingClass).filter(
            models.TrainingClass.barcode_name == folder
        ).order_by(models.TrainingClass.class_index).all()
        q = db.query(models.TrainingImage).filter(
            models.TrainingImage.barcode_name == folder
        )
    else:
        classes = []
        q = db.query(models.TrainingImage)
    images = q.all()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img in images:
            # 拡張子を小文字に正規化（.JPG → .jpg）してColabの大文字小文字問題を回避
            p = Path(img.filename)
            normalized_name = p.stem + p.suffix.lower()
            if Path(img.image_path).exists():
                zf.write(img.image_path, f"images/{normalized_name}")
            if img.label_path and Path(img.label_path).exists():
                lbl = Path(img.label_path)
                zf.writestr(f"labels/{p.stem}.txt", _label_text_for_yolo(lbl))
            else:
                # 負例（アノテーションなし）: 空ラベルファイルを書き込んでYOLOに認識させる
                zf.writestr(f"labels/{p.stem}.txt", "")

        data_yaml = {
            "path": "/content/dataset",
            "train": "images",
            "val": "images",
            "names": {c.class_index: c.name for c in classes},
        }
        zf.writestr("data.yaml", yaml.dump(data_yaml, allow_unicode=True))

    fname = f"{folder}_dataset.zip" if folder else "training_dataset.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/notebook.ipynb")
def download_notebook(folder: Optional[str] = Query(None), db: Session = Depends(get_db), _=Depends(get_current_user)):
    if folder:
        classes = db.query(models.TrainingClass).filter(
            models.TrainingClass.barcode_name == folder
        ).order_by(models.TrainingClass.class_index).all()
    else:
        classes = []
    class_names = [c.name for c in classes]
    class_names_py = repr(class_names)

    def cell(source: str, cell_type: str = "code"):
        lines = [line + "\n" for line in source.split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        base = {"cell_type": cell_type, "metadata": {}, "source": lines}
        if cell_type == "code":
            base.update({"outputs": [], "execution_count": None})
        return base

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
        },
        "cells": [
            cell("# 🔍 検品システム YOLOv8 2モデル学習ノートブック\n\n"
                 "**このノートブックは2つのモデルを学習します:**\n"
                 "- 🔲 **tray_model.pt** : トレイ検出モデル（全バーコード共通）\n"
                 "- 📦 **product_model.pt** : 製品検出モデル（このトレイ専用）\n\n"
                 "**使い方:**\n"
                 "1. ランタイム → ランタイムのタイプを変更 → **T4 GPU** を選択\n"
                 "2. 上から順にセルを実行する\n"
                 "3. Step 3 でダウンロードした `training_dataset.zip` をアップロードする\n"
                 "4. 学習完了後、`tray_model_best.pt` と `product_model_best.pt` の2ファイルをダウンロード\n"
                 "5. 管理画面「⚙️ 検品設定」でそれぞれのモデルを登録する",
                 "markdown"),

            cell('# Step 1: 必要ライブラリのインストール\n'
                 '# albumentations は 2.x で API が変わるため 1.x にピン留め\n'
                 '!pip install ultralytics "albumentations>=1.4,<2.0" pyyaml -q\n'
                 'print("✅ インストール完了")'),

            cell('# Step 2: Google Drive をマウント（学習結果の保存先）\n'
                 'from google.colab import drive\n'
                 'drive.mount("/content/drive")\n'
                 'SAVE_DIR = "/content/drive/MyDrive/yolo_inspection"\n'
                 'import os; os.makedirs(SAVE_DIR, exist_ok=True)\n'
                 'print(f"✅ 保存先: {SAVE_DIR}")'),

            cell('# Step 3: training_dataset.zip をアップロード\n'
                 'from google.colab import files\n'
                 'import zipfile, shutil, os\n\n'
                 'print("training_dataset.zip を選択してください...")\n'
                 'uploaded = files.upload()\n\n'
                 'DATASET_DIR = "/content/dataset"\n'
                 'if os.path.exists(DATASET_DIR):\n'
                 '    shutil.rmtree(DATASET_DIR)\n'
                 'os.makedirs(DATASET_DIR)\n\n'
                 'for fn in uploaded:\n'
                 '    with zipfile.ZipFile(fn) as z:\n'
                 '        z.extractall(DATASET_DIR)\n'
                 '    print(f"✅ 解凍完了: {fn}")\n\n'
                 'imgs = [f for f in os.listdir(f"{DATASET_DIR}/images") if f.lower().endswith((".jpg",".jpeg",".png"))]\n'
                 'print(f"📷 画像数: {len(imgs)} 枚")'),

            cell(f'# Step 4: クラス設定確認・トレイクラス特定\n'
                 f'import yaml\n\n'
                 f'CLASS_NAMES = {class_names_py}\n'
                 f'# tray* で始まるクラスをすべてトレイファミリーとして扱う\n'
                 f'TRAY_FAMILY_IDXS = {{i for i, n in enumerate(CLASS_NAMES) if n.lower().startswith("tray")}}\n'
                 f'TRAY_REMAP = {{orig: new for new, orig in enumerate(sorted(TRAY_FAMILY_IDXS))}}\n'
                 f'TRAY_NAMES = {{TRAY_REMAP[i]: CLASS_NAMES[i] for i in sorted(TRAY_FAMILY_IDXS)}}\n'
                 f'PRODUCT_FAMILY_IDXS = {{i for i, n in enumerate(CLASS_NAMES) if not n.lower().startswith("tray")}}\n'
                 f'PRODUCT_ID_MAP = {{orig: new for new, orig in enumerate(sorted(PRODUCT_FAMILY_IDXS))}}\n'
                 f'PRODUCT_NAMES = {{PRODUCT_ID_MAP[i]: CLASS_NAMES[i] for i in sorted(PRODUCT_FAMILY_IDXS)}}\n\n'
                 f'print("クラス一覧:")\n'
                 f'for i, n in enumerate(CLASS_NAMES):\n'
                 f'    tag = " ← トレイ系" if i in TRAY_FAMILY_IDXS else " ← 製品"\n'
                 f'    print(f"  {{i}}: {{n}}{{tag}}")\n'
                 f'print(f"\\n🔲 トレイ系クラス: {{list(TRAY_NAMES.values())}}")\n'
                 f'print(f"📦 製品クラス: {{list(PRODUCT_NAMES.values())}}")'),

            cell('# Step 5: データ拡張（Augmentation）で画像を水増し\n'
                 'import cv2\n'
                 'import albumentations as A\n'
                 'from pathlib import Path\n'
                 'import numpy as np\n'
                 'import shutil, os\n\n'
                 'DATASET_DIR = "/content/dataset"  # Step 3 で定義済み（念のため再定義）\n\n'
                 '# 枚数が少ない場合は 10〜20 で十分。増やすと時間が伸びる\n'
                 'N_AUG = 20\n\n'
                 'aug = A.Compose([\n'
                 '    A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.3, p=0.8),\n'
                 '    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=25, p=0.5),\n'
                 '    A.GaussNoise(var_limit=(5, 40), p=0.6),\n'
                 '    A.GaussianBlur(blur_limit=(3, 5), p=0.4),\n'
                 '    A.Rotate(limit=4, p=0.5),\n'
                 '    A.RandomShadow(p=0.3),\n'
                 '    A.ImageCompression(quality_lower=70, p=0.3),\n'
                 '], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],\n'
                 '                            min_visibility=0.3))\n\n'
                 'img_dir = Path(DATASET_DIR) / "images"\n'
                 'lbl_dir = Path(DATASET_DIR) / "labels"\n'
                 'aug_img_dir = Path(DATASET_DIR) / "aug_images"\n'
                 'aug_lbl_dir = Path(DATASET_DIR) / "aug_labels"\n'
                 'aug_img_dir.mkdir(exist_ok=True)\n'
                 'aug_lbl_dir.mkdir(exist_ok=True)\n\n'
                 'POINT_SIZE = 0.02  # 点アノテーション(w=0,h=0)をYOLO用に変換するサイズ\n\n'
                 'IMG_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]\n'
                 'for f in [f for ext in IMG_EXTS for f in img_dir.glob(ext)]:\n'
                 '    shutil.copy(f, aug_img_dir / f.name)\n'
                 '    lbl = lbl_dir / (f.stem + ".txt")\n'
                 '    if lbl.exists(): shutil.copy(lbl, aug_lbl_dir / lbl.name)\n\n'
                 'orig_files = [f for ext in IMG_EXTS for f in img_dir.glob(ext)]\n'
                 'print(f"📷 元画像: {len(orig_files)} 枚  水増し数: {N_AUG}回/枚  合計予定: {len(orig_files)*N_AUG + len(orig_files)} 枚")\n'
                 'for idx, img_path in enumerate(orig_files):\n'
                 '    print(f"  [{idx+1}/{len(orig_files)}] {img_path.name} を処理中...")\n'
                 '    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)\n'
                 '    lbl_path = lbl_dir / (img_path.stem + ".txt")\n'
                 '    bboxes, cls_ids = [], []\n'
                 '    if lbl_path.exists():\n'
                 '        for line in lbl_path.read_text().strip().splitlines():\n'
                 '            parts = line.split()\n'
                 '            if len(parts) >= 5:\n'
                 '                b = [float(x) for x in parts[1:5]]\n'
                 '                if b[2] == 0.0 and b[3] == 0.0:\n'
                 '                    b[2] = b[3] = POINT_SIZE\n'
                 '                cls_ids.append(int(parts[0]))\n'
                 '                bboxes.append(b)\n'
                 '    for i in range(N_AUG):\n'
                 '        try:\n'
                 '            r = aug(image=img, bboxes=bboxes, class_labels=cls_ids)\n'
                 '            name = f"{img_path.stem}_aug{i:03d}"\n'
                 '            cv2.imwrite(str(aug_img_dir / f"{name}.jpg"),\n'
                 '                        cv2.cvtColor(r["image"], cv2.COLOR_RGB2BGR))\n'
                 '            with open(aug_lbl_dir / f"{name}.txt", "w") as g:\n'
                 '                for c, b in zip(r["class_labels"], r["bboxes"]):\n'
                 '                    g.write(f"{int(c)} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\\n")\n'
                 '        except: pass\n\n'
                 'total = len(list(aug_img_dir.glob("*.jpg")))\n'
                 'print(f"✅ 水増し完了: {total} 枚")'),

            cell('# Step 6: データ分割（トレイ用・製品用）\n'
                 'import yaml, shutil\n'
                 'from pathlib import Path\n'
                 'DATASET_DIR = "/content/dataset"  # Step 3 で定義済み（念のため再定義）\n'
                 'aug_img_dir = Path(DATASET_DIR) / "aug_images"\n'
                 'aug_lbl_dir = Path(DATASET_DIR) / "aug_labels"\n\n'
                 'tray_img_dir = Path("/content/tray_dataset/images")\n'
                 'tray_lbl_dir = Path("/content/tray_dataset/labels")\n'
                 'prod_img_dir = Path("/content/product_dataset/images")\n'
                 'prod_lbl_dir = Path("/content/product_dataset/labels")\n'
                 'for d in [tray_img_dir, tray_lbl_dir, prod_img_dir, prod_lbl_dir]:\n'
                 '    d.mkdir(parents=True, exist_ok=True)\n\n'
                 'tray_count, prod_count = 0, 0\n\n'
                 'for lbl_file in aug_lbl_dir.glob("*.txt"):\n'
                 '    stem = lbl_file.stem\n'
                 '    img_file = None\n'
                 '    for ext in [".jpg", ".jpeg", ".png"]:\n'
                 '        c = aug_img_dir / (stem + ext)\n'
                 '        if c.exists():\n'
                 '            img_file = c\n'
                 '            break\n'
                 '    if not img_file:\n'
                 '        continue\n\n'
                 '    lines = lbl_file.read_text().strip().splitlines()\n'
                 '    tray_lines, prod_lines = [], []\n'
                 '    for line in lines:\n'
                 '        parts = line.split()\n'
                 '        if len(parts) < 5:\n'
                 '            continue\n'
                 '        cls_id = int(float(parts[0]))  # albumentations が float で返す場合に対応\n'
                 '        rest = " ".join(parts[1:])\n'
                 '        if cls_id in TRAY_FAMILY_IDXS:\n'
                 '            tray_lines.append(f"{TRAY_REMAP[cls_id]} {rest}")\n'
                 '        elif cls_id in PRODUCT_ID_MAP:\n'
                 '            prod_lines.append(f"{PRODUCT_ID_MAP[cls_id]} {rest}")\n\n'
                 '    if tray_lines:\n'
                 '        shutil.copy(img_file, tray_img_dir / img_file.name)\n'
                 '        (tray_lbl_dir / (stem + ".txt")).write_text("\\n".join(tray_lines))\n'
                 '        tray_count += 1\n'
                 '    if prod_lines:\n'
                 '        shutil.copy(img_file, prod_img_dir / img_file.name)\n'
                 '        (prod_lbl_dir / (stem + ".txt")).write_text("\\n".join(prod_lines))\n'
                 '        prod_count += 1\n\n'
                 'tray_yaml = {"path": "/content/tray_dataset", "train": "images", "val": "images", "names": TRAY_NAMES}\n'
                 'prod_yaml = {"path": "/content/product_dataset", "train": "images", "val": "images", "names": PRODUCT_NAMES}\n'
                 'with open("/content/tray_dataset/data.yaml", "w") as f:\n'
                 '    yaml.dump(tray_yaml, f, allow_unicode=True)\n'
                 'with open("/content/product_dataset/data.yaml", "w") as f:\n'
                 '    yaml.dump(prod_yaml, f, allow_unicode=True)\n\n'
                 'print(f"✅ トレイデータ: {tray_count} 枚 (クラス: {list(TRAY_NAMES.values())})")\n'
                 'print(f"✅ 製品データ: {prod_count} 枚")'),

            cell('# Step 7: tray_model.pt を学習（全バーコード共通のトレイ検出）\n'
                 '# epochs: 画像枚数が少ない場合は 50〜100 で十分。patience で早期終了する\n'
                 'EPOCHS_TRAY = 100\n'
                 'from ultralytics import YOLO\n\n'
                 'if tray_count == 0:\n'
                 '    print("⚠️ トレイアノテーションがありません。\'tray\' クラスでアノテーションされているか確認してください。")\n'
                 'else:\n'
                 '    tray_model = YOLO("yolov8n.pt")\n'
                 '    tray_model.train(\n'
                 '        data="/content/tray_dataset/data.yaml",\n'
                 '        epochs=EPOCHS_TRAY,\n'
                 '        imgsz=640,\n'
                 '        batch=16,\n'
                 '        patience=20,\n'
                 '        device=0,\n'
                 '        augment=False,\n'
                 '        project=SAVE_DIR,\n'
                 '        name="tray_model",\n'
                 '        save=True,\n'
                 '        exist_ok=True,\n'
                 '    )\n'
                 '    print("✅ tray_model 学習完了！")'),

            cell('# Step 8: product_model.pt を学習（このバーコード専用の製品検出）\n'
                 '# epochs: 画像枚数が少ない場合は 100〜150 で十分\n'
                 'EPOCHS_PRODUCT = 150\n'
                 'if prod_count == 0:\n'
                 '    print("⚠️ 製品アノテーションがありません。\'tray\' 以外のクラスでアノテーションされているか確認してください。")\n'
                 'else:\n'
                 '    prod_model = YOLO("yolov8n.pt")\n'
                 '    prod_model.train(\n'
                 '        data="/content/product_dataset/data.yaml",\n'
                 '        epochs=EPOCHS_PRODUCT,\n'
                 '        imgsz=640,\n'
                 '        batch=16,\n'
                 '        patience=25,\n'
                 '        device=0,\n'
                 '        augment=False,\n'
                 '        mosaic=0.0,\n'
                 '        degrees=3.0,\n'
                 '        translate=0.05,\n'
                 '        scale=0.1,\n'
                 '        flipud=0.0,\n'
                 '        fliplr=0.0,\n'
                 '        project=SAVE_DIR,\n'
                 '        name="product_model",\n'
                 '        save=True,\n'
                 '        exist_ok=True,\n'
                 '    )\n'
                 '    print("✅ product_model 学習完了！")'),

            cell('# Step 9: 精度確認\n'
                 'tray_best = f"{SAVE_DIR}/tray_model/weights/best.pt"\n'
                 'prod_best = f"{SAVE_DIR}/product_model/weights/best.pt"\n\n'
                 'if tray_count > 0:\n'
                 '    m = YOLO(tray_best).val(data="/content/tray_dataset/data.yaml", imgsz=640)\n'
                 '    print(f"🔲 tray_model  mAP50: {m.box.map50:.3f}  mAP50-95: {m.box.map:.3f}")\n\n'
                 'if prod_count > 0:\n'
                 '    m = YOLO(prod_best).val(data="/content/product_dataset/data.yaml", imgsz=640)\n'
                 '    print(f"📦 product_model  mAP50: {m.box.map50:.3f}  mAP50-95: {m.box.map:.3f}")'),

            cell('# Step 10: 2つのモデルをダウンロード\n'
                 'if tray_count > 0:\n'
                 '    shutil.copy(tray_best, "tray_model_best.pt")\n'
                 '    files.download("tray_model_best.pt")\n'
                 '    print("✅ tray_model_best.pt をダウンロード開始")\n\n'
                 'if prod_count > 0:\n'
                 '    shutil.copy(prod_best, "product_model_best.pt")\n'
                 '    files.download("product_model_best.pt")\n'
                 '    print("✅ product_model_best.pt をダウンロード開始")\n\n'
                 'print("\\n管理画面の「⚙️ 検品設定」からそれぞれのモデルを登録してください。")'),

            cell('# Step 11 (オプション): 基準写真から layout.txt を自動生成\n'
                 'print("基準写真をアップロードしてください...")\n'
                 'ref_uploaded = files.upload()\n'
                 'ref_img_path = list(ref_uploaded.keys())[0]\n\n'
                 'tray_det = None\n'
                 'if tray_count > 0 and os.path.exists(tray_best):\n'
                 '    tray_m = YOLO(tray_best)\n'
                 '    tr = tray_m(ref_img_path)[0]\n'
                 '    for box in tr.boxes:\n'
                 '        x1,y1,x2,y2 = box.xyxy[0].tolist()\n'
                 '        ih, iw = tr.orig_shape\n'
                 '        tray_det = {"cx":(x1+x2)/2/iw, "cy":(y1+y2)/2/ih,\n'
                 '                    "w":(x2-x1)/iw, "h":(y2-y1)/ih}\n'
                 '        break\n'
                 '    print(f"🔲 トレイ検出: {tray_det is not None}")\n\n'
                 'items = []\n'
                 'if prod_count > 0 and os.path.exists(prod_best):\n'
                 '    prod_m = YOLO(prod_best)\n'
                 '    pr = prod_m(ref_img_path, conf=0.3)[0]\n'
                 '    ih, iw = pr.orig_shape\n'
                 '    for box in pr.boxes:\n'
                 '        x1,y1,x2,y2 = box.xyxy[0].tolist()\n'
                 '        items.append({"class_name": prod_m.names[int(box.cls[0])],\n'
                 '                      "cx":(x1+x2)/2/iw, "cy":(y1+y2)/2/ih,\n'
                 '                      "w":(x2-x1)/iw, "h":(y2-y1)/ih})\n\n'
                 'if tray_det and items:\n'
                 '    tx1 = tray_det["cx"] - tray_det["w"]/2\n'
                 '    ty1 = tray_det["cy"] - tray_det["h"]/2\n'
                 '    items = [{**d,\n'
                 '              "cx":(d["cx"]-tx1)/tray_det["w"],\n'
                 '              "cy":(d["cy"]-ty1)/tray_det["h"],\n'
                 '              "w":d["w"]/tray_det["w"], "h":d["h"]/tray_det["h"]}\n'
                 '             for d in items]\n'
                 '    print("✅ トレイ相対座標で保存します")\n\n'
                 'with open("layout.txt", "w") as f:\n'
                 '    for item in items:\n'
                 '        f.write(f"{item[\'class_name\']} {item[\'cx\']:.6f} {item[\'cy\']:.6f} {item[\'w\']:.6f} {item[\'h\']:.6f}\\n")\n'
                 '        print(f"  {item[\'class_name\']}: ({item[\'cx\']:.3f}, {item[\'cy\']:.3f})")\n\n'
                 'files.download("layout.txt")\n'
                 'print(f"✅ {len(items)} アイテムの layout.txt をダウンロードしました")'),
        ],
    }

    return Response(
        content=json.dumps(nb, ensure_ascii=False, indent=1).encode("utf-8"),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="inspection_training.ipynb"'},
    )


@router.get("/train.py")
def download_train_script(folder: Optional[str] = Query(None), db: Session = Depends(get_db), _=Depends(get_current_user)):
    """ローカル環境用 Python 学習スクリプトを生成してダウンロード"""
    if folder:
        classes = db.query(models.TrainingClass).filter(
            models.TrainingClass.barcode_name == folder
        ).order_by(models.TrainingClass.class_index).all()
    else:
        classes = []
    class_names_py = repr([c.name for c in classes])

    script = r'''#!/usr/bin/env python3
"""
検品システム YOLOv8 2モデル学習スクリプト (ローカル実行用)
------------------------------------------------------
使い方:
  1. pip install ultralytics albumentations pyyaml opencv-python
  2. このファイルと同じフォルダに training_dataset.zip を置く
  3. python train.py

オプション:
  --dataset        学習データZIPのパス  (デフォルト: training_dataset.zip)
  --output         出力フォルダ         (デフォルト: ./output)
  --device         GPU番号 or cpu       (デフォルト: 0)
  --no-aug         水増し処理をスキップ
  --epochs-tray    トレイモデル学習回数    (デフォルト: 200)
  --epochs-product 製品モデル学習回数     (デフォルト: 300)
"""
import argparse, os, shutil, sys, zipfile, yaml
from pathlib import Path

CLASS_NAMES = __CLASS_NAMES__


def main():
    parser = argparse.ArgumentParser(description="検品システム 2モデル学習")
    parser.add_argument("--dataset", default="training_dataset.zip")
    parser.add_argument("--output", default="./output")
    parser.add_argument("--device", default="0", help="GPU番号 or cpu")
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--epochs-tray", type=int, default=200)
    parser.add_argument("--epochs-product", type=int, default=300)
    args = parser.parse_args()

    try:
        import ultralytics, albumentations, cv2
    except ImportError:
        print("必要なライブラリがありません。以下を実行してください:")
        print("  pip install ultralytics albumentations pyyaml opencv-python")
        sys.exit(1)

    from ultralytics import YOLO
    import albumentations as A
    import cv2
    import numpy as np

    DATASET_ZIP = Path(args.dataset)
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_ZIP.exists():
        print(f"エラー: {DATASET_ZIP} が見つかりません")
        print("管理画面の「写真データをダウンロード」でZIPを取得してください")
        sys.exit(1)

    # ── 解凍
    DATASET_DIR = OUTPUT_DIR / "dataset"
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    DATASET_DIR.mkdir()
    print(f"📦 解凍中: {DATASET_ZIP}")
    with zipfile.ZipFile(DATASET_ZIP) as z:
        z.extractall(DATASET_DIR)
    img_dir = DATASET_DIR / "images"
    lbl_dir = DATASET_DIR / "labels"
    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"📷 画像数: {len(imgs)} 枚")

    # ── クラス設定
    TRAY_FAMILY_IDXS = {i for i, n in enumerate(CLASS_NAMES) if n.lower().startswith("tray")}
    TRAY_REMAP = {orig: new for new, orig in enumerate(sorted(TRAY_FAMILY_IDXS))}
    TRAY_NAMES = {TRAY_REMAP[i]: CLASS_NAMES[i] for i in sorted(TRAY_FAMILY_IDXS)}
    PRODUCT_FAMILY_IDXS = {i for i, n in enumerate(CLASS_NAMES) if not n.lower().startswith("tray")}
    PRODUCT_ID_MAP = {orig: new for new, orig in enumerate(sorted(PRODUCT_FAMILY_IDXS))}
    PRODUCT_NAMES = {PRODUCT_ID_MAP[i]: CLASS_NAMES[i] for i in sorted(PRODUCT_FAMILY_IDXS)}
    print("クラス一覧:")
    for i, n in enumerate(CLASS_NAMES):
        tag = " ← トレイ系" if i in TRAY_FAMILY_IDXS else " ← 製品"
        print(f"  {i}: {n}{tag}")

    # ── 水増し
    aug_img_dir = DATASET_DIR / "aug_images"
    aug_lbl_dir = DATASET_DIR / "aug_labels"
    aug_img_dir.mkdir(exist_ok=True)
    aug_lbl_dir.mkdir(exist_ok=True)
    if not args.no_aug:
        print("\n🔄 水増し中 (×50)...")
        aug = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.3, p=0.8),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=25, p=0.5),
            A.GaussNoise(var_limit=(5, 40), p=0.6),
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
            A.Rotate(limit=4, p=0.5),
            A.RandomShadow(p=0.3),
            A.ImageCompression(quality_lower=70, p=0.3),
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3))
        for f in imgs:
            shutil.copy(f, aug_img_dir / f.name)
            lbl = lbl_dir / (f.stem + ".txt")
            if lbl.exists():
                # 点アノテーション(w=0,h=0)を YOLO 用サイズに変換してコピー
                (aug_lbl_dir / lbl.name).write_text(_label_text_for_yolo(lbl))
        for img_path in imgs:
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            bboxes, cls_ids = [], []
            if lbl_path.exists():
                for line in lbl_path.read_text().strip().splitlines():
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_ids.append(int(parts[0]))
                        b = [float(x) for x in parts[1:5]]
                        if b[2] == 0.0 and b[3] == 0.0:  # 点アノテーション
                            b[2] = b[3] = POINT_EXPORT_SIZE
                        bboxes.append(b)
            for i in range(50):
                try:
                    r = aug(image=img, bboxes=bboxes, class_labels=cls_ids)
                    name = f"{img_path.stem}_aug{i:03d}"
                    cv2.imwrite(str(aug_img_dir / f"{name}.jpg"),
                                cv2.cvtColor(r["image"], cv2.COLOR_RGB2BGR))
                    with open(aug_lbl_dir / f"{name}.txt", "w") as g:
                        for c, b in zip(r["class_labels"], r["bboxes"]):
                            g.write(f"{int(c)} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                except Exception:
                    pass
        print(f"✅ 水増し完了: {len(list(aug_img_dir.glob('*.jpg')))} 枚")
    else:
        for f in imgs:
            shutil.copy(f, aug_img_dir / f.name)
            lbl = lbl_dir / (f.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, aug_lbl_dir / lbl.name)

    # ── 分割
    print("\n🔀 データ分割中...")
    tray_img = OUTPUT_DIR / "tray_dataset/images"
    tray_lbl = OUTPUT_DIR / "tray_dataset/labels"
    prod_img = OUTPUT_DIR / "product_dataset/images"
    prod_lbl = OUTPUT_DIR / "product_dataset/labels"
    for d in [tray_img, tray_lbl, prod_img, prod_lbl]:
        d.mkdir(parents=True, exist_ok=True)
    tray_count = prod_count = 0
    for lbl_file in aug_lbl_dir.glob("*.txt"):
        stem = lbl_file.stem
        img_file = next(
            (aug_img_dir / (stem + ext) for ext in [".jpg", ".jpeg", ".png"]
             if (aug_img_dir / (stem + ext)).exists()), None)
        if not img_file:
            continue
        lines = lbl_file.read_text().strip().splitlines()
        tray_lines, prod_lines = [], []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))  # albumentations が float で返す場合に対応
            rest = " ".join(parts[1:])
            if cls_id in TRAY_FAMILY_IDXS:
                tray_lines.append(f"{TRAY_REMAP[cls_id]} {rest}")
            elif cls_id in PRODUCT_ID_MAP:
                prod_lines.append(f"{PRODUCT_ID_MAP[cls_id]} {rest}")
        if tray_lines:
            shutil.copy(img_file, tray_img / img_file.name)
            (tray_lbl / (stem + ".txt")).write_text("\n".join(tray_lines))
            tray_count += 1
        if prod_lines:
            shutil.copy(img_file, prod_img / img_file.name)
            (prod_lbl / (stem + ".txt")).write_text("\n".join(prod_lines))
            prod_count += 1
    with open(OUTPUT_DIR / "tray_dataset/data.yaml", "w") as f:
        yaml.dump({"path": str(OUTPUT_DIR / "tray_dataset"), "train": "images",
                   "val": "images", "names": TRAY_NAMES}, f, allow_unicode=True)
    with open(OUTPUT_DIR / "product_dataset/data.yaml", "w") as f:
        yaml.dump({"path": str(OUTPUT_DIR / "product_dataset"), "train": "images",
                   "val": "images", "names": PRODUCT_NAMES}, f, allow_unicode=True)
    print(f"✅ トレイ: {tray_count} 枚 (クラス: {list(TRAY_NAMES.values())}) / 製品: {prod_count} 枚")

    # ── 学習: tray_model
    if tray_count > 0:
        print("\n🔲 tray_model 学習中...")
        YOLO("yolov8n.pt").train(
            data=str(OUTPUT_DIR / "tray_dataset/data.yaml"),
            epochs=args.epochs_tray, imgsz=640, batch=8, patience=30,
            device=args.device, augment=False,
            project=str(OUTPUT_DIR), name="tray_model", save=True, exist_ok=True,
        )
        shutil.copy(OUTPUT_DIR / "tray_model/weights/best.pt", OUTPUT_DIR / "tray_model_best.pt")
        print(f"✅ 出力: {(OUTPUT_DIR / 'tray_model_best.pt').resolve()}")

    # ── 学習: product_model
    if prod_count > 0:
        print("\n📦 product_model 学習中...")
        YOLO("yolov8n.pt").train(
            data=str(OUTPUT_DIR / "product_dataset/data.yaml"),
            epochs=args.epochs_product, imgsz=640, batch=8, patience=50,
            device=args.device, augment=False,
            mosaic=0.0, degrees=3.0, translate=0.05, scale=0.1, flipud=0.0, fliplr=0.0,
            project=str(OUTPUT_DIR), name="product_model", save=True, exist_ok=True,
        )
        shutil.copy(OUTPUT_DIR / "product_model/weights/best.pt", OUTPUT_DIR / "product_model_best.pt")
        print(f"✅ 出力: {(OUTPUT_DIR / 'product_model_best.pt').resolve()}")

    print("\n🎉 学習完了！")
    print(f"出力フォルダ: {OUTPUT_DIR.resolve()}")
    print("管理画面の「⚙️ 検品設定」から .pt ファイルを登録してください")


if __name__ == "__main__":
    main()
'''
    script = script.replace("__CLASS_NAMES__", class_names_py)
    return Response(
        content=script.encode("utf-8"),
        media_type="text/x-python",
        headers={"Content-Disposition": 'attachment; filename="train.py"'},
    )
