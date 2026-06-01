from contextlib import asynccontextmanager
import logging
import threading
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s",
    force=True,  # uvicorn が先に設定していても上書きする
)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from database import engine, SessionLocal, Base
import models
from routers import auth, barcode, capture, admin_barcodes, admin_models, admin_images, admin_users, training

_logger = logging.getLogger(__name__)


def _create_initial_admin():
    from auth import hash_password
    db = SessionLocal()
    try:
        if not db.query(models.User).first():
            admin = models.User(
                username="admin",
                password_hash=hash_password("admin123"),
                is_active=True,
            )
            db.add(admin)
            db.commit()
            print("Initial admin user created: admin / admin123")
    finally:
        db.close()


def _migrate():
    """新カラムを既存DBに追加（冪等）"""
    from sqlalchemy import text
    migrations = [
        "ALTER TABLE barcodes ADD COLUMN layout_path VARCHAR(1000) NULL",
        "ALTER TABLE training_images ADD COLUMN barcode_name VARCHAR(255) NULL",
        "ALTER TABLE barcodes ADD COLUMN tray_model_id INT NULL",
        "ALTER TABLE training_classes ADD COLUMN barcode_name VARCHAR(255) NULL",
        "ALTER TABLE training_classes DROP INDEX `name`",           # unique制約を削除
        "ALTER TABLE training_classes DROP INDEX `ix_training_classes_name`",
        "ALTER TABLE barcodes ADD COLUMN reference_image_path VARCHAR(1000) NULL",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(text(sql))
                conn.commit()
                print(f"Migration applied: {sql[:60]}")
            except Exception:
                pass  # カラムが既にある場合はスキップ


def _migrate_global_classes_to_folders():
    """旧グローバルクラスをラベル済みフォルダにマージ（欠けているクラスのみ追加・冪等）"""
    db = SessionLocal()
    try:
        global_classes = db.query(models.TrainingClass).filter(
            models.TrainingClass.barcode_name == None
        ).order_by(models.TrainingClass.class_index).all()
        if not global_classes:
            return

        rows = db.query(models.TrainingImage.barcode_name).filter(
            models.TrainingImage.barcode_name.isnot(None),
            models.TrainingImage.label_path.isnot(None),
        ).distinct().all()

        for (folder_name,) in rows:
            if not folder_name:
                continue
            existing = db.query(models.TrainingClass).filter(
                models.TrainingClass.barcode_name == folder_name
            ).all()
            existing_names = {c.name for c in existing}
            next_idx = max((c.class_index for c in existing), default=-1) + 1
            added = False
            for gc in global_classes:
                if gc.name not in existing_names:
                    db.add(models.TrainingClass(
                        name=gc.name,
                        class_index=next_idx,
                        barcode_name=folder_name,
                    ))
                    next_idx += 1
                    added = True
            if added:
                db.commit()
                print(f"Migration: merged global classes → folder '{folder_name}'")
    finally:
        db.close()


def _warmup_models():
    """登録済み YOLO モデルを起動時にロード＆ダミー推論して初回リクエストの遅延を解消する。"""
    try:
        from services.yolo_service import _get_model
        import numpy as np
        from PIL import Image
        db = SessionLocal()
        paths = [r[0] for r in db.query(models.YoloModel.model_path)
                 .filter(models.YoloModel.model_path.isnot(None)).all()]
        db.close()
        _logger.info("Warmup: %d model(s) to warm up", len(paths))
        dummy = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        for path in paths:
            if not path or not Path(path).exists():
                _logger.warning("Warmup skip (not found): %s", path)
                continue
            try:
                m = _get_model(path)
                m.predict(source=dummy, imgsz=64, conf=0.5, verbose=False)
                _logger.info("Warmup OK: %s", path)
            except Exception as e:
                _logger.warning("Warmup failed (%s): %s", path, e)
        _logger.info("Warmup complete")
    except Exception as e:
        _logger.error("Warmup error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    _migrate()
    _migrate_global_classes_to_folders()
    _create_initial_admin()
    threading.Thread(target=_warmup_models, daemon=True, name="warmup").start()
    yield


app = FastAPI(title="Inspection API", version="1.0.0", docs_url="/api/docs", lifespan=lifespan)

app.include_router(auth.router)
app.include_router(barcode.router)
app.include_router(capture.router)
app.include_router(admin_barcodes.router)
app.include_router(admin_models.router)
app.include_router(admin_images.router)
app.include_router(admin_users.router)
app.include_router(training.router)

FRONTEND = Path("/frontend")


@app.get("/")
def scan_page():
    return FileResponse(str(FRONTEND / "scan.html"))


@app.get("/capture")
def capture_page():
    return FileResponse(str(FRONTEND / "capture.html"))


@app.get("/admin")
def admin_index():
    return FileResponse(str(FRONTEND / "admin" / "index.html"))


@app.get("/admin/login")
def admin_login():
    return FileResponse(str(FRONTEND / "admin" / "login.html"))


@app.get("/admin/_common.js")
def common_js():
    return FileResponse(str(FRONTEND / "admin" / "_common.js"), media_type="application/javascript")


@app.get("/admin/{page}")
def admin_page(page: str):
    p = FRONTEND / "admin" / f"{page}.html"
    if p.exists():
        return FileResponse(str(p))
    return RedirectResponse("/admin")
