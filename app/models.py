from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())


class YoloModel(Base):
    __tablename__ = "yolo_models"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    model_path = Column(String(500))
    layout_path = Column(String(500))
    status = Column(String(20), default="active")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    barcodes = relationship("Barcode", foreign_keys="Barcode.yolo_model_id", back_populates="yolo_model")


class Barcode(Base):
    __tablename__ = "barcodes"
    id = Column(Integer, primary_key=True, index=True)
    barcode_value = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    tray_model_id = Column(Integer, ForeignKey("yolo_models.id"), nullable=True)   # トレイ検出モデル
    yolo_model_id = Column(Integer, ForeignKey("yolo_models.id"), nullable=True)   # 製品検出モデル
    layout_path = Column(String(1000), nullable=True)
    reference_image_path = Column(String(1000), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    tray_model = relationship("YoloModel", foreign_keys=[tray_model_id])
    yolo_model  = relationship("YoloModel", foreign_keys=[yolo_model_id], back_populates="barcodes")
    images = relationship("CapturedImage", back_populates="barcode", cascade="all, delete-orphan")


class TrainingClass(Base):
    __tablename__ = "training_classes"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)          # unique per barcode_name (enforced in API)
    barcode_name = Column(String(255), nullable=True, index=True)  # NULL = 旧グローバル（後方互換）
    class_index = Column(Integer, nullable=False)       # フォルダ内での順序 (0, 1, 2…)
    created_at = Column(DateTime, server_default=func.now())


class TrainingImage(Base):
    __tablename__ = "training_images"
    id = Column(Integer, primary_key=True, index=True)
    barcode_name = Column(String(255), nullable=True)   # ZIPファイル名 = バーコード名 = フォルダ名
    filename = Column(String(500), nullable=False)
    image_path = Column(String(1000), nullable=False)
    label_path = Column(String(1000))
    created_at = Column(DateTime, server_default=func.now())


class CapturedImage(Base):
    __tablename__ = "captured_images"
    id = Column(Integer, primary_key=True, index=True)
    barcode_id = Column(Integer, ForeignKey("barcodes.id"), nullable=False)
    filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    thumbnail_path = Column(String(1000))
    captured_at = Column(DateTime, server_default=func.now())
    detection_result = Column(JSON)

    barcode = relationship("Barcode", back_populates="images")
