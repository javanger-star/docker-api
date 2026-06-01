from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel


# Auth
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserResponse(BaseModel):
    id: int
    username: str
    is_active: bool
    created_at: datetime
    model_config = {"from_attributes": True}

class UserCreate(BaseModel):
    username: str
    password: str

class UserUpdate(BaseModel):
    password: Optional[str] = None
    is_active: Optional[bool] = None


# YoloModel
class YoloModelResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    model_path: Optional[str]
    layout_path: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime
    model_config = {"from_attributes": True}

class YoloModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None


# Barcode
class BarcodeCheck(BaseModel):
    found: bool
    barcode_value: str
    name: Optional[str] = None
    has_model: bool = False
    has_layout: bool = False
    model_id: Optional[int] = None

class BarcodeResponse(BaseModel):
    id: int
    barcode_value: str
    name: str
    description: Optional[str]
    tray_model_id: Optional[int]
    tray_model: Optional[YoloModelResponse]
    yolo_model_id: Optional[int]
    yolo_model: Optional[YoloModelResponse]
    layout_path: Optional[str]
    reference_image_path: Optional[str]
    created_at: datetime
    updated_at: datetime
    model_config = {"from_attributes": True}

class BarcodeCreate(BaseModel):
    barcode_value: str
    name: str
    description: Optional[str] = None
    tray_model_id: Optional[int] = None
    yolo_model_id: Optional[int] = None

class BarcodeUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tray_model_id: Optional[int] = None
    yolo_model_id: Optional[int] = None

class LayoutGenerateResult(BaseModel):
    layout_path: str
    item_count: int
    items: List[Any]


# CapturedImage
class CapturedImageResponse(BaseModel):
    id: int
    barcode_id: int
    filename: str
    file_path: str
    thumbnail_path: Optional[str]
    captured_at: datetime
    detection_result: Optional[Any]
    barcode: Optional[BarcodeResponse] = None
    model_config = {"from_attributes": True}
