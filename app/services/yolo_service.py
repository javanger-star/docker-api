from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ── Environment config ────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except Exception: return default

def _env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except Exception: return default

DEFAULT_IMGSZ         = _env_int("YOLO_IMGSZ", 1536)
PRODUCT_IMGSZ         = _env_int("YOLO_PRODUCT_IMGSZ", DEFAULT_IMGSZ)
TRAY_IMGSZ            = _env_int("YOLO_TRAY_IMGSZ", DEFAULT_IMGSZ)   # トレイモデル用（個別に制御）
DEFAULT_CONF          = _env_float("YOLO_CONF", 0.05)
DEFAULT_NMS_IOU       = _env_float("YOLO_NMS_IOU", 0.45)
DEFAULT_MAX_DET       = _env_int("YOLO_MAX_DET", 500)
INFER_MIN_CONF        = _env_float("YOLO_INFER_MIN_CONF", 0.001)
SLOT_MIN_CONF         = _env_float("YOLO_SLOT_MIN_CONF", 0.15)  # スロット存在確認の最低 conf（空トレイ誤検出を防ぐ）
ANCHOR_CONF           = _env_float("YOLO_ANCHOR_CONF", 0.05)    # 全 det の最大 conf がこれ未満なら空トレイとみなす
POSITION_GATE_RATIO   = _env_float("YOLO_POSITION_GATE_RATIO", 1.5)  # レイアウト元座標からこの倍率×スロットサイズ以上離れた検出は demote
UNIFORM_GATE_MAX_CONF = _env_float("YOLO_UNIFORM_GATE_MAX_CONF", 0.50)  # 均一ゲート: max_conf がこれ未満のときのみ評価
UNIFORM_GATE_MAX_STD  = _env_float("YOLO_UNIFORM_GATE_MAX_STD", 0.03)   # 均一ゲート: conf の標準偏差がこれ未満なら空トレイとみなす
UNIFORM_GATE_MIN_DET  = _env_int("YOLO_UNIFORM_GATE_MIN_DET", 3)         # 均一ゲートを適用する最低 det 数
YOLO_AGNOSTIC_NMS     = os.getenv("YOLO_AGNOSTIC_NMS", "0") not in ("0", "false", "False")

AUTO_CROP_ENABLE         = os.getenv("YOLO_AUTO_CROP", "1") not in ("0", "false", "False")
AUTO_CROP_MIN_SIDE       = _env_int("YOLO_AUTO_CROP_MIN_SIDE", 2000)
AUTO_CROP_WHITE_THRESH   = _env_int("YOLO_AUTO_CROP_WHITE_THRESH", 245)
AUTO_CROP_PADDING        = _env_int("YOLO_AUTO_CROP_PADDING", 40)
AUTO_CROP_MIN_AREA_RATIO = _env_float("YOLO_AUTO_CROP_MIN_AREA_RATIO", 0.12)
AUTO_CROP_BORDER_FRAC    = _env_float("YOLO_AUTO_CROP_BORDER_FRAC", 0.03)
AUTO_CROP_BG_PERCENTILE  = _env_float("YOLO_AUTO_CROP_BG_PERCENTILE", 95.0)
AUTO_CROP_BG_DELTA       = _env_int("YOLO_AUTO_CROP_BG_DELTA", 15)

ALIGN_MIN_CONF  = _env_float("YOLO_ALIGN_MIN_CONF", 0.01)
ALIGN_SCALE_MIN = _env_float("YOLO_ALIGN_SCALE_MIN", 0.80)
ALIGN_SCALE_MAX = _env_float("YOLO_ALIGN_SCALE_MAX", 1.25)

LAYOUT_GATE_ENABLE  = os.getenv("YOLO_LAYOUT_GATE", "1") not in ("0", "false", "False")
CLASS_CHECK_ENABLE  = os.getenv("YOLO_CLASS_CHECK", "1") not in ("0", "false", "False")  # クラス不一致→misplaced 判定を行うか
GATE_CONF_FACTOR   = _env_float("YOLO_GATE_CONF_FACTOR", 0.25)
GATE_MIN_CONF      = _env_float("YOLO_GATE_MIN_CONF", 0.001)
GATE_DIST_RATIO    = _env_float("YOLO_GATE_DIST_RATIO", 1.25)
GATE_DIST_MIN      = _env_float("YOLO_GATE_DIST_MIN", 80.0)
GATE_DIST_MAX      = _env_float("YOLO_GATE_DIST_MAX", 360.0)

EXTRA_MIN_CONF_ENV  = os.getenv("YOLO_EXTRA_MIN_CONF", "").strip()
EXTRA_SUPPRESS_IOU  = _env_float("YOLO_EXTRA_SUPPRESS_IOU", 0.65)
EXTRA_KEEP_TOPK     = _env_int("YOLO_EXTRA_KEEP_TOPK", 0)

LABEL_MATCH_BONUS   = _env_float("YOLO_LABEL_MATCH_BONUS", 0.5)  # class-match priority in greedy matching
TRAY_CROP_PAD       = _env_float("YOLO_TRAY_CROP_PAD", 0.02)    # padding ratio around tray crop

# ── Misplaced (yellow) thresholds ─────────────────────────────────────────────
# IoU: スロットとの空間的重なり。0 に近いほど位置がずれている。
# アライメント誤差を考慮し 0.05 程度が実用的。
MISMATCH_MIN_IOU   = _env_float("YOLO_MISMATCH_MIN_IOU", 0.02)
# Size: sqrt(det_area / slot_area) = 線形スケール比。
# スロットは「枠」の大きさ、検出は製品の大きさなので同一ではない。
# 0.2 未満（スロットの20%未満）または 3.0 超（スロットの3倍超）を異常とみなす。
MISMATCH_SCALE_MIN = _env_float("YOLO_MISMATCH_SCALE_MIN", 0.2)
MISMATCH_SCALE_MAX = _env_float("YOLO_MISMATCH_SCALE_MAX", 3.0)

# ── Quality / retake thresholds ───────────────────────────────────────────────
QUALITY_MIN_TRAY_CONF   = _env_float("QUALITY_MIN_TRAY_CONF", 0.40)
QUALITY_MIN_MATCH_RATE  = _env_float("QUALITY_MIN_MATCH_RATE", 0.50)
QUALITY_BLUR_THRESH     = _env_float("QUALITY_BLUR_THRESH", 60.0)
QUALITY_BLUR_ENABLE     = os.getenv("QUALITY_BLUR_CHECK", "1") not in ("0", "false", "False")

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class _Det:
    label: str
    class_id: int
    confidence: float
    x1: float; y1: float; x2: float; y2: float
    requested_conf: float = 0.0

    @property
    def center_x(self) -> float: return (self.x1 + self.x2) / 2
    @property
    def center_y(self) -> float: return (self.y1 + self.y2) / 2
    @property
    def center(self) -> Tuple[float, float]: return (self.center_x, self.center_y)
    def as_xyxy(self) -> Tuple[float, float, float, float]: return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class _Slot:
    slot_id: str
    x1: float; y1: float; x2: float; y2: float
    class_name: str = ""

    @property
    def width(self) -> float: return max(0.0, self.x2 - self.x1)
    @property
    def height(self) -> float: return max(0.0, self.y2 - self.y1)
    @property
    def center_x(self) -> float: return (self.x1 + self.x2) / 2
    @property
    def center_y(self) -> float: return (self.y1 + self.y2) / 2
    @property
    def center(self) -> Tuple[float, float]: return (self.center_x, self.center_y)
    def as_xyxy(self) -> Tuple[float, float, float, float]: return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class _MatchResult:
    slot: Optional[_Slot]
    det: Optional[_Det]
    iou: float
    status: str  # ok / missing / extra

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _laplacian_variance(pil_img: Image.Image) -> float:
    """Image sharpness via Laplacian variance. Lower = more blurry."""
    try:
        arr = np.array(pil_img.convert("L"), dtype=np.float32)
        if _HAS_CV2:
            lap = cv2.Laplacian(arr, cv2.CV_64F)
            return float(np.var(lap))
        # Fallback: simple second-order difference
        dy = arr[2:, :] - 2 * arr[1:-1, :] + arr[:-2, :]
        dx = arr[:, 2:] - 2 * arr[:, 1:-1] + arr[:, :-2]
        return float(np.var(dy) + np.var(dx))
    except Exception:
        return 9999.0  # safe: don't flag as blurry on error

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _compute_imgsz(img: Image.Image, max_size: int) -> Tuple[int, int]:
    """Return (H, W) imgsz that preserves aspect ratio with minimal letterboxing.
    YOLO requires dimensions to be multiples of the model stride (32).
    """
    stride = 32
    w, h = img.size  # PIL gives (width, height)
    scale = min(1.0, max_size / max(w, h))
    nw = max(stride, math.ceil(w * scale / stride) * stride)
    nh = max(stride, math.ceil(h * scale / stride) * stride)
    return (nh, nw)  # YOLO expects (H, W)

def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0

# ── Auto-crop (V36) ───────────────────────────────────────────────────────────

def _auto_crop_nonwhite(img: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
    """Crop away white/bright background to improve detection on large images."""
    try:
        arr = np.asarray(img.convert("RGB"))
        h, w = arr.shape[:2]
        if h < 2 or w < 2:
            return img, (0, 0)

        rgb = arr.astype(np.float32)
        luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

        border_px = max(10, min(int(min(w, h) * AUTO_CROP_BORDER_FRAC), 120))
        border = np.concatenate([
            luma[:border_px, :].ravel(),
            luma[max(0, h - border_px):, :].ravel(),
            luma[:, :border_px].ravel(),
            luma[:, max(0, w - border_px):].ravel(),
        ])
        bg = float(np.percentile(border, AUTO_CROP_BG_PERCENTILE))
        thr = float(np.clip(min(float(AUTO_CROP_WHITE_THRESH), bg - float(AUTO_CROP_BG_DELTA)), 0, 255))

        best = None
        for _ in range(4):
            mask = luma < thr
            coords = np.argwhere(mask)
            if coords.size == 0:
                thr = min(float(AUTO_CROP_WHITE_THRESH), thr + 10.0)
                continue
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            area_ratio = (x1 - x0 + 1) * (y1 - y0 + 1) / float(w * h)
            best = (int(x0), int(y0), int(x1), int(y1), area_ratio)
            if x0 <= 0 and y0 <= 0 and x1 >= w - 1 and y1 >= h - 1:
                thr = max(0.0, thr - 12.0)
                continue
            break

        if best is None or best[4] < AUTO_CROP_MIN_AREA_RATIO:
            return img, (0, 0)

        x0, y0, x1, y1 = best[:4]
        x0p = max(0, x0 - AUTO_CROP_PADDING)
        y0p = max(0, y0 - AUTO_CROP_PADDING)
        x1p = min(w, x1 + 1 + AUTO_CROP_PADDING)
        y1p = min(h, y1 + 1 + AUTO_CROP_PADDING)
        if x0p == 0 and y0p == 0 and x1p == w and y1p == h:
            return img, (0, 0)

        logger.info("autocrop: (%d,%d) -> (%d,%d) off=(%d,%d)", w, h, x1p - x0p, y1p - y0p, x0p, y0p)
        return img.crop((x0p, y0p, x1p, y1p)), (x0p, y0p)
    except Exception as e:
        logger.warning("autocrop failed: %s", e)
        return img, (0, 0)

# ── YOLO inference ────────────────────────────────────────────────────────────

_MODEL_CACHE: Dict[str, object] = {}

def _get_model(path: str):
    from ultralytics import YOLO
    if path not in _MODEL_CACHE:
        m = YOLO(path)
        logger.info("Model loaded: %s  nc=%s  names=%s", path,
                    getattr(m.model, "nc", "?"), getattr(m, "names", "?"))
        _MODEL_CACHE[path] = m
    return _MODEL_CACHE[path]


def _run_yolo(model_path: str, pil_img: Image.Image, req_conf: float = DEFAULT_CONF,
              exact_conf: bool = False) -> List[_Det]:
    """Run YOLO with auto-crop; returns pixel-space detections.
    exact_conf=True: YOLO は req_conf で直接推論（レイアウト生成用）
    exact_conf=False: YOLO は INFER_MIN_CONF で推論しゲートフィルタ後に絞り込み（通常判定用）
    """
    model = _get_model(model_path)
    infer_conf = req_conf if exact_conf else (min(req_conf, INFER_MIN_CONF) if INFER_MIN_CONF > 0 else req_conf)

    infer_img, (off_x, off_y) = pil_img, (0, 0)
    if AUTO_CROP_ENABLE and max(pil_img.size) >= AUTO_CROP_MIN_SIDE:
        infer_img, (off_x, off_y) = _auto_crop_nonwhite(pil_img)

    imgsz = _compute_imgsz(infer_img, PRODUCT_IMGSZ)
    try:
        res = model.predict(
            source=infer_img,
            imgsz=imgsz,
            conf=infer_conf,
            iou=DEFAULT_NMS_IOU,
            max_det=DEFAULT_MAX_DET,
            agnostic_nms=YOLO_AGNOSTIC_NMS,
            verbose=False,
        )
    except TypeError:
        try:
            res = model.predict(source=infer_img, imgsz=imgsz, conf=infer_conf,
                                iou=DEFAULT_NMS_IOU, max_det=DEFAULT_MAX_DET, verbose=False)
        except TypeError:
            res = model.predict(source=infer_img, imgsz=imgsz, conf=infer_conf, verbose=False)

    dets: List[_Det] = []
    for r in res:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls_id = int(b.cls)
            names = getattr(r, "names", None)
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            x1, y1, x2, y2 = [float(v) for v in b.xyxy.cpu().numpy()[0]]
            dets.append(_Det(label, cls_id, float(b.conf),
                             x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y,
                             requested_conf=req_conf))

    logger.info("yolo det=%d img=%s infer=%s imgsz=%s conf=%.4f", len(dets), pil_img.size, infer_img.size, imgsz, req_conf)
    return dets

# ── Layout file I/O ───────────────────────────────────────────────────────────

def parse_layout(layout_path: str) -> Tuple[list, bool]:
    """Read layout.txt → (items, is_tray_relative).

    First line may be '# coord: tray_relative' or '# coord: image_relative'.
    If no header, defaults to True (annotation-based layouts are always tray-relative;
    backward-compatible with existing files generated before the header was added).
    """
    items = []
    is_tray_relative: Optional[bool] = None
    try:
        with open(layout_path) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("# coord:"):
                    coord = stripped.split(":", 1)[1].strip()
                    is_tray_relative = (coord == "tray_relative")
                    continue
                parts = stripped.split()
                if len(parts) >= 5:
                    items.append({
                        "class_name": parts[0],
                        "cx": float(parts[1]),
                        "cy": float(parts[2]),
                        "w":  float(parts[3]),
                        "h":  float(parts[4]),
                    })
    except Exception:
        pass
    if is_tray_relative is None:
        is_tray_relative = True  # backward compat default
    return items, is_tray_relative


def _layout_to_slots(layout_items: list, img_w: int, img_h: int) -> List[_Slot]:
    """Convert normalized layout items to pixel-space _Slot objects."""
    slots = []
    for i, item in enumerate(layout_items):
        cx = item["cx"] * img_w
        cy = item["cy"] * img_h
        half_w = (item["w"] * img_w) / 2
        half_h = (item["h"] * img_h) / 2
        slots.append(_Slot(
            slot_id=str(i),
            x1=cx - half_w, y1=cy - half_h,
            x2=cx + half_w, y2=cy + half_h,
            class_name=item.get("class_name", ""),
        ))
    return slots

# ── Tray corner labels & reference positions ──────────────────────────────────

# tray-relative (0,0)=左上 (1,0)=右上 (0,1)=左下 (1,1)=右下
_CORNER_REF: Dict[str, Tuple[float, float]] = {
    "tray_lu": (0.0, 0.0), "tray-lu": (0.0, 0.0),
    "tray_ru": (1.0, 0.0), "tray-ru": (1.0, 0.0),
    "tray_ld": (0.0, 1.0), "tray-ld": (0.0, 1.0),
    "tray_rd": (1.0, 1.0), "tray-rd": (1.0, 1.0),
}

# ── Tray detection & coordinate conversion ────────────────────────────────────

def _detect_tray(tray_model_path: str, image_source) -> Optional[dict]:
    """Detect tray bounding box and optionally 4 corner markers.
    image_source: ファイルパス(str) または PIL.Image（リサイズ済み画像を渡す場合）
    """
    model = _get_model(tray_model_path)
    results = model(image_source, conf=0.005, imgsz=TRAY_IMGSZ, verbose=False)[0]
    img_h, img_w = results.orig_shape
    names = getattr(results, "names", {})

    tray_box: Optional[dict] = None
    corners: Dict[str, dict] = {}

    for box in results.boxes:
        cls_id = int(box.cls)
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        conf = round(float(box.conf[0]), 3)

        if label == "tray":
            if tray_box is None or conf > tray_box.get("conf", 0):
                tray_box = {
                    "class_name": "tray",
                    "cx": cx, "cy": cy,
                    "w": (x2 - x1) / img_w,
                    "h": (y2 - y1) / img_h,
                    "conf": conf,
                }
        elif label in _CORNER_REF:
            norm = label.replace("-", "_")
            if norm not in corners or conf > corners[norm].get("conf", 0):
                corners[norm] = {"cx": cx, "cy": cy, "conf": conf}

    # "tray" クラスがなくコーナーのみのモデルの場合: コーナーからトレイ矩形を合成する
    # ただし x・y 両方向に十分なスパンがある場合のみ（片側コーナーのみでは合成しない）
    if tray_box is None and corners:
        cxs = [c["cx"] for c in corners.values()]
        cys = [c["cy"] for c in corners.values()]
        min_cx, max_cx = min(cxs), max(cxs)
        min_cy, max_cy = min(cys), max(cys)
        span_x = max_cx - min_cx
        span_y = max_cy - min_cy
        MIN_SPAN = 0.10  # 最低限の幅・高さスパン（画像幅/高さの10%）

        if len(corners) >= 2 and span_x >= MIN_SPAN and span_y >= MIN_SPAN:
            pad_x = span_x * 0.08
            pad_y = span_y * 0.08
            tray_box = {
                "class_name": "tray",
                "cx": (min_cx + max_cx) / 2,
                "cy": (min_cy + max_cy) / 2,
                "w":  min(1.0, span_x + 2 * pad_x),
                "h":  min(1.0, span_y + 2 * pad_y),
                "conf": round(sum(c["conf"] for c in corners.values()) / len(corners), 3),
            }
            logger.info("tray synthesized from %d corners: cx=%.3f cy=%.3f w=%.3f h=%.3f conf=%.3f",
                        len(corners), tray_box["cx"], tray_box["cy"],
                        tray_box["w"], tray_box["h"], tray_box["conf"])
        else:
            logger.warning("tray synthesis skipped: corners=%d span_x=%.3f span_y=%.3f (need >=2 corners and span>=%.2f)",
                           len(corners), span_x, span_y, MIN_SPAN)

    if tray_box is None:
        return None
    tray_box["corners"] = corners
    logger.info("tray detected conf=%.3f corners=%s", tray_box["conf"], list(corners.keys()))
    return tray_box



def _corner_homography(corners: Dict[str, dict], img_w: int, img_h: int) -> Optional[np.ndarray]:
    """Compute transform: tray-relative [0,1]² → image pixels from detected corners.

    4 corners → perspective homography (3×3, best accuracy).
    2-3 corners → similarity transform (scale + rotation + translation) embedded in 3×3.
    Returns a 3×3 matrix compatible with cv2.perspectiveTransform, or None.
    """
    if not _HAS_CV2 or len(corners) < 2:
        return None
    src, dst = [], []
    for norm_label, ref_pos in _CORNER_REF.items():
        if norm_label in corners:
            src.append(list(ref_pos))
            dst.append([corners[norm_label]["cx"] * img_w, corners[norm_label]["cy"] * img_h])
            if len(src) == 4:
                break
    if len(src) < 2:
        return None
    src_arr = np.array(src, dtype=np.float32)
    dst_arr = np.array(dst, dtype=np.float32)

    if len(src) >= 4:
        logger.info("corner H input: src=%s dst=%s",
                    [(round(p[0],3), round(p[1],3)) for p in src_arr.tolist()],
                    [(round(p[0],1), round(p[1],1)) for p in dst_arr.tolist()])
        H, _ = cv2.findHomography(src_arr, dst_arr)
        if H is not None:
            # H が [0,1]² をどの程度の画素領域にマップするか検証
            test = cv2.perspectiveTransform(
                np.array([[[0.,0.],[1.,0.],[0.,1.],[1.,1.]]], dtype=np.float32), H)[0]
            span_x = float(test[:,0].max() - test[:,0].min())
            span_y = float(test[:,1].max() - test[:,1].min())
            logger.info("corner H: span_x=%.0fpx span_y=%.0fpx (image %dx%d)",
                        span_x, span_y, img_w, img_h)
            # スパンが画像の10%未満は縮退ホモグラフィとして棄却
            if span_x < img_w * 0.10 or span_y < img_h * 0.10:
                logger.warning("corner H degenerate (span too small) → rejecting, will use tray bbox")
                return None
            # コーナーが正しい象限にあるか検証（逆順/入れ替えを検出）
            # tray(0,0)→左上, tray(1,0)→右上, tray(0,1)→左下, tray(1,1)→右下 になるはず
            cx_mean = float(test[:,0].mean())
            cy_mean = float(test[:,1].mean())
            lu, ru, ld, rd = test[0], test[1], test[2], test[3]  # src順: (0,0),(1,0),(0,1),(1,1)
            quadrant_ok = (
                lu[0] < cx_mean and lu[1] < cy_mean and  # lu: 左上
                ru[0] > cx_mean and ru[1] < cy_mean and  # ru: 右上
                ld[0] < cx_mean and ld[1] > cy_mean and  # ld: 左下
                rd[0] > cx_mean and rd[1] > cy_mean       # rd: 右下
            )
            if not quadrant_ok:
                logger.warning(
                    "corner H: quadrant mismatch (tray model label assignment is wrong) → rejecting, will use tray bbox"
                    "\n  lu→(%.0f,%.0f) ru→(%.0f,%.0f) ld→(%.0f,%.0f) rd→(%.0f,%.0f) center=(%.0f,%.0f)",
                    lu[0], lu[1], ru[0], ru[1], ld[0], ld[1], rd[0], rd[1], cx_mean, cy_mean)
                return None
        return H

    # 2-3 corners: similarity transform (fully determined with 2 correspondences)
    M, _ = cv2.estimateAffinePartial2D(
        src_arr.reshape(-1, 1, 2),
        dst_arr.reshape(-1, 1, 2),
        method=cv2.RANSAC,
    )
    if M is None:
        return None
    # Embed 2×3 affine into 3×3 homogeneous for perspectiveTransform compatibility
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M.astype(np.float64)
    logger.info("corner transform: similarity from %d corners (scale+rot+trans)", len(src))
    return H


def _slots_from_homography(layout_items: list, H: np.ndarray) -> List[_Slot]:
    """Transform tray-relative layout items → pixel _Slot objects via homography H."""
    if not layout_items or not _HAS_CV2:
        return []
    slots = []
    for i, item in enumerate(layout_items):
        cx, cy, w, h = item["cx"], item["cy"], item["w"], item["h"]
        # Transform bounding box corners to get pixel-space box
        pts = np.array([[
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx - w / 2, cy + h / 2],
            [cx + w / 2, cy + h / 2],
        ]], dtype=np.float32)
        pts_px = cv2.perspectiveTransform(pts, H)[0]
        x1 = float(pts_px[:, 0].min())
        y1 = float(pts_px[:, 1].min())
        x2 = float(pts_px[:, 0].max())
        y2 = float(pts_px[:, 1].max())
        slots.append(_Slot(str(i), x1, y1, x2, y2, item.get("class_name", "")))
    return slots


def _normalize_to_tray(detections: list, tray_box: dict) -> list:
    """Convert image-relative normalized coords → tray-relative normalized coords."""
    tx1 = tray_box["cx"] - tray_box["w"] / 2
    ty1 = tray_box["cy"] - tray_box["h"] / 2
    tw, th = tray_box["w"], tray_box["h"]
    return [{
        **det,
        "cx": (det["cx"] - tx1) / tw,
        "cy": (det["cy"] - ty1) / th,
        "w":  det["w"] / tw,
        "h":  det["h"] / th,
    } for det in detections]


def _tray_relative_to_image(layout_items: list, tray_box: dict) -> list:
    """Convert tray-relative layout → image-relative normalized coords."""
    tx1 = tray_box["cx"] - tray_box["w"] / 2
    ty1 = tray_box["cy"] - tray_box["h"] / 2
    tw, th = tray_box["w"], tray_box["h"]
    return [{
        **item,
        "cx": item["cx"] * tw + tx1,
        "cy": item["cy"] * th + ty1,
        "w":  item["w"] * tw,
        "h":  item["h"] * th,
    } for item in layout_items]

# ── RANSAC layout alignment (V36) ─────────────────────────────────────────────

def _align_layout_to_detections(slots: List[_Slot], dets: List[_Det]) -> List[_Slot]:
    """Align layout slots to detections via RANSAC similarity transform (V36)."""
    if not slots or not dets:
        return slots

    dets_sorted = sorted(dets, key=lambda d: d.confidence, reverse=True)
    dets_hi = [d for d in dets_sorted if d.confidence >= ALIGN_MIN_CONF]
    dets_for_align = dets_hi if len(dets_hi) >= 2 else dets_sorted[:min(10, len(dets_sorted))]

    src_pts: List[List[float]] = []
    dst_pts: List[List[float]] = []
    used: set = set()

    for sl in slots:
        det = None
        if sl.class_name:
            # Prefer nearest same-class detection for better geometric spread
            cands = [d for d in dets_for_align if d.label == sl.class_name and id(d) not in used]
            if cands:
                det = min(cands, key=lambda d: _dist(d.center, sl.center))
        if det is not None:
            used.add(id(det))
            src_pts.append([sl.center_x, sl.center_y])
            dst_pts.append([det.center_x, det.center_y])

    # クラス名一致ペアが不十分な場合は近傍マッチにフォールバック
    # クラス名体系が異なる場合（CLASS_CHECK無効時など）でも位置補正を可能にする
    if len(src_pts) < 2 and dets_for_align:
        logger.info("align: class-match %d pairs → nearest-neighbor fallback", len(src_pts))
        src_pts, dst_pts = [], []
        used_nn: set = set()
        for sl in slots:
            cands = [d for d in dets_for_align if id(d) not in used_nn]
            if not cands:
                break
            det = min(cands, key=lambda d: _dist(d.center, sl.center))
            used_nn.add(id(det))
            src_pts.append([sl.center_x, sl.center_y])
            dst_pts.append([det.center_x, det.center_y])
        logger.info("align: nearest-neighbor gave %d pairs", len(src_pts))

    # Adaptive RANSAC threshold: ~40% of average slot size, clamped
    if slots:
        avg_slot_px = float(np.mean([max(s.width, s.height) for s in slots]))
        ransac_thr = float(np.clip(avg_slot_px * 0.4, 15.0, 120.0))
    else:
        ransac_thr = 50.0

    src_arr = np.array(src_pts, dtype=np.float32)
    dst_arr = np.array(dst_pts, dtype=np.float32)

    # --- Try homography first (handles perspective distortion from camera angle) ---
    H: Optional[np.ndarray] = None
    if _HAS_CV2 and len(src_pts) >= 4:
        try:
            H_raw, hmask = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, ransac_thr)
            if H_raw is not None:
                n_in = int(np.sum(hmask)) if hmask is not None else 0
                inlier_ratio = n_in / len(src_pts) if src_pts else 0.0
                logger.info("Homography align: inliers=%d/%d (ratio=%.2f)", n_in, len(src_pts), inlier_ratio)
                if n_in >= 3 and inlier_ratio >= 0.50:
                    H = H_raw.astype(np.float64)
                else:
                    logger.warning("Homography unreliable (inlier ratio %.2f < 0.50) → raw layout", inlier_ratio)
        except Exception as e:
            logger.warning("Homography failed: %s → similarity fallback", e)

    if H is not None:
        # Apply homography: transform each slot center; estimate local scale via Jacobian
        centers = np.array([[s.center_x, s.center_y] for s in slots],
                            dtype=np.float32).reshape(-1, 1, 2)
        new_centers = cv2.perspectiveTransform(centers, H).reshape(-1, 2)
        eps = 1.0
        aligned: List[_Slot] = []
        for i, sl in enumerate(slots):
            nx, ny = float(new_centers[i, 0]), float(new_centers[i, 1])
            # Local scale from finite-difference Jacobian at this slot's center
            ptx = cv2.perspectiveTransform(
                np.array([[[sl.center_x + eps, sl.center_y]]], dtype=np.float32), H)[0][0]
            pty = cv2.perspectiveTransform(
                np.array([[[sl.center_x, sl.center_y + eps]]], dtype=np.float32), H)[0][0]
            sx = float(math.hypot(ptx[0] - nx, ptx[1] - ny)) / eps
            sy = float(math.hypot(pty[0] - nx, pty[1] - ny)) / eps
            nw = sl.width * sx
            nh = sl.height * sy
            aligned.append(_Slot(sl.slot_id, nx - nw / 2, ny - nh / 2,
                                  nx + nw / 2, ny + nh / 2, sl.class_name))
        return aligned

    # --- Fallback: RANSAC similarity transform (handles scale + rotation) ---
    M: Optional[np.ndarray] = None
    if _HAS_CV2 and len(src_pts) >= 2:
        try:
            M_sim, inliers = cv2.estimateAffinePartial2D(
                src_arr, dst_arr,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thr,
            )
            if M_sim is not None:
                scale = math.sqrt(float(M_sim[0, 0] ** 2 + M_sim[0, 1] ** 2))
                n_in = int(np.sum(inliers)) if inliers is not None else 0
                inlier_ratio = n_in / len(src_pts) if src_pts else 0.0
                logger.info("Similarity align: scale=%.3f inliers=%d/%d (ratio=%.2f)", scale, n_in, len(src_pts), inlier_ratio)
                if ALIGN_SCALE_MIN <= scale <= ALIGN_SCALE_MAX and n_in >= 2 and inlier_ratio >= 0.50:
                    M = M_sim.astype(np.float32)
                elif not (ALIGN_SCALE_MIN <= scale <= ALIGN_SCALE_MAX):
                    logger.warning("Similarity scale %.3f out of range → centroid fallback", scale)
                else:
                    logger.warning("Similarity unreliable (inlier ratio %.2f < 0.50) → raw layout", inlier_ratio)
        except Exception as e:
            logger.warning("Similarity RANSAC failed: %s → centroid fallback", e)

    if M is None:
        # RANSAC 失敗 → レイアウト座標をそのまま使う（重心シフトは不確実性が高いため適用しない）
        logger.warning("Alignment failed: returning raw layout slots without transform")
        return slots

    scale = math.sqrt(float(M[0, 0] ** 2 + M[0, 1] ** 2))
    aligned_sim: List[_Slot] = []
    for sl in slots:
        nx = float(M[0, 0] * sl.center_x + M[0, 1] * sl.center_y + M[0, 2])
        ny = float(M[1, 0] * sl.center_x + M[1, 1] * sl.center_y + M[1, 2])
        nw = sl.width * scale
        nh = sl.height * scale
        aligned_sim.append(_Slot(sl.slot_id, nx - nw / 2, ny - nh / 2,
                                  nx + nw / 2, ny + nh / 2, sl.class_name))
    return aligned_sim

# ── Layout gate (V36) ─────────────────────────────────────────────────────────

def _gate_detections_by_layout(dets: List[_Det], slots: List[_Slot]) -> List[_Det]:
    """Drop weak detections that are far from any layout slot."""
    if not LAYOUT_GATE_ENABLE or not dets or not slots:
        return dets

    req_conf = max((getattr(d, "requested_conf", 0.0) for d in dets), default=0.0)
    weak_min = max(GATE_MIN_CONF, req_conf * GATE_CONF_FACTOR)
    slot_centers = [s.center for s in slots]
    slot_gates = [
        _clamp(max(s.width, s.height) * GATE_DIST_RATIO, GATE_DIST_MIN, GATE_DIST_MAX)
        for s in slots
    ]

    kept: List[_Det] = []
    dropped = 0
    for det in dets:
        if det.confidence >= req_conf:
            kept.append(det)
            continue
        if det.confidence < weak_min:
            dropped += 1
            continue
        dists = [_dist(det.center, sc) for sc in slot_centers]
        best_i = min(range(len(dists)), key=lambda i: dists[i])
        if dists[best_i] <= slot_gates[best_i]:
            kept.append(det)
        else:
            dropped += 1

    if dropped:
        logger.info("layout-gate: kept=%d dropped=%d", len(kept), dropped)

    # ゲートで全検出が除外された場合はゲートなしで全検出を通す（フォールバック）
    # これにより RANSAC 失敗時でも matching に検出を渡せる
    if not kept and dets:
        logger.warning("layout-gate: all detections dropped → bypassing gate (fallback)")
        return [d for d in dets if d.confidence >= weak_min]
    return kept

# ── Greedy matching ───────────────────────────────────────────────────────────

def _match_greedy(dets: List[_Det], slots: List[_Slot]) -> List[_MatchResult]:
    """1-det-per-slot greedy match with class-label priority, then -IoU, then distance."""
    if not slots:
        return [_MatchResult(None, d, 0.0, "extra") for d in dets]
    if not dets:
        return [_MatchResult(s, None, 0.0, "missing") for s in slots]

    candidates: List[Tuple] = []
    for si, sl in enumerate(slots):
        # 閾値を広くしてアライメント誤差に対してロバストにする
        dist_thr = max(80.0, min(480.0, max(sl.width, sl.height) * 2.0))
        for di, det in enumerate(dets):
            iou = _iou_xyxy(sl.as_xyxy(), det.as_xyxy())
            dist = _dist(sl.center, det.center)
            if iou <= 0.0 and dist > dist_thr:
                continue
            # class_tier=0 when labels match (preferred); 1 when they differ
            class_tier = 0 if (sl.class_name and det.label == sl.class_name) else 1
            # Within same tier, rank by combined score (IoU + label bonus over distance)
            score = iou + (LABEL_MATCH_BONUS if class_tier == 0 else 0.0)
            candidates.append((class_tier, -score, dist, si, di, iou))

    candidates.sort(key=lambda t: (t[0], t[1], t[2]))

    used_slots: set = set()
    used_dets: set = set()
    slot_to_det: Dict[int, Tuple[int, float]] = {}
    for _, _, _, si, di, iou in candidates:
        if si in used_slots or di in used_dets:
            continue
        used_slots.add(si)
        used_dets.add(di)
        slot_to_det[si] = (di, float(iou))

    results: List[_MatchResult] = []
    for si, sl in enumerate(slots):
        if si not in slot_to_det:
            results.append(_MatchResult(sl, None, 0.0, "missing"))
        else:
            di, iou = slot_to_det[si]
            results.append(_MatchResult(sl, dets[di], float(iou), "ok"))

    for di, det in enumerate(dets):
        if di not in used_dets:
            results.append(_MatchResult(None, det, 0.0, "extra"))

    return results

# ── Slot presence check ───────────────────────────────────────────────────────

def _apply_slot_min_conf(mrs: List[_MatchResult]) -> List[_MatchResult]:
    """ok マッチでも conf < SLOT_MIN_CONF のものは missing に降格する。
    空トレイで低 conf の背景ノイズが拾われる誤検出を防ぐ。
    SLOT_MIN_CONF=0.0 の場合は何もしない。
    """
    if SLOT_MIN_CONF <= 0.0:
        return mrs
    result: List[_MatchResult] = []
    demoted = 0
    for mr in mrs:
        if (mr.status == "ok"
                and mr.det is not None
                and mr.det.confidence < SLOT_MIN_CONF):
            result.append(_MatchResult(mr.slot, None, 0.0, "missing"))
            demoted += 1
        else:
            result.append(mr)
    if demoted:
        logger.info("slot_min_conf=%.4f: %d ok→missing (conf too low)", SLOT_MIN_CONF, demoted)
    return result


# ── Match quality classification (yellow / misplaced) ────────────────────────

def _classify_match_quality(mrs: List[_MatchResult]) -> List[_MatchResult]:
    """Reclassify 'ok' matches as 'misplaced' (yellow) when the wrong product class
    is detected at a slot position.

    サイズ・IoU による判定は除外:
      アノテーションベースのレイアウトはスロットサイズが均一な固定値であり
      実製品の大きさを表していない。YOLO 検出ボックスとのスケール差は常に大きく
      なるため false positive を大量発生させる。
    判定条件: スロットに期待するクラスと異なるクラスが検出された場合のみ黄色。
    ただし、スロット側のクラス名がモデルの出力するいずれのラベルとも一致しない場合は
    クラス名比較を行わない（モデルとレイアウトのクラス名体系が異なる場合の誤判定防止）。
    """
    # モデルが出力しているラベルセットを収集
    det_labels: set = {mr.det.label for mr in mrs if mr.det is not None}
    slot_classes: set = {mr.slot.class_name for mr in mrs if mr.slot and mr.slot.class_name}
    # スロットのクラス名がいずれも検出ラベルに存在しない場合はクラス比較をスキップ
    class_check_enabled = CLASS_CHECK_ENABLE and bool(slot_classes & det_labels)
    logger.info("classify_quality: det_labels=%s slot_classes=%s class_check=%s",
                sorted(det_labels), sorted(slot_classes), class_check_enabled)

    result: List[_MatchResult] = []
    n_class = 0
    for mr in mrs:
        if mr.status != "ok" or mr.det is None or mr.slot is None:
            result.append(mr)
            continue

        if class_check_enabled and mr.slot.class_name and mr.det.label != mr.slot.class_name:
            logger.info("  misplaced: slot_class=%r det_label=%r iou=%.3f", mr.slot.class_name, mr.det.label, mr.iou)
            result.append(_MatchResult(mr.slot, mr.det, mr.iou, "misplaced"))
            n_class += 1
        else:
            result.append(mr)

    if n_class:
        logger.info("classify_quality: %d ok→misplaced (class mismatch)", n_class)
    return result


# ── Extra filtering (V36) ─────────────────────────────────────────────────────

def _filter_extras(mrs: List[_MatchResult]) -> List[_MatchResult]:
    """Remove low-conf and redundant extra detections."""
    non_extra = [m for m in mrs if m.status != "extra"]
    extras = [m for m in mrs if m.status == "extra"]
    if not extras:
        return mrs

    req_conf = next(
        (float(m.det.requested_conf) for m in mrs if m.det is not None and hasattr(m.det, "requested_conf")),
        0.0,
    )

    try:
        min_conf = float(EXTRA_MIN_CONF_ENV) if EXTRA_MIN_CONF_ENV else req_conf
    except Exception:
        min_conf = req_conf

    # 1) Drop below min_conf
    extras2 = [m for m in extras if m.det is not None and m.det.confidence >= min_conf]

    # 2) Drop extras that strongly overlap with matched detections
    if EXTRA_SUPPRESS_IOU > 0:
        matched_boxes = [m.det.as_xyxy() for m in non_extra if m.det is not None]
        deduped: List[_MatchResult] = []
        for m in extras2:
            if m.det is None:
                continue
            overlap = any(_iou_xyxy(m.det.as_xyxy(), mb) >= EXTRA_SUPPRESS_IOU for mb in matched_boxes)
            if not overlap:
                deduped.append(m)
        extras2 = deduped

    # 3) Keep top-K by confidence
    if EXTRA_KEEP_TOPK > 0 and len(extras2) > EXTRA_KEEP_TOPK:
        extras2 = sorted(extras2, key=lambda m: m.det.confidence if m.det else 0.0, reverse=True)[:EXTRA_KEEP_TOPK]

    if len(extras2) != len(extras):
        logger.info("extra filter: %d → %d (min_conf=%.4f suppress_iou=%.2f topk=%d)",
                    len(extras), len(extras2), min_conf, EXTRA_SUPPRESS_IOU, EXTRA_KEEP_TOPK)

    return non_extra + extras2

# ── Layout generation helpers ─────────────────────────────────────────────────

def _spatial_filter_dense(dets: List[_Det], top_frac: float = 0.5, pad: float = 0.30) -> List[_Det]:
    """高信頼度の検出物の重心領域に含まれる検出物だけを返す（孤立した FP を除去）。
    top_frac: 上位何割をアンカーとして使うか
    pad: バウンディングボックスに加えるパディング比率
    """
    if len(dets) < 4:
        return dets
    n = max(4, int(len(dets) * top_frac))
    anchors = sorted(dets, key=lambda d: d.confidence, reverse=True)[:n]
    x1 = min(d.x1 for d in anchors)
    y1 = min(d.y1 for d in anchors)
    x2 = max(d.x2 for d in anchors)
    y2 = max(d.y2 for d in anchors)
    dx = (x2 - x1) * pad
    dy = (y2 - y1) * pad
    kept = [d for d in dets
            if x1 - dx <= d.center_x <= x2 + dx and y1 - dy <= d.center_y <= y2 + dy]
    logger.info("spatial_filter_dense: %d → %d (anchor bbox with pad=%.2f)", len(dets), len(kept), pad)
    return kept


def _spatial_dedup(dets: List[_Det], diag_frac: float = 1.0) -> List[_Det]:
    """クラスをまたいだ空間的重複排除（YOLO の per-class NMS では除去できない重複を解消）。
    信頼度の高い検出を優先し、中心距離がボックス対角の diag_frac 以内の検出を除去する。
    diag_frac=1.0: ボックス対角と同距離以内を重複とみなす（デフォルト）
    """
    if len(dets) <= 1:
        return dets
    diags = [math.hypot(d.x2 - d.x1, d.y2 - d.y1) for d in dets]
    thresh = float(np.median(diags)) * diag_frac
    sorted_dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: List[_Det] = []
    for det in sorted_dets:
        if not any(_dist(det.center, k.center) < thresh for k in kept):
            kept.append(det)
    logger.info("spatial_dedup: %d → %d (center_thresh=%.1fpx, diag_frac=%.1f)", len(dets), len(kept), thresh, diag_frac)
    return kept

# ── Annotation-based layout generation ───────────────────────────────────────

# トレイ隅クラスのラベルとトレイ基準フレーム内の位置の対応
_CORNER_LABEL_TO_TRAY: Dict[str, Tuple[float, float]] = {
    "tray_lu": (0.0, 0.0), "tray-lu": (0.0, 0.0),
    "tray_ru": (1.0, 0.0), "tray-ru": (1.0, 0.0),
    "tray_ld": (0.0, 1.0), "tray-ld": (0.0, 1.0),
    "tray_rd": (1.0, 1.0), "tray-rd": (1.0, 1.0),
}


def generate_layout_from_annotations_direct(
    image_label_pairs: List[Tuple[str, str]],
    training_classes: List[Dict],
) -> List[Dict]:
    """アノテーションファイルから画像相対座標のレイアウトを直接生成。

    トレイモデル・ホモグラフィ変換不要。
    YOLO アノテーションの cx/cy/w/h をそのままクラスごとに平均する。
    学習画像が検査カメラで撮影されている（固定カメラ）前提。
    """
    product_idx_to_name: Dict[int, str] = {}
    for tc in training_classes:
        idx, name = tc["class_index"], tc["name"]
        norm = name.replace("-", "_").lower()
        if norm not in _CORNER_LABEL_TO_TRAY:
            product_idx_to_name[idx] = name

    if not product_idx_to_name:
        raise ValueError("製品クラスが TrainingClass に見つかりません")

    product_positions: Dict[int, List[Tuple[float, float, float, float]]] = {}

    for img_path, label_path in image_label_pairs:
        lp = Path(label_path)
        if not lp.exists():
            continue
        for line in open(lp):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            if cid not in product_idx_to_name:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            if cid not in product_positions:
                product_positions[cid] = []
            product_positions[cid].append((cx, cy, w, h))

    if not product_positions:
        return []

    items: List[Dict] = []
    for cid in sorted(product_positions.keys()):
        positions = product_positions[cid]
        avg_cx = float(np.mean([p[0] for p in positions]))
        avg_cy = float(np.mean([p[1] for p in positions]))
        avg_w  = float(np.mean([p[2] for p in positions]))
        avg_h  = float(np.mean([p[3] for p in positions]))
        items.append({
            "class_name": product_idx_to_name[cid],
            "cx": avg_cx, "cy": avg_cy,
            "w": max(avg_w, 0.02), "h": max(avg_h, 0.02),
        })
        logger.info("direct_layout[%d] class=%s cx=%.3f cy=%.3f w=%.3f h=%.3f",
                    cid, product_idx_to_name[cid], avg_cx, avg_cy, avg_w, avg_h)

    logger.info("generate_layout_from_annotations_direct: %d items from %d image pairs",
                len(items), len(image_label_pairs))
    return items


def generate_layout_from_annotations(
    image_label_pairs: List[Tuple[str, str]],
    training_classes: List[Dict],
) -> List[Dict]:
    """アノテーションファイルからトレイ相対レイアウトを生成する。

    image_label_pairs: [(image_path, label_path), ...]
    training_classes:  [{"class_index": int, "name": str}, ...]

    アルゴリズム:
      1. TrainingClass からトレイ隅クラスと製品クラスを判別
      2. 各学習画像のアノテーションを読み込み、隅位置からホモグラフィを計算
      3. 製品位置を画像座標 → トレイ相対座標に変換
      4. クラスごとに平均し安定したレイアウトを生成
      5. グリッド間隔からバウンディングボックスサイズを推定
    """
    # TrainingClass からコーナークラスと製品クラスを分類
    corner_idx_to_tray: Dict[int, Tuple[float, float]] = {}
    product_idx_to_name: Dict[int, str] = {}
    for tc in training_classes:
        idx, name = tc["class_index"], tc["name"]
        norm = name.replace("-", "_").lower()
        if norm in _CORNER_LABEL_TO_TRAY:
            corner_idx_to_tray[idx] = _CORNER_LABEL_TO_TRAY[norm]
        else:
            product_idx_to_name[idx] = name

    if not corner_idx_to_tray:
        raise ValueError("トレイ隅クラス（tray_lu/ru/ld/rd）が TrainingClass に見つかりません")
    if not product_idx_to_name:
        raise ValueError("製品クラスが TrainingClass に見つかりません")

    product_tray_positions: Dict[int, List[Tuple[float, float]]] = {}

    for img_path, label_path in image_label_pairs:
        lp = Path(label_path)
        if not lp.exists():
            continue

        corners_img: Dict[Tuple[float, float], Tuple[float, float]] = {}  # tray_ref → img_pos
        products_img: List[Tuple[int, float, float]] = []

        for line in open(lp):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cid, cx, cy = int(parts[0]), float(parts[1]), float(parts[2])
            if cid in corner_idx_to_tray:
                corners_img[corner_idx_to_tray[cid]] = (cx, cy)
            elif cid in product_idx_to_name:
                products_img.append((cid, cx, cy))

        if len(corners_img) < 3 or not products_img:
            logger.debug("generate_layout_from_annotations: skip %s (corners=%d products=%d)",
                         img_path, len(corners_img), len(products_img))
            continue

        # 3 corners → estimate 4th using parallelogram rule (tray is rectangular)
        if len(corners_img) == 3:
            all_c = {(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)}
            missing_set = all_c - set(corners_img.keys())
            if len(missing_set) == 1:
                mx, my = missing_set.pop()
                adj1 = (1.0 - mx, my)
                adj2 = (mx, 1.0 - my)
                opp  = (1.0 - mx, 1.0 - my)
                if adj1 in corners_img and adj2 in corners_img and opp in corners_img:
                    ex = corners_img[adj1][0] + corners_img[adj2][0] - corners_img[opp][0]
                    ey = corners_img[adj1][1] + corners_img[adj2][1] - corners_img[opp][1]
                    corners_img[(mx, my)] = (ex, ey)
                    logger.info("parallelogram: estimated corner (%.0f,%.0f) → img(%.3f,%.3f)", mx, my, ex, ey)

        # Validate: detected corners must span a reasonable area of the image
        corner_vals = list(corners_img.values())
        xs_c = [p[0] for p in corner_vals]
        ys_c = [p[1] for p in corner_vals]
        span_x, span_y = max(xs_c) - min(xs_c), max(ys_c) - min(ys_c)
        if span_x < 0.12 or span_y < 0.12:
            logger.warning("generate_layout_from_annotations: skip %s — corner span too small (x=%.3f y=%.3f)",
                           lp.name, span_x, span_y)
            continue

        logger.info("generate_layout_from_annotations: %s  corners=%d span=(%.3f,%.3f)",
                    lp.name, len(corners_img), span_x, span_y)

        # Compute H: image [0,1] → tray [0,1]² directly (avoids inversion error)
        H_direct: Optional[np.ndarray] = None
        if _HAS_CV2 and len(corners_img) >= 4:
            src_h = np.array([list(v) for v in corners_img.values()], dtype=np.float32)  # image coords
            dst_h = np.array([list(k) for k in corners_img.keys()], dtype=np.float32)    # tray ref coords
            H_d, _ = cv2.findHomography(src_h, dst_h, cv2.RANSAC, 0.02)  # threshold in normalized [0,1] space
            if H_d is not None:
                H_direct = H_d.astype(np.float64)

        for cid, img_cx, img_cy in products_img:
            if H_direct is not None:
                pt = np.array([[[img_cx, img_cy]]], dtype=np.float32)
                tray_pt = cv2.perspectiveTransform(pt, H_direct.astype(np.float32))[0][0]
                tray_cx, tray_cy = float(tray_pt[0]), float(tray_pt[1])
            else:
                # Fallback: bounding-box scaling using available corner positions
                tw = span_x if span_x > 0.01 else 1.0
                th = span_y if span_y > 0.01 else 1.0
                tray_cx = (img_cx - min(xs_c)) / tw
                tray_cy = (img_cy - min(ys_c)) / th

            # Discard positions clearly outside the tray frame
            if not (-0.1 <= tray_cx <= 1.1 and -0.1 <= tray_cy <= 1.1):
                logger.warning("generate_layout_from_annotations: product %d pos (%.3f,%.3f) out of tray in %s — skipping",
                               cid, tray_cx, tray_cy, lp.name)
                continue

            if cid not in product_tray_positions:
                product_tray_positions[cid] = []
            product_tray_positions[cid].append((tray_cx, tray_cy))

    if not product_tray_positions:
        return []

    items: List[Dict] = []
    for cid in sorted(product_tray_positions.keys()):
        positions = product_tray_positions[cid]
        avg_cx = float(np.mean([p[0] for p in positions]))
        avg_cy = float(np.mean([p[1] for p in positions]))
        items.append({
            "class_name": product_idx_to_name[cid],
            "cx": avg_cx,
            "cy": avg_cy,
        })

    # グリッド間隔からバウンディングボックスサイズを推定
    if len(items) >= 2:
        all_cx_sorted = sorted(it["cx"] for it in items)
        all_cy_sorted = sorted(it["cy"] for it in items)
        _GAP_THRESH = 0.04
        cx_gaps = [all_cx_sorted[i+1] - all_cx_sorted[i] for i in range(len(all_cx_sorted)-1)]
        cy_gaps = [all_cy_sorted[i+1] - all_cy_sorted[i] for i in range(len(all_cy_sorted)-1)]
        dx = min((g for g in cx_gaps if g > _GAP_THRESH), default=0.20)
        dy = min((g for g in cy_gaps if g > _GAP_THRESH), default=0.20)
        default_w = max(0.06, dx * 0.85)
        default_h = max(0.06, dy * 0.85)
    else:
        default_w, default_h = 0.12, 0.15

    for item in items:
        item["w"] = default_w
        item["h"] = default_h

    logger.info("generate_layout_from_annotations: %d items from %d images",
                len(items), len(image_label_pairs))
    for i, it in enumerate(items):
        logger.info("  layout[%d] class=%s  cx=%.3f cy=%.3f  w=%.3f h=%.3f",
                    i, it["class_name"], it["cx"], it["cy"], it["w"], it["h"])
    return items


# ── Public API ────────────────────────────────────────────────────────────────

_STATUS_COLOR = {"ok": "green", "missing": "red", "extra": "#4af", "misplaced": "yellow"}
_STATUS_TYPE  = {"ok": "correct", "missing": "missing", "extra": "extra", "misplaced": "misplaced"}


def _crop_to_tray(pil_img: Image.Image, tray_box: dict) -> Tuple[Image.Image, int, int]:
    """Crop image to tray region with padding; returns (crop, off_x, off_y)."""
    img_w, img_h = pil_img.size
    pad_x = tray_box["w"] * TRAY_CROP_PAD
    pad_y = tray_box["h"] * TRAY_CROP_PAD
    tx1 = max(0, int((tray_box["cx"] - tray_box["w"] / 2 - pad_x) * img_w))
    ty1 = max(0, int((tray_box["cy"] - tray_box["h"] / 2 - pad_y) * img_h))
    tx2 = min(img_w, int((tray_box["cx"] + tray_box["w"] / 2 + pad_x) * img_w))
    ty2 = min(img_h, int((tray_box["cy"] + tray_box["h"] / 2 + pad_y) * img_h))
    return pil_img.crop((tx1, ty1, tx2, ty2)), tx1, ty1


def _run_yolo_in_tray(model_path: str, pil_img: Image.Image, tray_box: dict, req_conf: float,
                      exact_conf: bool = False) -> List[_Det]:
    """Run product YOLO only on the tray region and translate coords back to full image."""
    crop, off_x, off_y = _crop_to_tray(pil_img, tray_box)
    dets_crop = _run_yolo(model_path, crop, req_conf, exact_conf=exact_conf)
    return [
        _Det(d.label, d.class_id, d.confidence,
             d.x1 + off_x, d.y1 + off_y,
             d.x2 + off_x, d.y2 + off_y,
             d.requested_conf)
        for d in dets_crop
    ]


def run_detection(
    product_model_path: str,
    layout_path: Optional[str],
    image_path: str,
    tray_model_path: Optional[str] = None,
) -> dict:
    pil_img = Image.open(image_path).convert("RGB")
    img_w, img_h = pil_img.size

    # 1. Load layout first — must know is_tray_relative before deciding detection strategy
    layout_items, is_tray_relative = (
        parse_layout(layout_path)
        if layout_path and Path(layout_path).exists()
        else ([], True)
    )

    # 2. Detect tray (normalized coords)
    tray_box = None
    tray_configured = bool(tray_model_path and Path(tray_model_path).exists())
    if tray_configured:
        tray_box = _detect_tray(tray_model_path, image_path)

    # 3. Detect products
    #    image-relative layout → full image detection (tray crop would misalign coords)
    #    tray-relative layout  → confine to tray region to reduce background noise
    if is_tray_relative and tray_box:
        dets = _run_yolo_in_tray(product_model_path, pil_img, tray_box, DEFAULT_CONF)
        # 3c. tray cropで検出0件の場合は全画像で再推論（トレイ位置推定誤りを補完）
        if not dets:
            logger.info("tray_crop: 0 dets → full-image detection fallback")
            dets = _run_yolo(product_model_path, pil_img, req_conf=DEFAULT_CONF)
            if dets:
                logger.info("full-image fallback: %d dets (tray bounds filter to follow)", len(dets))
    else:
        dets = _run_yolo(product_model_path, pil_img, req_conf=DEFAULT_CONF)

    # 3b-pre. Tray bounds filter: トレイが検出されている場合、トレイ外の検出を除外する
    # image_relative レイアウトは全画像で推論するためトレイ外の誤検出が発生しやすい
    if dets and tray_box:
        tx1_px = (tray_box["cx"] - tray_box["w"] / 2) * img_w
        ty1_px = (tray_box["cy"] - tray_box["h"] / 2) * img_h
        tx2_px = (tray_box["cx"] + tray_box["w"] / 2) * img_w
        ty2_px = (tray_box["cy"] + tray_box["h"] / 2) * img_h
        before = len(dets)
        dets = [d for d in dets if tx1_px <= d.center_x <= tx2_px and ty1_px <= d.center_y <= ty2_px]
        if len(dets) < before:
            logger.info("tray bounds filter: %d → %d dets (%d outside tray removed)",
                        before, len(dets), before - len(dets))

    # 3b. Per-class dedup + layout-class filter
    #     (a) Only keep detections whose label exists in the layout (filters tray-corner bleed-in)
    #     (b) Keep only the highest-confidence detection per class label
    if dets and layout_items:
        layout_classes = {it["class_name"] for it in layout_items}
        dets_in_layout = [d for d in dets if d.label in layout_classes]
        removed_labels = {d.label for d in dets} - {d.label for d in dets_in_layout}
        if removed_labels:
            logger.info("layout-class filter: removed non-layout classes %s (%d dets)",
                        sorted(removed_labels), len(dets) - len(dets_in_layout))
        best_per_class: Dict[str, _Det] = {}
        for det in sorted(dets_in_layout, key=lambda d: d.confidence, reverse=True):
            if det.label not in best_per_class:
                best_per_class[det.label] = det
        dets_dedup = list(best_per_class.values())
        logger.info("per-class dedup: %d → %d dets (layout classes: %s)", len(dets), len(dets_dedup), sorted(layout_classes))
        dets = dets_dedup

    # ── アンカーゲート: 全検出の最大 conf が低すぎる場合は空トレイとみなす ────────
    if dets and layout_items and ANCHOR_CONF > 0.0:
        max_det_conf = max(d.confidence for d in dets)
        if max_det_conf < ANCHOR_CONF:
            logger.info(
                "anchor gate: max_conf=%.4f < %.4f → no meaningful detection, treating as empty tray",
                max_det_conf, ANCHOR_CONF,
            )
            dets = []

    # ── 均一性ゲート: 全検出の conf が均一（stdが小さい）かつ max が中程度以下 → 空トレイ誤検出とみなす ──
    # 実製品: 1クラスが突出して高conf(>0.5)で他は低い → 大きなばらつき
    # 空トレイ誤検出: 全クラスが似た中程度 conf (~0.4) → std が非常に小さい
    if (dets and layout_items
            and UNIFORM_GATE_MAX_STD > 0.0
            and len(dets) >= UNIFORM_GATE_MIN_DET):
        confs = [d.confidence for d in dets]
        max_conf_u = max(confs)
        if max_conf_u < UNIFORM_GATE_MAX_CONF:
            std_conf = float(np.std(confs))
            if std_conf < UNIFORM_GATE_MAX_STD:
                logger.info(
                    "uniform gate: n=%d max=%.3f std=%.3f < %.3f → uniform false-positive cluster, treating as empty tray",
                    len(dets), max_conf_u, std_conf, UNIFORM_GATE_MAX_STD,
                )
                dets = []

    # ── 診断ログ ──────────────────────────────────────────────────────────────
    logger.info("=== run_detection START ===")
    logger.info("image: %dx%d  tray_configured=%s  is_tray_relative=%s", img_w, img_h, tray_configured, is_tray_relative)
    logger.info("layout: %d items  path=%s", len(layout_items), layout_path)
    if layout_items:
        for i, it in enumerate(layout_items[:5]):
            logger.info("  layout[%d] class=%s cx=%.3f cy=%.3f w=%.3f h=%.3f",
                        i, it.get("class_name"), it["cx"], it["cy"], it["w"], it["h"])
    if tray_box:
        logger.info("tray_box: cx=%.3f cy=%.3f w=%.3f h=%.3f conf=%.3f corners=%s",
                    tray_box["cx"], tray_box["cy"], tray_box["w"], tray_box["h"],
                    tray_box.get("conf", 0), list(tray_box.get("corners", {}).keys()))
    else:
        logger.info("tray_box: NOT DETECTED")
    logger.info("detections: %d (strategy: %s)", len(dets), "tray_crop" if (is_tray_relative and tray_box) else "full_image")
    for i, d in enumerate(dets[:5]):
        logger.info("  det[%d] label=%s conf=%.3f center=(%.1f,%.1f) box=(%.1f,%.1f,%.1f,%.1f)",
                    i, d.label, d.confidence, d.center_x, d.center_y, d.x1, d.y1, d.x2, d.y2)
    # ─────────────────────────────────────────────────────────────────────────

    # 4. Build pixel-space slots
    #    座標系ルール:
    #      - is_tray_relative=True  (アノテーション生成・トレイあり参照画像生成): トレイ相対 [0,1]²
    #      - is_tray_relative=False (トレイ未検出の参照画像生成): 画像相対 [0,1]²
    #    優先度: コーナーホモグラフィ > トレイbbox変換 > 画像直接適用
    #    重要: トレイモデル設定済みでトレイ未検出の場合は layout を適用しない
    slots: List[_Slot] = []
    tray_applied = False  # True = トレイジオメトリでスロットを配置済み → RANSAC 不要

    if tray_box and layout_items:
        corners = tray_box.get("corners", {})
        H = _corner_homography(corners, img_w, img_h)
        if H is not None:
            # 4コーナーマーカーによる正確なホモグラフィでスロット配置
            # トレイ相対・画像相対いずれも適用（画像相対の場合はRANSACで追補正）
            slots = _slots_from_homography(layout_items, H)
            tray_applied = True
            logger.info("Corner homography: %d slots (is_tray_relative=%s)", len(slots), is_tray_relative)
        elif is_tray_relative:
            # トレイbbox: トレイ相対座標 → 画像相対 → ピクセル
            layout_items_abs = _tray_relative_to_image(layout_items, tray_box)
            slots = _layout_to_slots(layout_items_abs, img_w, img_h)
            tray_applied = True
            logger.info("Tray bbox (tray-relative layout): %d slots", len(slots))
        else:
            # 画像相対レイアウト (直接生成) → そのままピクセル変換 (tray_applied=False → RANSAC使用)
            slots = _layout_to_slots(layout_items, img_w, img_h)
            tray_applied = False
            logger.info("Image-relative layout (tray present but unused for slots): %d slots", len(slots))
    elif not tray_configured and layout_items:
        # トレイモデルなし: layout は画像相対座標
        slots = _layout_to_slots(layout_items, img_w, img_h)
        tray_applied = False
    elif not tray_box and not is_tray_relative and layout_items:
        # トレイモデルあり・トレイ未検出・画像相対レイアウト → トレイ不要なのでそのまま適用
        slots = _layout_to_slots(layout_items, img_w, img_h)
        tray_applied = False
        logger.info("Image-relative layout, tray not found: %d slots", len(slots))
    else:
        logger.warning("tray model configured but tray not detected — skipping layout matching")

    # ── 診断ログ（スロット構築後）────────────────────────────────────────────
    logger.info("slots built: %d  tray_applied=%s", len(slots), tray_applied)
    for i, s in enumerate(slots[:5]):
        logger.info("  slot[%d] class=%s center=(%.1f,%.1f) size=(%.1f,%.1f)",
                    i, s.class_name, s.center_x, s.center_y, s.width, s.height)
    # ─────────────────────────────────────────────────────────────────────────

    # 5. アライメント → ゲート → マッチング → extras フィルタ
    #    トレイ信頼度が高い場合はトレイジオメトリで確定したスロット位置をそのまま使う。
    #    信頼度が低い場合はRANSACで製品検出を元にスロット位置を補正する。
    #
    #    【表示位置の方針】
    #      ok/misplaced → RANSAC アライメント後の位置（実製品がいる場所）
    #      missing      → レイアウト元の位置（あるべき場所）  ← RANSAC で歪まない
    if slots and dets:
        # 検出物の実位置を使って常にRANSACでスロット位置を補正する
        # RANSACが信頼性不足と判断した場合は元のスロット位置をそのまま返す
        tray_conf_val = tray_box.get("conf", 1.0) if tray_box else 0.0
        aligned = _align_layout_to_detections(slots, dets)
        logger.info("RANSAC alignment applied (tray_applied=%s tray_conf=%.3f)", tray_applied, tray_conf_val)
        dets_gated = _gate_detections_by_layout(dets, aligned)
        match_results = _match_greedy(dets_gated, aligned)
        match_results = _filter_extras(match_results)
        match_results = _apply_slot_min_conf(match_results)
        match_results = _classify_match_quality(match_results)
    elif slots:
        match_results = [_MatchResult(s, None, 0.0, "missing") for s in slots]
    elif dets:
        # スロットなし（トレイ未検出等）: DEFAULT_CONF 以上の検出のみ余分として表示
        # INFER_MIN_CONF で推論しているため低信頼度ノイズが大量にある
        dets_vis = [d for d in dets if d.confidence >= DEFAULT_CONF]
        match_results = [_MatchResult(None, d, 0.0, "extra") for d in dets_vis]
    else:
        match_results = []

    # 7. Build output boxes (normalized [0,1])
    # 欠品/誤配置ボックスの表示サイズ参照: ok + misplaced すべてのマッチ済み検出から中央値を取る
    # (classify_match_quality が ok→misplaced に変換した後でも参照サイズが消えないよう inclusive に集計)
    matched_det_sizes = [
        ((mr.det.x2 - mr.det.x1) / img_w, (mr.det.y2 - mr.det.y1) / img_h)
        for mr in match_results
        if mr.status in ("ok", "misplaced") and mr.det is not None
    ]
    if matched_det_sizes:
        ref_w = float(np.median([s[0] for s in matched_det_sizes]))
        ref_h = float(np.median([s[1] for s in matched_det_sizes]))
    else:
        ref_w = ref_h = None

    boxes = []
    for mr in match_results:
        if mr.status == "missing" and mr.slot:
            disp_w = ref_w if ref_w else mr.slot.width / img_w
            disp_h = ref_h if ref_h else mr.slot.height / img_h
            boxes.append({
                "class_name": mr.slot.class_name,
                "cx": mr.slot.center_x / img_w,
                "cy": mr.slot.center_y / img_h,
                "w":  disp_w,
                "h":  disp_h,
                "conf": 0.0,
                "color": "red",
                "type": "missing",
            })
        elif mr.status == "misplaced" and mr.slot:
            # misplaced は常に mr.det が存在するので実検出サイズを使う
            if mr.det:
                disp_w = (mr.det.x2 - mr.det.x1) / img_w
                disp_h = (mr.det.y2 - mr.det.y1) / img_h
            else:
                disp_w = ref_w if ref_w else mr.slot.width / img_w
                disp_h = ref_h if ref_h else mr.slot.height / img_h
            boxes.append({
                "class_name": mr.slot.class_name,
                "cx": mr.slot.center_x / img_w,
                "cy": mr.slot.center_y / img_h,
                "w":  disp_w,
                "h":  disp_h,
                "conf": mr.det.confidence if mr.det else 0.0,
                "color": "yellow",
                "type": "misplaced",
            })
        elif mr.det is not None:
            boxes.append({
                "class_name": mr.det.label,
                "cx": mr.det.center_x / img_w,
                "cy": mr.det.center_y / img_h,
                "w":  (mr.det.x2 - mr.det.x1) / img_w,
                "h":  (mr.det.y2 - mr.det.y1) / img_h,
                "conf": mr.det.confidence,
                "color": _STATUS_COLOR.get(mr.status, "yellow"),
                "type": _STATUS_TYPE.get(mr.status, mr.status),
            })

    # ── Quality / retake check ────────────────────────────────────────────────
    retake_reasons: List[str] = []

    if is_tray_relative:
        # Tray-relative layout: tray detection quality matters for slot placement
        # 1. Tray model configured but tray not found
        if tray_model_path and Path(tray_model_path).exists() and tray_box is None:
            retake_reasons.append("トレイが画面内に見つかりませんでした。トレイをガイド枠に合わせて再撮影してください。")
        # 2. Tray detected with low confidence
        elif tray_box and tray_box.get("conf", 1.0) < QUALITY_MIN_TRAY_CONF:
            retake_reasons.append(
                f"トレイの検出精度が低いです（{tray_box['conf']:.0%}）。"
                "トレイ全体が映るよう位置を調整して再撮影してください。"
            )
    # image-relative layout: tray conf does not affect slot placement → no retake for tray

    # 3. Match rate check (informational only — does not trigger retake)
    if slots:
        n_ok = sum(1 for mr in match_results if mr.status in ("ok", "misplaced"))
        match_rate = n_ok / len(slots)
        logger.info("match_rate=%.2f (%d/%d ok+misplaced)", match_rate, n_ok, len(slots))

    # 4. Image blur check (on tray crop if available, else full image)
    if QUALITY_BLUR_ENABLE:
        check_img = pil_img
        if tray_box:
            try:
                check_img, _, _ = _crop_to_tray(pil_img, tray_box)
            except Exception:
                pass
        blur_score = _laplacian_variance(check_img)
        if blur_score < QUALITY_BLUR_THRESH:
            retake_reasons.append(
                f"画像がぼやけています（鮮明度: {blur_score:.0f}）。"
                "カメラを安定させて再撮影してください。"
            )

    retake_needed = len(retake_reasons) > 0
    if retake_needed:
        logger.info("retake_needed: %s", retake_reasons)

    orientation = "landscape" if img_w >= img_h else "portrait"
    n_ok = sum(1 for mr in match_results if mr.status == "ok")
    n_miss = sum(1 for mr in match_results if mr.status == "missing")
    n_extra = sum(1 for mr in match_results if mr.status == "extra")
    logger.info("run_detection done: image=%dx%d(%s) dets=%d slots=%d ok=%d miss=%d extra=%d",
                img_w, img_h, orientation, len(dets), len(slots), n_ok, n_miss, n_extra)

    return {
        "boxes": boxes,
        "layout": layout_items,
        "tray_detected": tray_box is not None,
        "item_count": len(dets),
        "slot_count": len(slots),
        "image_size": f"{img_w}×{img_h}({orientation[0]})",
        "retake_needed": retake_needed,
        "retake_reasons": retake_reasons,
    }


def generate_layout_from_image(
    product_model_path: str,
    image_path: str,
    conf: float = 0.3,
    tray_model_path: Optional[str] = None,
) -> list:
    """レイアウト自動生成。
    conf は最終的な信頼度フィルタに使うが、モデルが低信頼度の場合は
    空間密集フィルタを優先し、conf フィルタは緩やかに適用する。
    """
    pil_img = Image.open(image_path).convert("RGB")

    # ── 高解像度参照画像を事前リサイズ
    MAX_REF_SIDE = 2560
    if max(pil_img.size) > MAX_REF_SIDE:
        ratio = MAX_REF_SIDE / max(pil_img.size)
        pil_img = pil_img.resize(
            (int(pil_img.width * ratio), int(pil_img.height * ratio)), Image.LANCZOS
        )
        logger.info("generate_layout: ref image resized to %s", pil_img.size)

    img_w, img_h = pil_img.size

    # ── トレイ検出（PIL 画像を直接渡しリサイズ後を使う）
    tray_box = None
    if tray_model_path and Path(tray_model_path).exists():
        tray_box = _detect_tray(tray_model_path, pil_img)
        if tray_box:
            logger.info("generate_layout: tray OK cx=%.3f cy=%.3f w=%.3f h=%.3f corners=%d",
                        tray_box["cx"], tray_box["cy"], tray_box["w"], tray_box["h"],
                        len(tray_box.get("corners", {})))
        else:
            logger.warning("generate_layout: tray NOT detected → fall back to full-image + density filter")

    # ── 製品検出
    # トレイ信頼度が十分高い場合のみトレイ切り抜きで検出する。
    # 低信頼度の場合は全画像を run_detection と同じ方式（exact_conf=False / INFER_MIN_CONF）で推論し
    # 空間密集フィルタで製品密集領域に絞り込む。
    LAYOUT_TRAY_MIN_CONF = 0.20
    tray_conf_for_layout = tray_box.get("conf", 0.0) if tray_box else 0.0
    use_tray_for_det = tray_box and tray_conf_for_layout >= LAYOUT_TRAY_MIN_CONF

    if use_tray_for_det:
        infer_conf = max(conf, 0.01)
        dets = _run_yolo_in_tray(product_model_path, pil_img, tray_box, infer_conf, exact_conf=True)
        logger.info("generate_layout: %d raw dets within tray crop (conf=%.3f)", len(dets), infer_conf)
    else:
        if tray_box:
            logger.info("generate_layout: tray conf=%.3f < %.2f → full-image detection (exact_conf=False)",
                        tray_conf_for_layout, LAYOUT_TRAY_MIN_CONF)
        # exact_conf=False: YOLO を INFER_MIN_CONF で推論して後段フィルタで絞り込む（run_detection と同方式）
        dets = _run_yolo(product_model_path, pil_img, req_conf=max(conf, 0.01), exact_conf=False)
        logger.info("generate_layout: %d raw dets on full image", len(dets))
        if len(dets) > 5:
            dets = _spatial_filter_dense(dets)

    # ── 対角線が画像の短辺の 60% を超える巨大ボックスを除去（トレイ全体などの誤検出）
    max_diag = min(img_w, img_h) * 0.6
    dets = [d for d in dets if math.hypot(d.x2 - d.x1, d.y2 - d.y1) <= max_diag]
    logger.info("generate_layout: %d dets after large-box filter (max_diag=%.0fpx)", len(dets), max_diag)

    # ── クラスごとに最高信頼度の検出を1個だけ残す（per-class dedup）
    # 画像解像度に依存しないため、高解像度参照画像でも正しく動作する。
    # クラス情報がない場合（全検出が同一クラス）は空間重複排除にフォールバック。
    if dets:
        best_per_class: dict = {}
        for det in sorted(dets, key=lambda d: d.confidence, reverse=True):
            if det.label not in best_per_class:
                best_per_class[det.label] = det
        dets_class_dedup = list(best_per_class.values())
        logger.info("generate_layout: per-class dedup: %d → %d (classes: %s)",
                    len(dets), len(dets_class_dedup),
                    sorted(best_per_class.keys()))
        # クラス種類が少なすぎる（≤2）場合は空間重複排除も試みる
        if len(dets_class_dedup) <= 2 and len(dets) > len(dets_class_dedup):
            dets_spatial = _spatial_dedup(dets, diag_frac=0.4)
            if len(dets_spatial) > len(dets_class_dedup):
                logger.info("generate_layout: spatial dedup gave more items (%d > %d), using spatial",
                            len(dets_spatial), len(dets_class_dedup))
                dets = dets_spatial
            else:
                dets = dets_class_dedup
        else:
            dets = dets_class_dedup

    logger.info("generate_layout: %d final items", len(dets))

    items = [{
        "class_name": d.label,
        "cx": d.center_x / img_w,
        "cy": d.center_y / img_h,
        "w":  (d.x2 - d.x1) / img_w,
        "h":  (d.y2 - d.y1) / img_h,
        "conf": d.confidence,
    } for d in dets]

    # トレイ信頼度が十分高い場合のみ tray-relative 座標を使う
    # 信頼度が低い場合は image-relative 座標のままにする（RANSAC でアライメント）
    LAYOUT_TRAY_MIN_CONF = 0.20
    tray_conf_for_layout = tray_box.get("conf", 0.0) if tray_box else 0.0
    if tray_box and tray_conf_for_layout >= LAYOUT_TRAY_MIN_CONF:
        items = _normalize_to_tray(items, tray_box)
        before = len(items)
        items = [it for it in items if -0.05 <= it["cx"] <= 1.05 and -0.05 <= it["cy"] <= 1.05]
        if len(items) < before:
            logger.info("generate_layout: removed %d out-of-tray items", before - len(items))
        is_tray_relative = True
    else:
        if tray_box:
            logger.info("generate_layout: tray conf=%.3f < %.2f → keeping image-relative coords (tray unreliable)",
                        tray_conf_for_layout, LAYOUT_TRAY_MIN_CONF)
        is_tray_relative = False

    return items, is_tray_relative
