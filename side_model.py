from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
import json

PathLike = Union[str, Path]


@dataclass(frozen=True)
class SideConfig:
    thr: float
    roi_top: Tuple[float, float, float, float]
    roi_bottom: Tuple[float, float, float, float]
    method: str = "TM_CCOEFF_NORMED"


def _frac_roi_box(W: int, H: int, roi_frac: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    x1f, y1f, x2f, y2f = roi_frac
    x1, y1, x2, y2 = int(W * x1f), int(H * y1f), int(W * x2f), int(H * y2f)
    return x1, y1, x2, y2


def _get_cv2_method(method: str) -> int:
    m = method.upper().strip()
    if m in {"TM_CCOEFF_NORMED", "CCOEFF_NORMED"}:
        return cv2.TM_CCOEFF_NORMED
    if m in {"TM_CCORR_NORMED", "CCORR_NORMED"}:
        return cv2.TM_CCORR_NORMED
    if m in {"TM_SQDIFF_NORMED", "SQDIFF_NORMED"}:
        return cv2.TM_SQDIFF_NORMED
    raise ValueError(f"Unsupported matchTemplate method: {method}")


def _match_in_roi(
    img_gray: np.ndarray,
    template_gray: np.ndarray,
    roi_frac: Tuple[float, float, float, float],
    method: int,
) -> Tuple[float, Tuple[int, int, int, int]]:
    H, W = img_gray.shape[:2]
    rx1, ry1, rx2, ry2 = _frac_roi_box(W, H, roi_frac)
    roi = img_gray[ry1:ry2, rx1:rx2]

    th, tw = template_gray.shape[:2]
    if roi.shape[0] <= th or roi.shape[1] <= tw:
        raise ValueError(f"ROI too small: ROI={roi.shape}, template={template_gray.shape}")

    heat = cv2.matchTemplate(roi, template_gray, method)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(heat)
    if method == cv2.TM_SQDIFF_NORMED:
        score = float(1.0 - minVal)  # higher is better
        x, y = minLoc
    else:
        score = float(maxVal)
        x, y = maxLoc

    x1 = rx1 + x
    y1 = ry1 + y
    x2 = x1 + tw
    y2 = y1 + th
    return score, (x1, y1, x2, y2)


class SideClassifier:
    """
    Predict SIDE: RIGHT / LEFT / UNCERTAIN based on template matching location.
    RIGHT: delta >= thr (top_score - bottom_score)
    LEFT:  delta <= -thr
    """

    def __init__(self, template_gray: np.ndarray, config: SideConfig):
        if template_gray is None or template_gray.size == 0:
            raise ValueError("Empty template")
        if template_gray.ndim != 2:
            raise ValueError("Template must be grayscale (H,W)")
        self.template = template_gray
        self.cfg = config
        self.method = _get_cv2_method(config.method)

    @staticmethod
    def load(model_dir: PathLike) -> "SideClassifier":
        model_dir = Path(model_dir)
        template_path = model_dir / "template.png"
        config_path = model_dir / "config.json"

        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Cannot read template: {template_path}")

        cfg_raw = json.loads(config_path.read_text(encoding="utf-8"))
        cfg = SideConfig(
            thr=float(cfg_raw.get("thr", 0.03)),
            roi_top=tuple(cfg_raw.get("roi_top", [0.12, 0.05, 0.95, 0.45])),
            roi_bottom=tuple(cfg_raw.get("roi_bottom", [0.12, 0.55, 0.95, 0.95])),
            method=str(cfg_raw.get("method", "TM_CCOEFF_NORMED")),
        )
        return SideClassifier(template, cfg)

    def predict_from_gray(self, img_gray: np.ndarray) -> Dict[str, Any]:
        top_score, top_box = _match_in_roi(img_gray, self.template, self.cfg.roi_top, self.method)
        bot_score, bot_box = _match_in_roi(img_gray, self.template, self.cfg.roi_bottom, self.method)

        delta = float(top_score - bot_score)

        if delta >= self.cfg.thr:
            side = "RIGHT"
            confidence = float(delta)
            best_box = top_box
        elif delta <= -self.cfg.thr:
            side = "LEFT"
            confidence = float(-delta)
            best_box = bot_box
        else:
            side = "UNCERTAIN"
            confidence = float(abs(delta))
            best_box = top_box if top_score >= bot_score else bot_box

        return {
            "side_pred": side,
            "confidence": confidence,
            "top_score": float(top_score),
            "bottom_score": float(bot_score),
            "delta": delta,
            "box_xyxy": [int(best_box[0]), int(best_box[1]), int(best_box[2]), int(best_box[3])],
        }

    @staticmethod
    def check_expected(pred: Dict[str, Any], expected_side: str) -> Dict[str, Any]:
        expected_side = expected_side.upper().strip()
        if expected_side not in {"RIGHT", "LEFT"}:
            raise ValueError("expected_side must be RIGHT or LEFT")

        if pred["side_pred"] == "UNCERTAIN":
            status = "UNCERTAIN"
        elif pred["side_pred"] == expected_side:
            status = "OK"
        else:
            status = "WRONG_SIDE"

        return {**pred, "expected_side": expected_side, "status": status}


def draw_overlay_bgr(img_gray: np.ndarray, res: Dict[str, Any]) -> np.ndarray:
    """
    Draw overlay in BGR:
      OK -> green
      WRONG_SIDE -> red
      UNCERTAIN -> orange
    """
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = res["box_xyxy"]
    status = res.get("status", "OK")

    if status == "OK":
        color = (0, 255, 0)
    elif status == "UNCERTAIN":
        color = (0, 165, 255)  # orange (BGR)
    else:
        color = (0, 0, 255)

    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

    txt = f'{status} exp={res.get("expected_side","-")} pred={res["side_pred"]} conf={res["confidence"]:.3f}'
    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return vis
