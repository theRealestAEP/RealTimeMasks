import os
import threading
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageSequence

try:
    from ultralytics import YOLO  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("ultralytics is required") from exc


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def load_overlay_frames(path: str) -> List[np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Overlay not found: {path}")
    pil_img = Image.open(path)
    frames: List[np.ndarray] = []
    is_animated = getattr(pil_img, "is_animated", False)
    if is_animated:
        for frame in ImageSequence.Iterator(pil_img):
            frames.append(np.array(frame.convert("RGBA")))
    else:
        frames.append(np.array(pil_img.convert("RGBA")))
    if not frames:
        raise RuntimeError(f"No frames decoded from overlay: {path}")
    return frames


def composite_on_mask(frame_bgr: np.ndarray, mask: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    h, w = frame_bgr.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return frame_bgr
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return frame_bgr

    ov = cv2.resize(overlay_rgba, (bw, bh), interpolation=cv2.INTER_LINEAR)
    ov_rgb = ov[..., :3].astype(np.float32)
    ov_a = (ov[..., 3:4].astype(np.float32) / 255.0)
    local = mask[y1:y2, x1:x2].astype(np.float32)[..., None]
    alpha = ov_a * local

    out = frame_bgr.copy()
    bg = out[y1:y2, x1:x2].astype(np.float32)
    blended = alpha * ov_rgb + (1.0 - alpha) * bg
    out[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)
    return out


def composite_on_bbox(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int], overlay_rgba: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    h, w = frame_bgr.shape[:2]
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return frame_bgr
    ov = cv2.resize(overlay_rgba, (bw, bh), interpolation=cv2.INTER_LINEAR)
    ov_rgb = ov[..., :3].astype(np.float32)
    ov_a = (ov[..., 3:4].astype(np.float32) / 255.0)
    out = frame_bgr.copy()
    bg = out[y1:y2, x1:x2].astype(np.float32)
    blended = ov_a * ov_rgb + (1.0 - ov_a) * bg
    out[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)
    return out


def detect_targets(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf: float,
    device: Optional[str],
    imgsz: int,
    target_names: Set[str],
) -> Tuple[List[Tuple[int, int, int, int]], List[str], List[Optional[np.ndarray]], Dict[str, int], List[float]]:
    results = model.predict(source=frame_bgr, conf=conf, device=device, imgsz=imgsz, verbose=False)
    if not results:
        return [], [], [], {}, []
    r = results[0]
    names = getattr(r, "names", getattr(model.model, "names", {})) or {}
    boxes = getattr(r, "boxes", None)
    seg = getattr(r, "masks", None)
    selected_boxes: List[Tuple[int, int, int, int]] = []
    selected_names: List[str] = []
    selected_scores: List[float] = []
    if boxes is None:
        return selected_boxes, selected_names, [], {}, selected_scores
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
    cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else None
    confs = boxes.conf.cpu().numpy().astype(float) if hasattr(boxes, "conf") else None
    if xyxy is None or cls is None:
        return selected_boxes, selected_names, [], {}, selected_scores

    frame_counts: Dict[str, int] = Counter()

    selected_indices: List[int] = []
    for i, (bb, c) in enumerate(zip(xyxy, cls)):
        name = _normalize_name(names.get(int(c), str(c)))
        frame_counts[name] += 1
        if name in target_names or int(c) in {67}:  # legacy phone id
            x1, y1, x2, y2 = [int(v) for v in bb[:4]]
            selected_boxes.append((x1, y1, x2, y2))
            selected_indices.append(i)
            selected_names.append(name)
            score = float(confs[i]) if confs is not None else 1.0
            selected_scores.append(score)

    masks_per_det: List[Optional[np.ndarray]] = []
    if seg is not None and seg.data is not None and len(selected_indices) > 0:
        masks_all = seg.data.cpu().numpy().astype(np.uint8)  # (N, H, W)
        chosen = [i for i in selected_indices if i < masks_all.shape[0]]
        idx_set = set(chosen)
        for i, _ in enumerate(selected_indices):
            si = selected_indices[i]
            if si in idx_set:
                masks_per_det.append((masks_all[si] > 0).astype(np.uint8))
            else:
                masks_per_det.append(None)
    else:
        masks_per_det = [None for _ in selected_indices]

    return selected_boxes, selected_names, masks_per_det, frame_counts, selected_scores


def draw_boxes_with_labels(
    frame_bgr: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    labels: List[str],
    scores: List[float],
    masks: Optional[List[Optional[np.ndarray]]] = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    for idx, ((x1, y1, x2, y2), label) in enumerate(zip(boxes, labels)):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        score_txt = f" {scores[idx]:.2f}" if idx < len(scores) else ""
        text = (label if label else "target") + score_txt
        cv2.putText(out, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    if masks is not None:
        h, w = out.shape[:2]
        for m in masks[: len(boxes)]:
            if m is None:
                continue
            m_bin = m
            if m_bin.dtype != np.uint8:
                m_bin = m_bin.astype(np.uint8)
            if m_bin.max() <= 1:
                m_bin = (m_bin * 255).astype(np.uint8)
            if m_bin.shape != (h, w):
                m_bin = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Cyan outline for mask borders
            cv2.drawContours(out, contours, -1, (255, 255, 0), 2)

    return out


class OverlayProcessor:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        imgsz: int = 640,
        conf: float = 0.25,
        mask_conf: Optional[float] = None,
        default_overlay_path: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.model = YOLO(model_path)
        if device is not None:
            try:
                self.model.to(device)
            except Exception:
                pass
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.mask_conf = float(mask_conf) if mask_conf is not None else float(conf)

        # Targets shared with API
        self._target_set_lock = threading.Lock()
        self._target_set: Set[str] = {"cell phone"}
        self._target_set = {_normalize_name(x) for x in self._target_set}

        # Overlay frames
        if default_overlay_path is None:
            project_root = os.path.dirname(os.path.dirname(__file__))
            default_overlay_path = os.path.join(project_root, "cat-smile-smiling-cat.png")
        self.overlay_frames: List[np.ndarray] = load_overlay_frames(default_overlay_path)
        self.overlay_index = 0
        self.current_overlay_name = os.path.basename(default_overlay_path)

        # Video source
        self.source: Union[str, int] = 0
        self.cap: Optional[cv2.VideoCapture] = None

    # ---- Targets management ----
    def get_targets(self) -> List[str]:
        with self._target_set_lock:
            return sorted(self._target_set)

    def set_targets(self, names: List[str]) -> None:
        normalized = {_normalize_name(n) for n in names if n and n.strip()}
        with self._target_set_lock:
            self._target_set = normalized

    def add_target(self, name: str) -> None:
        name_n = _normalize_name(name)
        if not name_n:
            return
        with self._target_set_lock:
            self._target_set.add(name_n)

    def remove_target(self, name: str) -> None:
        name_n = _normalize_name(name)
        with self._target_set_lock:
            self._target_set.discard(name_n)

    def clear_targets(self) -> None:
        with self._target_set_lock:
            self._target_set.clear()

    # ---- Overlay management ----
    def set_overlay_from_path(self, overlay_path: str) -> None:
        frames = load_overlay_frames(overlay_path)
        self.overlay_frames = frames
        self.overlay_index = 0
        self.current_overlay_name = os.path.basename(overlay_path)

    # ---- Settings management ----
    def set_conf(self, conf: float) -> None:
        self.conf = float(conf)

    def set_mask_conf(self, mask_conf: float) -> None:
        self.mask_conf = float(mask_conf)

    def set_imgsz(self, imgsz: int) -> None:
        self.imgsz = int(imgsz)

    def set_device(self, device: Optional[str]) -> bool:
        self.device = device
        if device is None:
            return True
        try:
            self.model.to(device)
            return True
        except Exception:
            return False

    def set_model_path(self, model_path: str) -> bool:
        try:
            new_model = YOLO(model_path)
            if self.device is not None:
                try:
                    new_model.to(self.device)
                except Exception:
                    pass
            self.model = new_model
            self.model_path = model_path
            return True
        except Exception:
            return False

    # ---- Source management ----
    def set_source(self, source: Union[str, int]) -> None:
        if isinstance(source, str) and source.isdigit():
            source_val: Union[str, int] = int(source)
        else:
            source_val = source
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.source = source_val
        self.cap = cv2.VideoCapture(source_val)

    # ---- Processing ----
    def process_frame(self, frame_bgr: np.ndarray, debug: bool = False) -> np.ndarray:
        with self._target_set_lock:
            targets_snapshot = set(self._target_set)
        boxes, labels, masks_per_det, frame_counts, scores = detect_targets(
            self.model, frame_bgr, self.conf, self.device, self.imgsz, targets_snapshot
        )
        out = frame_bgr
        if boxes:
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if i < len(scores) and scores[i] < self.mask_conf:
                    continue
                m = masks_per_det[i] if i < len(masks_per_det) else None
                ov = self.overlay_frames[self.overlay_index % len(self.overlay_frames)]
                if m is not None:
                    out = composite_on_mask(out, m.astype(bool), ov)
                else:
                    out = composite_on_bbox(out, (x1, y1, x2, y2), ov)
                self.overlay_index += 1
        if debug and boxes:
            out = draw_boxes_with_labels(out, boxes, labels, scores, masks_per_det)

        # HUD
        hud = f"Targets: {', '.join(sorted(targets_snapshot))}"
        cv2.putText(out, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        if frame_counts:
            det = ", ".join([f"{k}:{v}" for k, v in sorted(frame_counts.items())])
            cv2.putText(out, f"Detected: {det}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, f"Detected: {det}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        return out

    def _ensure_cap(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            self.set_source(self.source)

    def iter_mjpeg(self, debug: bool = False, conf: Optional[float] = None, mask_conf: Optional[float] = None):
        if conf is not None:
            self.conf = float(conf)
        if mask_conf is not None:
            self.mask_conf = float(mask_conf)

        self._ensure_cap()
        while True:
            if self.cap is None:
                break
            ok, frame = self.cap.read()
            if not ok:
                self._ensure_cap()
                ok, frame = (self.cap.read() if self.cap is not None else (False, None))  # type: ignore
                if not ok:
                    break
            processed = self.process_frame(frame, debug=debug)
            ok2, jpg = cv2.imencode(".jpg", processed, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok2:
                continue
            data = jpg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            )
