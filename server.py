import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from processor import OverlayProcessor


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    return val if val is not None else default


MODEL_PATH = get_env("YOLO_MODEL", "yolo11n-seg.pt")
DEVICE = get_env("YOLO_DEVICE", None)
IMG_SIZE = int(get_env("YOLO_IMGSZ", "640"))
CONF = float(get_env("YOLO_CONF", "0.25"))
MASK_CONF = float(get_env("YOLO_MASK_CONF", str(CONF)))
DEFAULT_OVERLAY = get_env("YOLO_DEFAULT_OVERLAY", str((BASE_DIR.parent / "cat-smile-smiling-cat.png")))

processor = OverlayProcessor(
    model_path=MODEL_PATH,
    device=DEVICE,
    imgsz=IMG_SIZE,
    conf=CONF,
    mask_conf=MASK_CONF,
    default_overlay_path=DEFAULT_OVERLAY,
)

app = FastAPI(title="YOLOE Cam Filter WebApp")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/stream")
def stream(conf: Optional[float] = None, mask_conf: Optional[float] = None, debug: bool = False):
    return StreamingResponse(
        processor.iter_mjpeg(debug=bool(debug), conf=conf, mask_conf=mask_conf),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/targets", response_model=List[str])
def list_targets() -> List[str]:
    return processor.get_targets()


@app.post("/targets/add")
def add_target(name: str = Form(...)):
    if not name.strip():
        raise HTTPException(400, "empty name")
    processor.add_target(name)
    return {"ok": True, "targets": processor.get_targets()}


@app.post("/targets/remove")
def remove_target(name: str = Form(...)):
    processor.remove_target(name)
    return {"ok": True, "targets": processor.get_targets()}


@app.post("/targets/clear")
def clear_targets():
    processor.clear_targets()
    return {"ok": True, "targets": processor.get_targets()}


@app.get("/devices")
def list_devices():
    devices = [{"id": "", "label": "Default"}, {"id": "cpu", "label": "CPU"}]
    try:
        import torch  # type: ignore

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append({"id": "mps", "label": "Apple Metal (MPS)"})
        if torch.cuda.is_available():
            # Optional aggregate option
            devices.append({"id": "cuda", "label": "CUDA (auto)"})
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                devices.append({"id": f"cuda:{i}", "label": f"CUDA:{i} - {name}"})
    except Exception:
        # If torch import fails, we still return CPU/default
        pass
    return devices


@app.get("/settings")
def get_settings():
    return {
        "model": MODEL_PATH,
        "device": DEVICE,
        "imgsz": IMG_SIZE,
        "conf": processor.conf,
        "mask_conf": processor.mask_conf,
        "source": processor.source,
        "overlay": processor.current_overlay_name,
    }


@app.post("/settings")
def update_settings(
    model: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    imgsz: Optional[int] = Form(None),
    conf: Optional[float] = Form(None),
    mask_conf: Optional[float] = Form(None),
    source: Optional[str] = Form(None),
    debug: Optional[bool] = Form(None),
):
    updated = {}
    if conf is not None:
        processor.set_conf(conf)
        updated["conf"] = conf
    if mask_conf is not None:
        processor.set_mask_conf(mask_conf)
        updated["mask_conf"] = mask_conf
    if imgsz is not None:
        processor.set_imgsz(imgsz)
        updated["imgsz"] = imgsz
    if device is not None:
        ok = processor.set_device(device)
        if not ok:
            raise HTTPException(400, "Failed to set device")
        updated["device"] = device
    if model is not None:
        ok = processor.set_model_path(model)
        if not ok:
            raise HTTPException(400, "Failed to load model")
        updated["model"] = model
    if source is not None:
        processor.set_source(source)
        updated["source"] = source
    if debug is not None:
        updated["debug"] = bool(debug)
    return {"ok": True, "updated": updated}


@app.post("/overlay/upload")
async def upload_overlay(file: UploadFile = File(...)):
    filename = file.filename or "overlay.png"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in {".png", ".gif"}:
        raise HTTPException(400, "Only .png or .gif supported")
    dest = UPLOAD_DIR / filename
    data = await file.read()
    dest.write_bytes(data)
    try:
        processor.set_overlay_from_path(str(dest))
    except Exception as e:  # invalid image
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to load overlay: {e}")
    return {"ok": True, "overlay": filename}


@app.get("/healthz")
def health() -> Response:
    return Response(content="ok", media_type="text/plain")


def create_app() -> FastAPI:
    return app


