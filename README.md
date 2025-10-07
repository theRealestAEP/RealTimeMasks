# YOLOE Cam Filter WebApp

Quick start (macOS/Linux):

```bash
git clone <this repo>
cd yoloe_cam_filter_webapp
./setup.sh
./run.sh --reload
```

Notes:
- The server will auto-download `yolo11n-seg.pt` via Ultralytics on first run.
- Override defaults via env vars: `YOLO_MODEL`, `YOLO_DEVICE` (`cpu|mps|cuda`), `YOLO_IMGSZ`, `YOLO_CONF`, `YOLO_MASK_CONF`, `YOLO_DEFAULT_OVERLAY`.
- Open http://localhost:8000/ after starting.


