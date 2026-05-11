import os
import sys
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template

# Load .env before anything else so ANTHROPIC_API_KEY is available
load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit(
        "\n[ERROR] ANTHROPIC_API_KEY is not set.\n"
        "Create a .env file in this folder with:\n"
        "  ANTHROPIC_API_KEY=sk-ant-...\n"
        "Get your key at https://console.anthropic.com/settings/keys\n"
    )

from config import (MODEL_PATH, CAMERA_INDEX, CONFIDENCE_THRESHOLD,
                    MIN_RECORD_INTERVAL, MAX_HISTORY, FRAMES_DIR)
from services.camera import CameraService
from services.twin import TwinState
from services.inference import InferencePipeline
from services.llm import analyse_bottle

app = Flask(__name__)

FRAMES_PATH = Path(FRAMES_DIR)
FRAMES_PATH.mkdir(exist_ok=True)

_camera = CameraService(CAMERA_INDEX)
_twin = TwinState(MAX_HISTORY)
_pipeline = InferencePipeline(MODEL_PATH, _camera, _twin, CONFIDENCE_THRESHOLD, MIN_RECORD_INTERVAL)


def _mjpeg_frames():
    while True:
        frame = _pipeline.get_frame()
        if frame is None:
            time.sleep(0.033)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(1 / 15)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    return Response(_mjpeg_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    data = _twin.to_dict()
    data["paused"] = _pipeline.is_paused()
    return jsonify(data)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    _twin.reset()
    return jsonify({"ok": True})


@app.route("/api/pause", methods=["POST"])
def api_pause():
    _pipeline.pause()
    return jsonify({"ok": True, "paused": True})


@app.route("/api/resume", methods=["POST"])
def api_resume():
    _pipeline.resume()
    return jsonify({"ok": True, "paused": False})


@app.route("/api/analyse/<int:bottle_id>", methods=["POST"])
def api_analyse(bottle_id: int):
    frame_path = FRAMES_PATH / f"marked_{bottle_id:04d}.jpg"
    if not frame_path.exists():
        return jsonify({"error": f"No saved frame for bottle #{bottle_id}"}), 404
    try:
        report = analyse_bottle(bottle_id, frame_path, _twin)
        result = {"bottle_id": bottle_id, "report": report}
        _twin.set_analysis(result)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("[app] Starting camera ...")
    _camera.start()
    print("[app] Starting inference pipeline ...")
    _pipeline.start()
    print("[app] Server ready at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
