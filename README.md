# AI Powered Defect Inspection System

A real-time quality control system built with YOLOv8, Flask, and visual LLM integration (Claude AI) for AI-powered defect analysis. Designed for deployment on a production conveyor belt line, the system uses a front-facing camera positioned at the end of the belt to inspect items as they pass through a defined inspection zone, classifying them as defective or clean and generating detailed defect reports for rejected items using vision-language model analysis.

> **Note:** The demo video below demonstrates a proof-of-concept using a handheld bottle in front of a webcam. In a production deployment the camera would be fixed and aimed at bottles passing through the inspection zone on a conveyor belt.

## Demo

[![Bottle Defect Inspection System Demo](https://img.youtube.com/vi/PF0E413GgYI/maxresdefault.jpg)](https://youtu.be/PF0E413GgYI)

## Dataset

The YOLOv8 classification model was trained on a custom dataset of marked and unmarked water bottles, published on Kaggle: [Marked and Unmarked Water Bottles](https://www.kaggle.com/datasets/mirzabilalbaig/marked-and-unmarked-watter-bottles).

While this project targets bottle defect detection, the pipeline is not specific to bottles. By training a YOLOv8 classification model on your own dataset, the system can be adapted to inspect any type of object on a production line, such as packaged goods, electronic components, or food items.

## Features

- **Real-time classification:** YOLOv8n classification model runs inference on every frame, labelling bottles as `unmarked` (pass) or `marked` (reject) with confidence score
- **Presence detection:** Static background reference built from warmup frames; `cv2.absdiff` + pixel threshold triggers only when a bottle is actually in the inspection zone, preventing phantom logs on an empty belt
- **Edge-triggered recording gate:** Each physical bottle is logged exactly once per pass; the system waits for the zone to clear before it will record the same bottle again
- **Slow background adaptation:** After each bottle exits, the reference blends 5% toward the current empty frame to track gradual lighting drift without ever absorbing the bottle
- **Live MJPEG stream:** Annotated camera feed streamed at 15 fps with coloured bounding box, verdict label, confidence, and status line
- **Digital twin state:** Thread-safe in-memory state tracking total inspected, passed, rejected, pass rate, and inspection log (last 50 bottles)
- **AI defect analysis:** On-demand visual LLM analysis of any rejected bottle, sending the saved ROI image and session context to a vision-language model and returning a 3-4 sentence report covering defect location, severity, pattern vs isolated, and recommendation
- **Pause / Resume:** Suspend recording without stopping the camera or inference loop
- **Session reset:** Clear all counters and history without restarting the server
- **Dark industrial dashboard:** Pure HTML/CSS/JS frontend with live-polling stats panel, inspection log table, and AI analysis modal
- **REST API:** All functionality exposed as JSON endpoints

## Project Structure

```
.
├── app.py                          # Flask app, routes, startup
├── config.py                       # All tunable constants
├── requirements.txt
├── .env                            # API key (not committed)
├── .env.example                    # Key template
├── services/
│   ├── camera.py                   # Background capture thread
│   ├── inference.py                # YOLOv8 inference + presence detection
│   ├── twin.py                     # Thread-safe digital twin state
│   └── llm.py                      # Visual LLM defect analysis
├── templates/
│   └── index.html                  # Dashboard
├── saved_frames/                   # Auto-created; ROI crops of rejected bottles
└── Bottle_Inspection_API.postman_collection.json
```

## Requirements

- Python 3.10+
- Webcam or USB camera
- Trained YOLOv8 classification model (`.pt` file) with classes `marked` / `unmarked`
- Anthropic API key (for AI defect analysis)

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Add your Anthropic API key**

Copy `.env.example` to `.env` and fill in your key:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Get a key at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).

**3. Verify the model path**

In `config.py`, set `MODEL_PATH` to point to your trained `.pt` file:

```python
MODEL_PATH = r"runs\classify\bottle_classifier-2\weights\best.pt"
```

**4. Run**

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

On startup the system captures 60 warmup frames to build the empty-belt reference. Keep the inspection zone clear during this period; the status bar shows `CALIBRATING 1/60`.

## Configuration

All knobs are in `config.py`:

| Constant | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | OpenCV camera index |
| `MODEL_PATH` | (required) | Path to trained `.pt` weights |
| `CONFIDENCE_THRESHOLD` | `0.75` | Minimum YOLO confidence to trigger a record |
| `BOX_SIZE` | `300` | Side length (px) of the square inspection zone |
| `BG_WARMUP_FRAMES` | `60` | Frames captured at startup to build the background reference |
| `DIFF_THRESHOLD` | `25` | Per-pixel brightness delta (0-255) to count a pixel as changed |
| `PRESENCE_THRESHOLD` | `0.15` | Fraction of ROI pixels that must differ before a bottle is declared present |
| `BG_ADAPT_RATE` | `0.05` | Weight of each clean frame blended into the background reference after a bottle exits |
| `MAX_HISTORY` | `50` | Maximum bottle records kept in the in-memory log |
| `STREAM_FPS` | `15` | MJPEG stream frame rate |
| `LLM_MODEL` | `claude-opus-4-7` | Vision-language model used for defect analysis |
| `FRAMES_DIR` | `saved_frames` | Directory where rejected bottle ROI crops are saved |

**Tuning presence detection:** If the empty belt triggers false detections, raise `DIFF_THRESHOLD` or `PRESENCE_THRESHOLD`. If a bottle is not being detected, lower them.

## API Reference

Base URL: `http://localhost:5000`

---

### `GET /api/health`

Health check.

**Response**
```json
{ "status": "ok" }
```

---

### `GET /api/status`

Full digital twin state: counters, inspection log, last AI analysis, and pause state.

**Response**
```json
{
  "total": 12,
  "passed": 9,
  "rejected": 3,
  "pass_rate": 75.0,
  "paused": false,
  "last_report": "2025-05-11T14:32:01.123456",
  "current": {
    "id": 12,
    "label": "unmarked",
    "confidence": 0.9821,
    "passed": true,
    "timestamp": "2025-05-11T14:32:01.123456"
  },
  "bottles": [
    {
      "id": 1,
      "label": "marked",
      "confidence": 0.8843,
      "passed": false,
      "timestamp": "2025-05-11T14:30:00.000000"
    }
  ],
  "last_analysis": {
    "bottle_id": 1,
    "report": "The bottle shows a dark printed mark on the upper shoulder..."
  }
}
```

---

### `POST /api/pause`

Suspend recording. The camera and inference loop continue running; the system simply will not log new bottles or save frames.

**Response**
```json
{ "ok": true, "paused": true }
```

---

### `POST /api/resume`

Resume recording after a pause.

**Response**
```json
{ "ok": true, "paused": false }
```

---

### `POST /api/reset`

Clear all counters and history. Does not affect the background reference or the running pipeline.

**Response**
```json
{ "ok": true }
```

---

### `POST /api/analyse/<bottle_id>`

Trigger visual LLM analysis of a rejected bottle's saved ROI image. The report covers defect location, severity, whether it's isolated or part of a pattern, and a recommendation. The result is stored in the twin state and returned in the response.

Only bottles with `passed: false` have a saved frame; calling this on a passing bottle returns 404.

**URL parameter:** `bottle_id` (integer) - the `id` from the inspection log

**Response `200`**
```json
{
  "bottle_id": 3,
  "report": "The bottle displays a prominent black printed label on its lower body, approximately one-third of the way up from the base. The marking appears to be a manufacturer's stamp or batch code, making this a moderate defect. Given that 2 of the last 5 bottles have been flagged, this may indicate a batch issue rather than an isolated incident. Recommendation: reject and escalate for batch review."
}
```

**Response `404`** - No saved frame for the given bottle ID

**Response `500`** - Vision model API error

---

### `GET /stream`

MJPEG video stream. Embed directly in an `<img>` tag:

```html
<img src="http://localhost:5000/stream">
```

---

### `GET /`

Serves the live dashboard.

---

## How It Works

### Presence Detection

On startup, `InferencePipeline` accumulates `BG_WARMUP_FRAMES` ROI crops and computes their pixel-wise mean as `_bg_ref`. During operation each frame is compared against this fixed reference using `cv2.absdiff`. The fraction of pixels exceeding `DIFF_THRESHOLD` determines `presence_ratio`; if it meets `PRESENCE_THRESHOLD` a bottle is declared present.

Because the reference is never updated during bottle frames, the same bottle can pass through indefinitely without the system learning to ignore it. After each bottle exits, a small `BG_ADAPT_RATE` blend refreshes the reference to handle slow lighting changes.

### Recording Gate

`_waiting_for_clear` ensures each physical bottle is recorded exactly once. When a bottle is recorded, the flag is set and no further records are written until the zone is empty for at least one frame, at which point the flag clears and the system is ready for the next pass.

### AI Defect Analysis

When the operator clicks **Analyse** on a rejected row, the dashboard calls `POST /api/analyse/<id>`. The server reads the saved JPEG crop, encodes it, and sends it to a vision-language model alongside a prompt containing the session pass rate, total defect count, and recent-10-bottle summary. The returned report is displayed in a modal and stored in the twin state.

## Postman Collection

Import `Bottle_Inspection_API.postman_collection.json` into Postman for a ready-made set of requests covering all endpoints, with example responses and basic test scripts.
