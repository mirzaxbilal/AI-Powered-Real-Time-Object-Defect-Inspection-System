import time
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import (BOX_SIZE, CONFIDENCE_THRESHOLD, MIN_RECORD_INTERVAL,
                    PRESENCE_THRESHOLD, BG_WARMUP_FRAMES, DIFF_THRESHOLD,
                    BG_ADAPT_RATE, FRAMES_DIR)


class InferencePipeline:
    def __init__(self, model_path: str, camera, twin,
                 threshold: float = CONFIDENCE_THRESHOLD,
                 min_interval: float = MIN_RECORD_INTERVAL,
                 presence_threshold: float = PRESENCE_THRESHOLD):
        print(f"[inference] Loading model from {model_path} ...")
        self._model = YOLO(model_path)
        print("[inference] Model ready.")
        self._camera = camera
        self._twin = twin
        self._threshold = threshold
        self._min_interval = min_interval
        self._presence_threshold = presence_threshold
        self._latest = None
        self._lock = threading.Lock()
        self._running = False
        self._paused = False

        # Background reference: built from warmup frames, never absorbs the bottle
        self._bg_ref: np.ndarray | None = None
        self._warmup_acc: list[np.ndarray] = []
        self._frame_count = 0

        # Post-record gate: prevents the same physical bottle being logged twice
        self._waiting_for_clear = False
        self._last_verdict: bool | None = None  # True = pass, False = reject

        self._frame_dir = Path(FRAMES_DIR)
        self._frame_dir.mkdir(exist_ok=True)

    # Public control

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def is_paused(self) -> bool:
        return self._paused

    def start(self):
        self._running = True
        threading.Thread(target=self._run_loop, daemon=True).start()

    def stop(self):
        self._running = False

    def get_frame(self):
        with self._lock:
            return self._latest.copy() if self._latest is not None else None

    # Internal loop

    def _run_loop(self):
        while self._running:
            frame = self._camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            annotated = self._process(frame)
            with self._lock:
                self._latest = annotated

    def _process(self, frame):
        h, w = frame.shape[:2]
        x1 = w // 2 - BOX_SIZE // 2
        y1 = h // 2 - BOX_SIZE // 2
        x2 = x1 + BOX_SIZE
        y2 = y1 + BOX_SIZE

        roi = frame[y1:y2, x1:x2]
        roi_clean = roi.copy()  # saved before annotation for Claude

        #Background reference build (warmup)
        self._frame_count += 1
        warming_up = self._frame_count <= BG_WARMUP_FRAMES

        if warming_up:
            self._warmup_acc.append(roi.astype(np.float32))
            if self._frame_count == BG_WARMUP_FRAMES:
                self._bg_ref = np.mean(self._warmup_acc, axis=0).astype(np.uint8)
                self._warmup_acc = []
                print("[inference] Background reference captured.")
            bottle_present = False
            presence_ratio = 0.0
        else:
            # Presence detection via static diff 
            # Compares ROI against a fixed reference that was captured on startup.
            # Because the reference never updates during bottle frames, the same
            # bottle can be passed through indefinitely without drift.
            diff = cv2.absdiff(roi, self._bg_ref)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
            presence_ratio = cv2.countNonZero(mask) / (BOX_SIZE * BOX_SIZE)
            bottle_present = presence_ratio >= self._presence_threshold

        #YOLOv8 classification
        results = self._model.predict(roi, imgsz=640, verbose=False)
        probs = results[0].probs
        class_id = probs.top1
        confidence = probs.top1conf.item()
        class_name = results[0].names[class_id]
        passed = class_name == "unmarked"

        verdict_color = (0, 200, 80) if passed else (0, 60, 220)
        verdict = "PASS" if passed else "REJECT"
        label = f"{verdict}  {class_name}  {confidence:.1%}"

        # Box stays the colour of the last verdict while clearing, goes grey when empty
        if self._waiting_for_clear and self._last_verdict is not None:
            box_color = (0, 200, 80) if self._last_verdict else (0, 60, 220)
        else:
            box_color = verdict_color if bottle_present else (70, 70, 70)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 12, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.putText(frame, "Inspection Zone - place bottle inside box",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Status line
        if self._paused:
            status_text = "PAUSED - recording suspended"
            status_color = (0, 165, 230)
        elif warming_up:
            status_text = f"CALIBRATING {self._frame_count}/{BG_WARMUP_FRAMES} - keep bottle out of zone"
            status_color = (110, 110, 110)
        elif self._waiting_for_clear:
            verb = "PASS" if self._last_verdict else "REJECT"
            status_text = f"{verb} RECORDED - waiting for bottle to clear zone"
            status_color = (0, 200, 80) if self._last_verdict else (0, 60, 220)
        elif bottle_present:
            status_text = f"BOTTLE DETECTED ({presence_ratio:.0%}) - scanning"
            status_color = (0, 200, 80)
        else:
            status_text = "NO BOTTLE - ready"
            status_color = (60, 140, 200)

        cv2.putText(frame, status_text, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1)

        # Record gate
        if self._waiting_for_clear:
            if not bottle_present:
                self._waiting_for_clear = False
                # Gently refresh the reference with this clean belt frame so it
                # tracks slow lighting drift without ever absorbing the bottle.
                if self._bg_ref is not None and BG_ADAPT_RATE > 0:
                    self._bg_ref = cv2.addWeighted(
                        self._bg_ref, 1 - BG_ADAPT_RATE,
                        roi, BG_ADAPT_RATE, 0
                    )
        elif (not self._paused
              and bottle_present
              and confidence >= self._threshold):
            bottle_id = self._twin.record({
                "label": class_name,
                "confidence": round(confidence, 4),
                "passed": passed,
            })
            self._last_verdict = passed
            self._waiting_for_clear = True
            if not passed:
                self._save_frame(roi_clean, bottle_id)

        return frame

    def _save_frame(self, roi, bottle_id: int):
        path = self._frame_dir / f"marked_{bottle_id:04d}.jpg"
        cv2.imwrite(str(path), roi)
