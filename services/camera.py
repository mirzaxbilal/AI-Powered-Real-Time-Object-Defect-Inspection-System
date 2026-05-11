import threading
import cv2


class CameraService:
    def __init__(self, index: int = 0):
        self._index = index
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self._index}")
        self._running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                with self._lock:
                    self._frame = frame

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
