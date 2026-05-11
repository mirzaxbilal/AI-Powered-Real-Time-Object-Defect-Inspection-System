import threading
from datetime import datetime


class TwinState:
    def __init__(self, max_history: int = 50):
        self._lock = threading.Lock()
        self._max = max_history
        self._total = 0
        self._passed = 0
        self._rejected = 0
        self._pass_rate = 0.0
        self._current = None
        self._bottles: list = []
        self._last_report = datetime.now().isoformat()
        self._last_analysis = None

    def record(self, result: dict) -> int:
        """Record a bottle result and return the assigned bottle id."""
        with self._lock:
            self._total += 1
            if result["passed"]:
                self._passed += 1
            else:
                self._rejected += 1
            self._pass_rate = self._passed / self._total * 100
            entry = {
                "id": self._total,
                "label": result["label"],
                "confidence": result["confidence"],
                "passed": result["passed"],
                "timestamp": datetime.now().isoformat(),
            }
            self._current = entry
            self._bottles.append(entry)
            if len(self._bottles) > self._max:
                self._bottles.pop(0)
            self._last_report = entry["timestamp"]
            return self._total

    def set_analysis(self, result: dict):
        with self._lock:
            self._last_analysis = result

    def reset(self):
        with self._lock:
            self._total = 0
            self._passed = 0
            self._rejected = 0
            self._pass_rate = 0.0
            self._current = None
            self._bottles = []
            self._last_report = datetime.now().isoformat()
            self._last_analysis = None

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "total": self._total,
                "passed": self._passed,
                "rejected": self._rejected,
                "pass_rate": round(self._pass_rate, 1),
                "current": self._current,
                "bottles": list(self._bottles),
                "last_report": self._last_report,
                "last_analysis": self._last_analysis,
            }
