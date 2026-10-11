import threading
import time
from typing import Any, Dict, List


class MetricsCollector:
    """In-memory operational metrics collector for model serving."""

    def __init__(self):
        self._lock = threading.Lock()
        self.request_count = 0
        self.error_count = 0
        self.latencies_ms: List[float] = []
        self.predictions: Dict[int, int] = {0: 0, 1: 0}

    def record_request(
        self, latency_ms: float, prediction: int | None = None, is_error: bool = False
    ):
        with self._lock:
            self.request_count += 1
            if is_error:
                self.error_count += 1
            else:
                self.latencies_ms.append(latency_ms)
                # Keep rolling window of last 1000 latencies to bound memory
                if len(self.latencies_ms) > 1000:
                    self.latencies_ms.pop(0)

                if prediction is not None:
                    self.predictions[prediction] = (
                        self.predictions.get(prediction, 0) + 1
                    )

    def get_summary(self, model_version: str = "unknown") -> Dict[str, Any]:
        with self._lock:
            total_predictions = sum(self.predictions.values())
            churn_ratio = (
                self.predictions[1] / total_predictions if total_predictions > 0 else 0.0
            )

            if self.latencies_ms:
                sorted_lat = sorted(self.latencies_ms)
                p95_idx = int(len(sorted_lat) * 0.95)
                p95_latency = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]
                avg_latency = sum(sorted_lat) / len(sorted_lat)
            else:
                p95_latency = 0.0
                avg_latency = 0.0

            return {
                "model_version": model_version,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "avg_latency_ms": round(avg_latency, 2),
                "p95_latency_ms": round(p95_latency, 2),
                "predictions_distribution": {
                    "stay_count (0)": self.predictions.get(0, 0),
                    "churn_count (1)": self.predictions.get(1, 0),
                    "predicted_churn_rate": round(churn_ratio, 4),
                },
            }


# Global singleton collector
collector = MetricsCollector()
