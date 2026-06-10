"""Prometheus exposition-format metrics, hand-rolled.

No prometheus_client dependency: the /metrics contract is a text format, and
the five counters/histograms this app needs fit in 60 lines — one fewer
dependency to audit, and the format itself becomes visible to the learner.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

_LATENCY_BUCKETS = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]


@dataclass
class Metrics:
    request_count: int = 0
    tokens_input_total: int = 0
    tokens_output_total: int = 0
    cannot_answer_total: int = 0
    latency_bucket_counts: dict[float, int] = field(
        default_factory=lambda: {b: 0 for b in _LATENCY_BUCKETS}
    )
    latency_sum: float = 0.0
    latency_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_request(
        self, latency_seconds: float, tokens_in: int, tokens_out: int, cannot_answer: bool
    ) -> None:
        with self._lock:
            self.request_count += 1
            self.tokens_input_total += tokens_in
            self.tokens_output_total += tokens_out
            if cannot_answer:
                self.cannot_answer_total += 1
            self.latency_sum += latency_seconds
            self.latency_count += 1
            for bucket in _LATENCY_BUCKETS:
                if latency_seconds <= bucket:
                    self.latency_bucket_counts[bucket] += 1

    def render(self) -> str:
        """Render in Prometheus exposition format."""
        with self._lock:
            lines = [
                "# HELP cncf_rag_request_count Total queries handled",
                "# TYPE cncf_rag_request_count counter",
                f"cncf_rag_request_count {self.request_count}",
                "# HELP cncf_rag_tokens_input_total LLM input tokens consumed",
                "# TYPE cncf_rag_tokens_input_total counter",
                f"cncf_rag_tokens_input_total {self.tokens_input_total}",
                "# HELP cncf_rag_tokens_output_total LLM output tokens generated",
                "# TYPE cncf_rag_tokens_output_total counter",
                f"cncf_rag_tokens_output_total {self.tokens_output_total}",
                "# HELP cncf_rag_cannot_answer_total Queries answered with cannot_answer",
                "# TYPE cncf_rag_cannot_answer_total counter",
                f"cncf_rag_cannot_answer_total {self.cannot_answer_total}",
                "# HELP cncf_rag_request_latency_seconds Request latency",
                "# TYPE cncf_rag_request_latency_seconds histogram",
            ]
            for bucket in _LATENCY_BUCKETS:
                lines.append(
                    f'cncf_rag_request_latency_seconds_bucket{{le="{bucket}"}} '
                    f"{self.latency_bucket_counts[bucket]}"
                )
            lines.append(
                f'cncf_rag_request_latency_seconds_bucket{{le="+Inf"}} {self.latency_count}'
            )
            lines.append(f"cncf_rag_request_latency_seconds_sum {self.latency_sum}")
            lines.append(f"cncf_rag_request_latency_seconds_count {self.latency_count}")
            return "\n".join(lines) + "\n"


metrics = Metrics()
