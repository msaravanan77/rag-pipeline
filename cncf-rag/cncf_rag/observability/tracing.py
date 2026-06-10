"""OpenTelemetry tracing setup — one span per request covering the full pipeline."""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_initialized = False


def init_tracing() -> trace.Tracer:
    """Idempotent tracer setup. Exports OTLP if an endpoint is configured;
    otherwise spans are created but not exported (zero-config local dev)."""
    global _initialized
    if not _initialized:
        provider = TracerProvider(
            resource=Resource.create(
                {"service.name": os.environ.get("OTEL_SERVICE_NAME", "cncf-rag")}
            )
        )
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        _initialized = True
    return trace.get_tracer("cncf-rag")
