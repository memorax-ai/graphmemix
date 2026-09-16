"""Recognize runner failures consistently before assigning any benchmark score."""
from collections.abc import Mapping


def method_failure(prediction):
    metadata = prediction.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    for source in (metadata, prediction):
        if (source.get("method_error") or source.get("error_type")
                or source.get("status") in {"error", "failed"}):
            return {
                "error_type": str(source.get("error_type") or "method_error"),
                "error": str(source.get("method_error") or source.get("error")
                             or "Method execution failed"),
            }
    return None
