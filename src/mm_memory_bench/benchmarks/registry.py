from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable


CONVERTERS = {
    "atm_bench": "mm_memory_bench.benchmarks.converters.atm:convert",
    "mem_gallery": "mm_memory_bench.benchmarks.converters.mem_gallery:convert",
    "memeye": "mm_memory_bench.benchmarks.converters.memeye:convert",
    "h2hmem": "mm_memory_bench.benchmarks.converters.h2hmem:convert",
}


def get_converter(name: str) -> Callable[..., dict[str, Any]]:
    try:
        target = CONVERTERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(CONVERTERS))
        raise KeyError(f"unknown benchmark {name!r}; available: {available}") from exc
    module_name, function_name = target.split(":", 1)
    module = import_module(module_name)
    return getattr(module, function_name)


def convert(name: str, raw_root: Path, output_root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    converter = get_converter(name)
    return converter(raw_root, output_root / name, overwrite=overwrite)
