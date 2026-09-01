#!/usr/bin/env python3
"""Audit the explicit repository allowlist without deleting local work."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWLIST = ROOT / "release/repository_allowlist.txt"
TEXT_SUFFIXES = {
    "", ".bib", ".cfg", ".csv", ".json", ".jsonl", ".md", ".py",
    ".sh", ".tex", ".txt", ".toml", ".yaml", ".yml",
}
SECRET_PATTERNS = {
    "openai_style_key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    "bearer_literal": re.compile(r"Bearer\s+[A-Za-z0-9_.-]{24,}"),
}
MACHINE_PATH_PATTERNS = {
    "workspace_user_path": re.compile(r"/(?:data2?|home)/ligeng/"),
}
EXCLUDED_PARTS = {".git", ".all-git", ".pytest_cache", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def release_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and not EXCLUDED_PARTS.intersection(relative.parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def expand_allowlist(path: Path) -> tuple[list[Path], list[str]]:
    files: set[Path] = set()
    missing = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        target = ROOT / value
        if target.is_dir():
            files.update(item for item in target.rglob("*") if release_file(item))
        elif target.is_file():
            files.add(target)
        else:
            missing.append(value)
    return sorted(files), missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files, missing = expand_allowlist(args.allowlist)
    issues = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = str(path.relative_to(ROOT))
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                issues.append({"file": relative, "kind": name})
        for name, pattern in MACHINE_PATH_PATTERNS.items():
            if pattern.search(text):
                issues.append({"file": relative, "kind": name})

    summary = {
        "protocol": "graphmemix-repository-audit-v1",
        "allowlist": str(args.allowlist.relative_to(ROOT)),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "missing": missing,
        "issues": issues,
        "status": "valid" if not missing and not issues else "invalid",
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if summary["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
