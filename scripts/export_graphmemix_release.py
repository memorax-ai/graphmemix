#!/usr/bin/env python3
"""Export the audited GraphMemix repository allowlist into a directory."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWLIST = ROOT / "release/repository_allowlist.txt"
EXCLUDED_PARTS = {".git", ".all-git", ".pytest_cache", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def release_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and not EXCLUDED_PARTS.intersection(relative.parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def entries(path: Path) -> list[str]:
    return [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def copy_entry(relative: str, target: Path) -> list[Path]:
    source = ROOT / relative
    destination = target / relative
    copied: list[Path] = []
    if source.is_dir():
        for path in source.rglob("*"):
            if not release_file(path):
                continue
            output = destination / path.relative_to(source)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
            copied.append(output.relative_to(target))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination.relative_to(target))
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument(
        "--update", action="store_true",
        help="Refresh an existing export without deleting extra files.",
    )
    args = parser.parse_args()
    if args.target.exists() and any(args.target.iterdir()) and not args.update:
        raise RuntimeError(f"refusing to overwrite non-empty target: {args.target}")
    args.target.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_graphmemix_release.py"),
         "--allowlist", str(args.allowlist)],
        cwd=ROOT,
        check=True,
    )
    copied: set[Path] = set()
    for relative in entries(args.allowlist):
        copied.update(copy_entry(relative, args.target))

    manifest = []
    for relative in sorted(copied):
        path = args.target / relative
        if relative == Path("MANIFEST.sha256"):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append(f"{digest}  {relative}")
    (args.target / "MANIFEST.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"exported {len(manifest)} files to {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
