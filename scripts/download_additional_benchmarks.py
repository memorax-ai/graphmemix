"""Fetch pinned public benchmark data (no model calls, no remote code execution).

Requires huggingface_hub. Large media belongs on a data disk. Published M3Exam
example_set is intentionally not represented as a full benchmark release.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPOS = {
    "mobilemem_omni": "zjunlp/MobileMem",
    "persona_mme": "ClareNie/Persona-MME",
    "smmbench": "HuacanChai/SMMBench",
    "personamem_v2": "bowen-upenn/PersonaMem-v2",
}


def fetch_ranges(url, path, size, workers):
    """Resume large archives in verified HTTP byte ranges; publish only when complete."""
    cache = path.with_name(
        path.name + "." + hashlib.sha256(url.encode()).hexdigest()[:12] + ".ranges"
    )
    cache.mkdir(parents=True, exist_ok=True)
    step = 8 * 1024 * 1024
    starts = list(range(0, size, step))

    def fetch_part(start):
        end = min(start + step, size) - 1
        part = cache / str(start)
        if part.is_file() and part.stat().st_size == end - start + 1:
            return
        for attempt in range(5):
            try:
                request = urllib.request.Request(
                    url, headers={"Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    expected = f"bytes {start}-{end}/{size}"
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != expected
                    ):
                        raise ValueError(
                            f"{path}: server did not return the requested byte range"
                        )
                    data = response.read()
                if len(data) != end - start + 1:
                    raise ValueError(f"{path}: incomplete byte range")
                temporary = part.with_suffix(".part")
                temporary.write_bytes(data)
                temporary.replace(part)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, future in enumerate(
            as_completed([pool.submit(fetch_part, start) for start in starts]), 1
        ):
            future.result()
            if i % 25 == 0 or i == len(starts):
                print(f"{path.name}: ranges {i}/{len(starts)}", flush=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.part")
    with temporary.open("wb") as output:
        for start in starts:
            output.write((cache / str(start)).read_bytes())
    temporary.replace(path)
    for start in starts:
        (cache / str(start)).unlink()
    cache.rmdir()


def fetch(url, path, expected_size=None, range_workers=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and (
        expected_size is None or path.stat().st_size == expected_size
    ):
        return
    if expected_size is not None and expected_size > 64 * 1024 * 1024:
        fetch_ranges(url, path, expected_size, range_workers)
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.part")
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "mm-memory-bench-data-downloader"}
            )
            with (
                urllib.request.urlopen(request, timeout=180) as response,
                temporary.open("wb") as out,
            ):
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if expected_size is not None and temporary.stat().st_size != expected_size:
                raise ValueError(f"{path}: download size mismatch")
            temporary.replace(path)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def fetch_all(tasks, workers, name):
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch, *task, range_workers=workers): str(task[1])
            for task in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 -- report every failed future, then fail the download
                errors.append({"file": futures[future], "error": str(exc)})
            if i % 100 == 0:
                print(f"{name}: {i}/{len(tasks)}", flush=True)
    if errors:
        raise RuntimeError(json.dumps(errors, ensure_ascii=False))


def github_json(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def github_tree(repo, revision="HEAD"):
    tree = github_json(
        f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1"
    )
    if tree.get("truncated"):
        raise RuntimeError(f"{repo}: GitHub returned a truncated file inventory")
    return tree


def extract_omni_images(path, destination):
    """Decode this Windows-created GBK archive, including on Python 3.10."""
    destination = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC validation failed: {bad}")
        members = archive.infolist()
        targets = set()
        for member in members:
            # Keep orig_filename untouched: ZipFile.open verifies local headers with it.
            if not member.flag_bits & 0x800:
                member.filename = member.filename.encode("cp437").decode("gbk")
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("ZIP contains a path outside the extraction directory")
            if target in targets:
                raise ValueError("ZIP contains duplicate paths after filename decoding")
            targets.add(target)
        archive.extractall(destination, members=members)


def download(name, args):
    root = args.raw_root / name
    extra = {}
    source_hashes = {}
    previous_path = root / "download-manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if name == "m3exam":
        repo = "EverM0re/M-3-Exam"
        tree = github_tree(repo, previous.get("revision", "HEAD"))
        revision = tree["sha"]
        selected = [
            x
            for x in tree["tree"]
            if x["type"] == "blob"
            and (
                x["path"].startswith("example_set/")
                or x["path"] in ("README.md", "LICENSE")
            )
        ]
        files = [x["path"] for x in selected]
        source_hashes = {x["path"]: ("git_sha1", x["sha"]) for x in selected}
        fetch_all(
            [
                (
                    f"https://raw.githubusercontent.com/{repo}/{revision}/"
                    + urllib.parse.quote(x["path"]),
                    root / x["path"],
                    x.get("size"),
                )
                for x in selected
            ],
            args.workers,
            name,
        )
        scope = "public example_set only; not full benchmark"
    else:
        from huggingface_hub import HfApi

        repo = REPOS[name]
        info = HfApi().dataset_info(
            repo, revision=previous.get("revision"), files_metadata=True
        )
        revision = info.sha
        inventory = {x.rfilename: x.size for x in info.siblings}
        source_hashes = {
            x.rfilename: (
                ("sha256", x.lfs.sha256) if x.lfs else ("git_sha1", x.blob_id)
            )
            for x in info.siblings
        }
        files = list(inventory)
        if name == "mobilemem_omni":
            files = [x for x in files if x.startswith("omni/") or x == "README.md"]
        elif name == "personamem_v2":
            files = [
                x
                for x in files
                if x
                in (
                    "README.md",
                    "column_descriptions.md",
                    "benchmark/text/benchmark.csv",
                    "benchmark/multimodal/benchmark.csv",
                )
            ]
        prefix = f"{args.hf_download_endpoint.rstrip('/')}/datasets/{repo}/resolve/{revision}/"
        fetch_all(
            [(prefix + urllib.parse.quote(x), root / x, inventory[x]) for x in files],
            args.workers,
            name,
        )
        if name == "personamem_v2":
            csv.field_size_limit(32 * 1024 * 1024)
            histories = set()
            for mode in ("text", "multimodal"):
                with (root / "benchmark" / mode / "benchmark.csv").open(
                    encoding="utf-8", newline=""
                ) as handle:
                    for row in csv.DictReader(handle):
                        histories.update(
                            row[k]
                            for k in ("chat_history_32k_link", "chat_history_128k_link")
                        )
            unknown = histories - inventory.keys()
            if unknown:
                raise ValueError(
                    f"Histories absent from source inventory: {sorted(unknown)}"
                )
            files += sorted(histories)
            fetch_all(
                [
                    (prefix + urllib.parse.quote(x), root / x, inventory[x])
                    for x in sorted(histories)
                ],
                args.workers,
                name,
            )
        scope = (
            "benchmark split; text/multimodal x 32k/128k"
            if name == "personamem_v2"
            else "official published files"
        )
        if name == "mobilemem_omni":
            destination = (root / "omni" / "image").resolve()
            extract_omni_images(root / "omni" / "image.zip", destination)
        if name == "smmbench":
            tool_repo = "FatCatCHC/SMMBench"
            tool_revision = previous.get("tool_source", {}).get("revision")
            if not tool_revision:
                tool_revision = github_json(
                    f"https://api.github.com/repos/{tool_repo}/commits?per_page=1"
                )[0]["sha"]
            tool_info = github_json(
                f"https://api.github.com/repos/{tool_repo}/contents/evaluation/candidate_tools.py?ref={tool_revision}"
            )
            path = root / "upstream/evaluation/candidate_tools.py"
            fetch(
                f"https://raw.githubusercontent.com/{tool_repo}/{tool_revision}/evaluation/candidate_tools.py",
                path,
                tool_info["size"],
            )
            candidates = None
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "candidate_tools"
                    for t in node.targets
                ):
                    candidates = ast.literal_eval(node.value)
            if not isinstance(candidates, list):
                raise ValueError("Missing literal candidate_tools list")
            (root / "candidate_tools.json").write_text(
                json.dumps(candidates, ensure_ascii=False, indent=2)
            )
            extra = {
                "tool_source": {
                    "repo": tool_repo,
                    "revision": tool_revision,
                    "file": "evaluation/candidate_tools.py",
                }
            }
            files += ["upstream/evaluation/candidate_tools.py", "candidate_tools.json"]
            source_hashes["upstream/evaluation/candidate_tools.py"] = (
                "git_sha1",
                tool_info["sha"],
            )
    checksums = []
    for name_in_repo in files:
        path = root / name_in_repo
        digest = hashlib.sha256()
        git_digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                git_digest.update(chunk)
        algorithm, expected = source_hashes.get(name_in_repo, (None, None))
        actual = digest.hexdigest() if algorithm == "sha256" else git_digest.hexdigest()
        if expected and actual != expected:
            path.replace(path.with_name(path.name + ".corrupt"))
            raise ValueError(
                f"{path}: upstream hash mismatch; quarantined as .corrupt, rerun to redownload"
            )
        checksums.append(
            {
                "path": name_in_repo,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
                "upstream_hash_verified": bool(expected),
            }
        )
    (root / "download-files.json").write_text(json.dumps(checksums, indent=2))
    manifest = {
        "repo": repo,
        "revision": revision,
        "files": len(files),
        "errors": [],
        "scope": scope,
        "file_inventory": "download-files.json",
        **extra,
    }
    (root / "download-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(name, json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", nargs="+", choices=[*REPOS, "m3exam"])
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--hf-download-endpoint",
        default="https://huggingface.co",
        help="Optional content mirror; metadata and pinned revision still come from the official HF API",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for name in args.benchmarks:
        download(name, args)


if __name__ == "__main__":
    main()
