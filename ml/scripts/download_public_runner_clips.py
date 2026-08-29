#!/usr/bin/env python3
"""Download and integrity-check licensed runner video sessions from a manifest.

The manifest is deliberately explicit: every clip must have a human-reviewed source
page, license, destination file, and clean media URL before this script will fetch it.
This keeps dataset collection reproducible without becoming a general web scraper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any


CHUNK_BYTES = 4 * 1024 * 1024
MP4_BRANDS = (b"ftyp",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def looks_like_mp4(path: Path) -> bool:
    if path.stat().st_size < 16:
        return False
    with path.open("rb") as stream:
        header = stream.read(32)
    return any(brand in header[4:16] for brand in MP4_BRANDS)


def fetch(url: str, destination: Path, timeout: int) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "runner-dataset-curation/1.0"}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=CHUNK_BYTES)
    os.replace(partial, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    license_info = manifest.get("license", {})
    if not license_info.get("url"):
        raise ValueError("Manifest must declare a license URL")

    clips = manifest.get("clips", [])
    if not clips:
        raise ValueError("Manifest contains no clips")

    failures: list[str] = []
    for index, clip in enumerate(clips, start=1):
        required = ("session_id", "file", "source_page", "download_url")
        missing = [key for key in required if not clip.get(key)]
        if missing:
            failures.append(f"{index}: missing {', '.join(missing)}")
            continue

        destination = (args.workspace / clip["file"]).resolve()
        expected = clip.get("sha256")
        valid_existing = destination.exists() and looks_like_mp4(destination)
        if valid_existing and expected:
            valid_existing = sha256_file(destination) == expected

        if args.force or not valid_existing:
            print(f"[{index}/{len(clips)}] downloading {clip['session_id']}", flush=True)
            try:
                fetch(clip["download_url"], destination, args.timeout)
            except Exception as exc:  # report all failed sessions in one run
                destination.with_suffix(destination.suffix + ".part").unlink(missing_ok=True)
                failures.append(f"{clip['session_id']}: download failed: {exc}")
                continue
        else:
            print(f"[{index}/{len(clips)}] verified existing {clip['session_id']}", flush=True)

        if not looks_like_mp4(destination):
            failures.append(f"{clip['session_id']}: downloaded file is not an MP4")
            continue
        clip["bytes"] = destination.stat().st_size
        clip["sha256"] = sha256_file(destination)

    manifest["download_validation"] = {
        "requested": len(clips),
        "valid": len(clips) - len(failures),
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["download_validation"], indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
