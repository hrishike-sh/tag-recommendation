"""Download the original MTG archives; never execute archive contents."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

BASE = "https://mtg.upf.edu/static/datasets/last.fm"
FILES = {
    "1k": "userid-timestamp-artid-artname-traid-traname.tsv",
    "360k": "usersha1-artmbid-artname-plays.tsv",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def acquire(root: Path, dataset: str) -> dict:
    label = "1K" if dataset == "1k" else "360K"
    name = f"lastfm-dataset-{label}"
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / f"{name}.tar.gz"
    url = f"{BASE}/{archive.name}"
    manifest_path = raw / f"{dataset}.manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (archive.exists() and sha256(archive) == manifest["archive_sha256"]
                and all((root / f["path"]).exists()
                        and sha256(root / f["path"]) == f["sha256"]
                        for f in manifest["files"])):
            return manifest
        raise ValueError("Previously verified data changed; inspect it before reacquiring")
    # curl uses the platform's trusted certificates on Windows and supports resuming.
    subprocess.run(["curl", "--fail", "--location", "--retry", "3",
                    "--continue-at", "-", "--output", str(archive), url], check=True)
    extracted = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            # Preserve data and license/readme files, but not bundled executable scripts.
            if not (member.name.endswith(".tsv") or "readme" in member.name.lower()
                    or "license" in member.name.lower() or "copying" in member.name.lower()):
                continue
            target = (raw / member.name).resolve()
            if not target.is_relative_to(raw.resolve()):
                raise ValueError(f"Unsafe archive path: {member.name}")
            tar.extract(member, raw, filter="data")
            extracted.append({"path": target.relative_to(root.resolve()).as_posix(),
                              "bytes": target.stat().st_size, "sha256": sha256(target)})
    if not (raw / name / FILES[dataset]).exists():
        raise ValueError("Archive does not contain the expected interaction file")
    manifest = {"dataset": dataset, "url": url,
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
                "integrity": "Local SHA-256 fingerprints; not publisher-signed checksums",
                "license": "Last.fm/MTG non-commercial dataset; see bundled README",
                "files": extracted}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
