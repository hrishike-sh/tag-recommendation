"""Acquire and inspect HetRec 2011 Last.fm 2K tag dataset."""

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import duckdb

HETREC_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def acquire_hetrec(root: Path) -> dict:
    raw_tags = root / "data/raw/hetrec2011-lastfm-2k"
    raw_tags.mkdir(parents=True, exist_ok=True)
    archive = raw_tags / "hetrec2011-lastfm-2k.zip"

    if not archive.exists():
        print(f"Downloading HetRec 2011 dataset from {HETREC_URL}...")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "-o", str(archive), HETREC_URL], check=True)

    print("Extracting HetRec 2011 files...")
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(raw_tags)

    files = [f.name for f in raw_tags.glob("*.dat")]
    manifest = {
        "url": HETREC_URL,
        "archive_sha256": sha256(archive),
        "files": files,
    }
    (raw_tags / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def inspect_hetrec(root: Path):
    raw_tags = root / "data/raw/hetrec2011-lastfm-2k"
    acquire_hetrec(root)

    print("\n--- Inspecting HetRec 2011 Files ---")
    db = duckdb.connect()
    
    # Check artists.dat: id \t name \t url \t pictureURL
    print("Artists sample count:")
    print(db.sql(f"SELECT count(*) FROM read_csv('{raw_tags.as_posix()}/artists.dat', delim='\t', header=true, encoding='latin1')").fetchone()[0])

    # Check tags.dat: tagID \t tagValue
    print("\nTags sample count:")
    print(db.sql(f"SELECT count(*) FROM read_csv('{raw_tags.as_posix()}/tags.dat', delim='\t', header=true, encoding='latin1')").fetchone()[0])

    tag_records = db.sql(f"SELECT count(*) FROM read_csv('{raw_tags.as_posix()}/user_taggedartists.dat', delim='\t', header=true, encoding='latin1')").fetchone()[0]
    unique_tags = db.sql(f"SELECT count(*) FROM read_csv('{raw_tags.as_posix()}/tags.dat', delim='\t', header=true, encoding='latin1')").fetchone()[0]
    tagged_artists = db.sql(f"SELECT count(DISTINCT artistID) FROM read_csv('{raw_tags.as_posix()}/user_taggedartists.dat', delim='\t', header=true, encoding='latin1')").fetchone()[0]
    
    print(f"\nHetRec Stats: {tag_records:,} tagging events, {unique_tags:,} distinct tags, {tagged_artists:,} tagged artists.")


if __name__ == "__main__":
    inspect_hetrec(Path.cwd())
