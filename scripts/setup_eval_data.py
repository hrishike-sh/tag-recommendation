"""Ingest a realistic Last.fm 1K chunk with events spanning pre-cutoff and post-cutoff periods for evaluation smoke test."""

from pathlib import Path
import numpy as np
from lastfm.ingest import ingest
from lastfm.matrix import matrix


def main():
    root = Path.cwd()
    print("=== Creating Last.fm 1K Chronological Dataset (Train: <2009-05-01, Test: >=2009-05-01) ===")

    lines = []
    rng = np.random.default_rng(42)
    users = [f"user_{u:04d}_{rng.integers(100000, 999999)}" for u in range(50)]
    
    # 20 artists, 100 tracks
    tracks = []
    for a in range(20):
        a_mbid = f"artmbid_{a:04d}"
        a_name = f"Artist {a}"
        for t in range(5):
            t_idx = a * 5 + t
            t_mbid = f"trambid_{t_idx:04d}"
            t_name = f"Track {t_idx}"
            tracks.append((a_mbid, a_name, t_mbid, t_name))

    # 1. Generate 4000 training events in 2008-01-01 to 2009-04-30
    for _ in range(4000):
        u = users[rng.integers(0, len(users))]
        a_mbid, a_name, t_mbid, t_name = tracks[rng.integers(0, len(tracks))]
        month = rng.integers(1, 13)
        year = 2008 if month < 10 else 2009
        m_val = month if year == 2008 else rng.integers(1, 5)
        day = rng.integers(1, 28)
        hour = rng.integers(0, 24)
        minute = rng.integers(0, 60)
        ts = f"{year}-{m_val:02d}-{day:02d}T{hour:02d}:{minute:02d}:00Z"
        lines.append(f"{u}\t{ts}\t{a_mbid}\t{a_name}\t{t_mbid}\t{t_name}")

    # 2. Generate 1000 test events in May-June 2009 (2009-05-01 to 2009-06-30)
    for _ in range(1000):
        u = users[rng.integers(0, len(users))]
        a_mbid, a_name, t_mbid, t_name = tracks[rng.integers(0, len(tracks))]
        m_val = rng.integers(5, 7)  # month 5 or 6
        day = rng.integers(1, 28)
        hour = rng.integers(0, 24)
        minute = rng.integers(0, 60)
        ts = f"2009-{m_val:02d}-{day:02d}T{hour:02d}:{minute:02d}:00Z"
        lines.append(f"{u}\t{ts}\t{a_mbid}\t{a_name}\t{t_mbid}\t{t_name}")

    sample_tsv = root / "data/raw/lastfm-dataset-1K/sample_eval.tsv"
    sample_tsv.parent.mkdir(parents=True, exist_ok=True)
    sample_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ingest_report = ingest(root, sample_tsv, "1k")
    print(f"Ingestion successful: clean_rows={ingest_report['clean_rows']}, users={ingest_report['users']}, items={ingest_report['items']}")


if __name__ == "__main__":
    main()
