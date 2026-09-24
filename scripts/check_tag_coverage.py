"""Inspect HetRec 2011 dataset and measure exact mapping to Last.fm 1K tracks/artists."""

import csv
import json
from pathlib import Path
import duckdb

def main():
    root = Path.cwd()
    raw_tags = root / "data/raw/hetrec2011-lastfm-2k"
    
    # 1. Parse tags
    tags_map = {} # tag_id -> tag_value
    with open(raw_tags / "tags.dat", "r", encoding="latin-1", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                tags_map[int(row[0])] = row[1].strip().lower()
                
    # 2. Parse artists in HetRec
    hetrec_artists = {} # artist_id -> artist_name
    with open(raw_tags / "artists.dat", "r", encoding="latin-1", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                hetrec_artists[int(row[0])] = row[1].strip()
                
    # 3. Parse user_taggedartists
    tag_assignments = []
    with open(raw_tags / "user_taggedartists.dat", "r", encoding="latin-1", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) >= 3:
                u, a, t = int(row[0]), int(row[1]), int(row[2])
                tag_assignments.append((u, a, t))
                
    print(f"Loaded {len(tags_map):,} distinct tags.")
    print(f"Loaded {len(hetrec_artists):,} HetRec artists.")
    print(f"Loaded {len(tag_assignments):,} tagging records.")
    
    # Check artist mapping to Last.fm 1K
    # In Last.fm 1K, track_metadata.parquet has artist_name and item_id
    meta_path = root / "data/lake/1k/529b8f83aada-3f0122fb/track_metadata.parquet"
    db = duckdb.connect(str(root / "data/lastfm.duckdb"), read_only=True)
    l1k_artists = db.execute(f"""
        SELECT lower(trim(artist_name)) AS artist_norm, count(DISTINCT item_id) AS tracks_count
        FROM read_parquet('{meta_path.as_posix()}')
        WHERE artist_name IS NOT NULL
        GROUP BY 1
    """).fetchall()
    
    l1k_artist_dict = {a: t_cnt for a, t_cnt in l1k_artists}
    print(f"\nLast.fm 1K has {len(l1k_artist_dict):,} distinct normalized artist names.")
    
    hetrec_artist_names = {a_name.lower(): a_id for a_id, a_name in hetrec_artists.items()}
    
    # Overlap
    matched_artists = set(l1k_artist_dict.keys()) & set(hetrec_artist_names.keys())
    matched_tracks = sum(l1k_artist_dict[a] for a in matched_artists)
    total_tracks = sum(l1k_artist_dict.values())
    
    print(f"Matched Artists with HetRec: {len(matched_artists):,} / {len(l1k_artist_dict):,} ({len(matched_artists)/len(l1k_artist_dict)*100:.2f}%)")
    print(f"Track Coverage via Matched Artists: {matched_tracks:,} / {total_tracks:,} ({matched_tracks/total_tracks*100:.2f}%)")
    
    # Event coverage in 1K
    matched_names_list = list(matched_artists)
    db.execute("CREATE TEMP TABLE matched_art AS SELECT unnest(?) AS artist_norm", [matched_names_list])
    matched_events = db.execute(f"""
        SELECT count(*)
        FROM recsys.events_1k e
        JOIN read_parquet('{meta_path.as_posix()}') m USING(item_id)
        JOIN matched_art ma ON lower(trim(m.artist_name)) = ma.artist_norm
    """).fetchone()[0]
    total_events = db.execute("SELECT count(*) FROM recsys.events_1k").fetchone()[0]
    
    print(f"Listening Event Coverage: {matched_events:,} / {total_events:,} ({matched_events/total_events*100:.2f}%)")

if __name__ == "__main__":
    main()
