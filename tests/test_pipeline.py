import json
import math
import duckdb
import pytest
from scipy.sparse import load_npz
from lastfm.ingest import ingest, connect
from lastfm.matrix import matrix, preference_confidence


def test_temporal_ingestion_and_matrix(tmp_path):
    source = tmp_path / "events.tsv"
    rows = [
        "1111111111111111111111111111111111111111\t2009-01-01T00:00:00Z\ta1\tArtist\tt1\tSong",
        "1111111111111111111111111111111111111111\t2009-01-01T00:00:00Z\ta1\tArtist\tt1\tSong", # duplicate
        "1111111111111111111111111111111111111111\t2009-04-01T00:00:00Z\ta1\tArtist\tt1\tSong",
        "2222222222222222222222222222222222222222\t2009-04-02T00:00:00Z\t\tOther\t\tUntitled",
        "1111111111111111111111111111111111111111\t2009-05-01T00:00:00Z\ta1\tArtist\tt2\tFuture", # exclusive cutoff
        "1111111111111111111111111111111111111111\tbroken\ta1\tArtist\tt1\tSong",
        "1111111111111111111111111111111111111111\t2013-01-01T00:00:00Z\ta1\tArtist\tt1\tSong",
        "too\tmany\tfields\tin\tthis\tbad\tline",
    ]
    source.write_text("\n".join(rows)+"\n",encoding="utf-8")
    report = ingest(tmp_path,source,"1k")
    assert report["clean_rows"] == 4
    assert report["duplicates_removed"] == 1
    assert report["validation_rejected_rows"] == 2
    assert report["parse_rejected_rows"] == 1
    assert report["partitions"] == 3
    ingest(tmp_path,source,"1k") # replace snapshot without doubling events
    result = matrix(tmp_path,"1k","2009-05-01T00:00:00Z",40,90,True)
    assert result["shape"] == [2,2]
    assert result["observed_pairs"] == 2
    with connect(tmp_path) as db:
        edge = db.execute("SELECT play_count,active_plays,historical_plays,confidence FROM recsys.edges_1k WHERE user_id='1111111111111111111111111111111111111111'").fetchone()
        assert edge[:3] == (2,1,1)
        assert edge[3] == pytest.approx(1+40*math.log(3))
        plan = db.execute("EXPLAIN SELECT * FROM recsys.events_1k WHERE year=2009 AND month=1").fetchone()[1]
        assert "File Filters" in plan
    from pathlib import Path
    p = load_npz(Path(result["path"])/"preferences.npz")
    delta = load_npz(Path(result["path"])/"confidence_delta.npz")
    assert p[0,1] == 0 and delta[0,1] == 0 # missing confidence = 1 + delta


def test_360k_separate_artist_counts(tmp_path):
    source=tmp_path/"artists.tsv"
    source.write_text("1111111111111111111111111111111111111111\ta1\tArtist\t3\n1111111111111111111111111111111111111111\ta1\tArtist\t4\n2222222222222222222222222222222222222222\t\tOther\t2\n3333333333333333333333333333333333333333\ta3\tBad\t-1\n")
    report=ingest(tmp_path,source,"360k")
    assert report["clean_rows"] == 2
    assert report["validation_rejected_rows"] == 1
    result=matrix(tmp_path,"360k")
    assert result["item_type"] == "artist"
    with connect(tmp_path) as db:
        assert db.execute("SELECT play_count FROM recsys.edges_360k WHERE user_id='1111111111111111111111111111111111111111'").fetchone()[0] == 7
    with pytest.raises(ValueError): matrix(tmp_path,"360k","2009-05-01T00:00:00Z")


def test_confidence_and_cutoff_validation(tmp_path):
    assert preference_confidence(0) == (0,1)
    assert preference_confidence(9)[1] == pytest.approx(93.10340371976183)
    for k in [-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError): preference_confidence(1,k)
    with pytest.raises(ValueError): matrix(tmp_path,"1k")
    with pytest.raises(ValueError): matrix(tmp_path,"1k","2009-05-01")


def test_invalid_identities_and_failed_publication(tmp_path):
    source = tmp_path/'source.tsv'
    source.write_text('1111111111111111111111111111111111111111\t2009-04-01T00:00:00Z\ta1\tArtist\tt1\tSong\n')
    ingest(tmp_path,source,'1k')
    # Name fallback must not create synthetic identities from missing names.
    source.write_text('1111111111111111111111111111111111111111\t2009-04-01T00:00:00Z\t\t\t\t\n')
    with pytest.raises(ValueError): ingest(tmp_path,source,'1k')
    with connect(tmp_path) as db:
        assert db.execute('SELECT count(*) FROM recsys.events_1k').fetchone()[0]==1


def test_fractional_counts_quarantined(tmp_path):
    source=tmp_path/'source.tsv'
    source.write_text('1111111111111111111111111111111111111111\ta1\tArtist\t2\n2222222222222222222222222222222222222222\ta2\tOther\t1.5\n')
    report=ingest(tmp_path,source,'360k')
    assert report['validation_rejected_rows']==1
    assert report['clean_rows']==1


def test_date_strings_cannot_become_360k_users(tmp_path):
    source=tmp_path/'source.tsv'
    source.write_text('1111111111111111111111111111111111111111\ta1\tArtist\t2\nsep 20, 2008\ta2\tOther\t5\n')
    report=ingest(tmp_path,source,'360k')
    assert report['validation_rejected_rows']==1
    assert report['users']==1
