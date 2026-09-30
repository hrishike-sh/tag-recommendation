"""Tests for proposal Eq. 9, chronological eligibility, blending and API wiring."""
from datetime import datetime, timedelta, timezone
import json
import math
import numpy as np
import pytest
import scipy.sparse as sp

from lastfm.ingest import connect
from lastfm.nostalgia import (listening_history, nostalgia_score, score_dormant,
                              DynamicTemporalArbiter)

END = datetime(2009, 5, 1, tzinfo=timezone.utc)
CUTOFF = END.isoformat()


def history_row(item, historical, active=0, age=100, first_age=200):
    return dict(item_id=item, historical_plays=historical, active_plays=active,
                first_played_at=END-timedelta(days=first_age),
                last_played_at=END-timedelta(days=age))


def test_equation_nine_and_half_life():
    assert nostalgia_score(9, 90, .4, 90) == pytest.approx(math.log(10)*.5*.4)
    assert nostalgia_score(9, 0, .4) == 0
    assert nostalgia_score(9, 90, 0) == 0
    assert nostalgia_score(9, 180, .4) > nostalgia_score(9, 90, .4)
    for value in [0, -1, float('nan'), float('inf')]:
        with pytest.raises(ValueError): nostalgia_score(9, 90, .4, value)


def test_cutoff_and_active_boundary_no_future_leakage(tmp_path):
    with connect(tmp_path) as db:
        db.execute('CREATE TABLE recsys.events_1k(user_id VARCHAR,item_id VARCHAR,played_at TIMESTAMPTZ)')
        rows = [('u','old',END-timedelta(days=100))]*5
        rows += [('u','boundary',END-timedelta(days=90)), ('u','old',END),
                 ('u','future',END+timedelta(days=1)),('v','other',END-timedelta(days=1))]
        db.executemany('INSERT INTO recsys.events_1k VALUES (?,?,?)', rows)
    h = listening_history(tmp_path, 'u', CUTOFF)
    assert {r['item_id'] for r in h} == {'old','boundary'}
    assert next(r for r in h if r['item_id']=='boundary')['active_plays'] == 1
    candidates = score_dormant(h, ['old','boundary'], sp.csr_matrix([[1],[1]]), CUTOFF)
    assert set(candidates) == {0}
    assert candidates[0]['historical_plays'] == 5
    assert candidates[0]['days_dormant'] == 100


def test_active_context_not_historical_and_threshold():
    h = [history_row('old',5), history_row('recent',0,1,age=1),
         history_row('few',4), history_row('returned',20,1,age=2),
         history_row('unrelated',10)]
    tags = sp.csr_matrix([[3,1],[1,0],[1,0],[1,0],[0,1]])
    d = score_dormant(h, ['old','recent','few','returned','unrelated'], tags, CUTOFF)
    assert set(d) == {0,4}
    # Eq.7: shared/union = 1/2, mean shared candidate relevance = 3/4.
    assert d[0]['tag_similarity'] == pytest.approx(.375)
    assert d[4]['score'] == 0
    no_context = score_dormant([h[0]], ['old'], sp.csr_matrix([[1]]), CUTOFF)
    assert no_context[0]['score'] == 0


def test_blend_endpoints_seen_exclusion_and_explanation_reconstruction():
    ids=['old','active','new','new2']
    h=[history_row('old',9), history_row('active',0,2,age=1,first_age=1)]
    d=score_dormant(h,ids,sp.csr_matrix([[1],[1],[1],[1]]),CUTOFF)
    arb=DynamicTemporalArbiter()
    for alpha, expected in [(0,['old']),(1,['new','new2']),(.5,['old','new','new2'])]:
        out=arb.recommend([100,100,2,1],ids,h,d,CUTOFF,alpha=alpha,k=99,tag_names={0:'rock'})
        assert [r['item_id'] for r in out['recommendations']] == expected
        for r in out['recommendations']:
            assert r['score'] == pytest.approx(sum(r['stream_contributions'].values()))
            assert r['explanation_tuple'].alpha == alpha
        json.dumps(out,allow_nan=False)
    assert arb.recommend([0,0,2,1],ids,h,d,CUTOFF)['alpha'] == pytest.approx(.8)
    h[1]['first_played_at']=END-timedelta(days=150)
    assert arb.recommend([0,0,2,1],ids,h,d,CUTOFF)['alpha'] == pytest.approx(.2)
    assert arb.recommend([0,0,2,1],ids,h,{},CUTOFF)['alpha'] == 1
    assert arb.recommend([0,0,0,0],ids,h,d,CUTOFF)['alpha'] == 0
    assert arb.recommend([0,0,2,1],ids,h,d,CUTOFF,k=0)['recommendations'] == []
    assert arb.recommend([],[],[],{},CUTOFF)['recommendations'] == []
    for alpha in [-1,1.1,float('nan')]:
        with pytest.raises(ValueError): arb.recommend([0,0,2,1],ids,h,d,CUTOFF,alpha=alpha)


def test_historical_tag_assignments_filtered(tmp_path):
    from lastfm.tag_msvd import build_tag_matrix
    import duckdb
    raw=tmp_path/'data/raw/hetrec2011-lastfm-2k'; raw.mkdir(parents=True)
    (raw/'artists.dat').write_text('id\tname\n1\tArtist\n')
    (raw/'user_taggedartists.dat').write_text('userID\tartistID\ttagID\tday\tmonth\tyear\n1\t1\t1\t30\t4\t2009\n2\t1\t2\t1\t5\t2009\n')
    path=tmp_path/'meta.parquet'
    with duckdb.connect() as db:
        db.execute("COPY (SELECT 'old' AS item_id, 'Artist' AS artist_name) TO ? (FORMAT PARQUET)",[str(path)])
    matrix, vocab=build_tag_matrix(tmp_path,['old'],path,min_tag_freq=1,cutoff_iso=CUTOFF)
    assert vocab == {1:0}
    assert matrix.toarray().tolist() == [[1]]


def test_engine_wiring_without_expensive_training(tmp_path,monkeypatch):
    import lastfm.recommend_engine as engine
    import lastfm.nostalgia as nostalgia
    import duckdb
    ids=['old','active','new']
    pref=sp.csr_matrix([[1,1,0]],dtype=np.float32)
    monkeypatch.setattr(engine,'split_train_val_test',lambda *a,**k: {'test_split':dict(
        user_ids=['u'],item_ids=ids,P_train=pref,D_train=pref*2)})
    class Model:
        def __init__(self,**kw): pass
        def fit(self,*a,**kw):
            self.user_factors=np.array([[1.]])
            self.item_factors=np.array([[1.],[2.],[3.]])
    monkeypatch.setattr(engine,'ImplicitMSVD',Model)
    snapshot=tmp_path/'snapshot'; snapshot.mkdir()
    reports=tmp_path/'artifacts/reports'; reports.mkdir(parents=True)
    (reports/'ingest-1k.json').write_text(json.dumps({'snapshot':str(snapshot)}))
    with duckdb.connect() as db:
        db.execute("COPY (SELECT 'old' AS item_id, 'Artist' AS artist_name, 'Song' AS track_name) TO ? (FORMAT PARQUET)",[str(snapshot/'track_metadata.parquet')])
    monkeypatch.setattr(nostalgia,'listening_history',lambda *a: [history_row('old',9),history_row('active',0,1,age=1)])
    missing=engine.get_recommendations_with_explanations(tmp_path,'u')
    assert missing['tags_available'] is False and missing['alpha']==1
    assert [r['item_id'] for r in missing['recommendations']]==['new']
    raw=tmp_path/'data/raw/hetrec2011-lastfm-2k'; raw.mkdir(parents=True)
    (raw/'user_taggedartists.dat').touch()
    (raw/'tags.dat').write_text('tagID\ttagValue\n1\trock\n')
    def tags(*a,**kw):
        assert kw['cutoff_iso']=='2009-05-01T00:00:00Z'
        return sp.csr_matrix([[1],[1],[1]]),{1:0}
    monkeypatch.setattr(engine,'build_tag_matrix',tags)
    result=engine.get_recommendations_with_explanations(tmp_path,'u',alpha=0)
    json.dumps(result,allow_nan=False)
    assert result['recommendations'][0]['kind']=='nostalgia'
    assert result['recommendations'][0]['track_name']=='Song'
    assert result['recommendations'][0]['explanation_tuple'].shared_tags==('rock',)
    with pytest.raises(ValueError,match='Unknown user'):
        engine.get_recommendations_with_explanations(tmp_path,'missing')
