"""Synthetic fixtures only: no fabricated tags are used in real-data reports."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from lastfm.discovery import artist_profile_cosine, recommend
from lastfm.dormancy import extract_dormant
from lastfm.factorization import als_confidence, fit_als, train
from lastfm.ingest import ingest
from lastfm.matrix import matrix
from lastfm.snapshots import read_db
from lastfm.tags import TagIndex, import_tags, proposal_overlap, tfidf, weighted_jaccard


@pytest.fixture
def prepared(tmp_path):
    source=tmp_path/'plays.tsv'
    # Jan 31 is exactly 90 days before May 1: it belongs to active, not historical.
    entries=[('u1','2009-01-01','a1','t1'),('u1','2009-01-02','a1','t1'),
             ('u1','2009-01-31','a2','t2'),('u1','2009-05-01','a9','future'),
             ('u2','2009-01-03','a1','t1'),('u2','2009-04-01','a1','t3'),
             ('u3','2009-01-01','a1','t3'),('u3','2009-04-02','a3','t4'),
             ('u3','2009-04-03','a4','t5')]
    source.write_text(''.join(f'{u}\t{d}T00:00:00Z\t{a}\tArtist {a}\t{t}\tTrack {t}\n' for u,d,a,t in entries))
    ingest(tmp_path,source,'1k')
    meta=matrix(tmp_path,'1k','2009-05-01T00:00:00Z',kappa=4,recent_days=90)
    return tmp_path,Path(meta['path'])


def test_als_uses_full_confidence_including_kappa_zero():
    counts=csr_matrix([[1,0,2],[0,3,0]],dtype=float)
    confidence=als_confidence(counts,0)
    assert confidence.nnz==3
    assert (confidence.data==1).all()
    assert als_confidence(counts,4)[0,2]==pytest.approx(1+4*math.log(3))
    with pytest.raises(ValueError): als_confidence(counts,-1)


def test_exact_als_item_solution_matches_dense_normal_equations():
    counts=csr_matrix([[3,0,1],[0,2,1],[1,0,0]],dtype=float)
    model=fit_als(counts,kappa=4,factors=2,regularization=.2,iterations=4,
                  threads=1,seed=7,exact=True)
    # The final item half-step must include C=1 for ALL unobserved user pairs.
    dense=counts.toarray()
    confidence=1+4*np.log1p(dense)
    preference=(dense>0).astype(float)
    x=model.user_factors.astype(float)
    for item in range(3):
        a=x.T@(confidence[:,item,None]*x)+.2*np.eye(2)
        b=x.T@(confidence[:,item]*preference[:,item])
        np.testing.assert_allclose(model.item_factors[item],np.linalg.solve(a,b),rtol=2e-4,atol=2e-4)


def test_tfidf_hand_calculation_and_empty_rows():
    counts=csr_matrix([[2,1,0],[0,1,0],[0,0,1]],dtype=float)
    raw,normalized,idf=tfidf(counts)
    assert raw[0,0]==pytest.approx(2/3*math.log(3))
    assert raw[0,1]==pytest.approx(1/3*math.log(1.5))
    np.testing.assert_allclose(np.asarray(normalized.sum(axis=1)).ravel(),1)
    raw,normalized,_=tfidf(csr_matrix([[1],[1]],dtype=float))
    assert raw.nnz==normalized.nnz==0 # ubiquitous tag has zero IDF
    raw,normalized,_=tfidf(csr_matrix([[0,0],[1,0]],dtype=float))
    assert np.isfinite(normalized.data).all()


def test_overlap_metrics_are_distinct_and_defined_at_zero():
    a={'x':.6,'y':.4}; b={'x':.2,'z':.8}
    assert weighted_jaccard(a,b)==pytest.approx(.2/1.8)
    assert weighted_jaccard(a,b)==weighted_jaccard(b,a)
    assert proposal_overlap(a,b)==pytest.approx(.6/3)
    assert proposal_overlap(b,a)==pytest.approx(.2/3)
    assert weighted_jaccard({}, {})==proposal_overlap({}, {})==0
    assert weighted_jaccard({'x':0},{'x':0})==0
    assert proposal_overlap({'x':1},{'y':1})==0
    with pytest.raises(ValueError): weighted_jaccard({'x':-1},{})


def test_tag_import_temporal_filter_and_matrix_binding(prepared):
    root,matrix_dir=prepared
    source=root/'tags.tsv'
    source.write_text('item_id\ttag\tcount\tassigned_at\n'
        'mbid:t1\tDream Pop\t2\t2009-01-01T00:00:00Z\n'
        'mbid:t1\t dream   pop \t1\t2009-02-01T00:00:00Z\n'
        'mbid:t3\tdream pop\t2\t2009-02-01T00:00:00Z\n'
        'mbid:t4\tfuture-tag\t1\t2009-05-01T00:00:00Z\n'
        'unknown\trock\t1\t2009-01-01T00:00:00Z\n')
    report=import_tags(root,source,matrix_dir=matrix_dir)
    assert report['future_rows_excluded']==1 and report['unknown_item_rows_excluded']==1
    assert report['tagged_items']==2 and report['vocabulary_size']==1
    index=TagIndex(Path(report['path']),report['matrix_fingerprint'])
    assert index.counts[0,0]==3
    assert index.vocabulary==['dream pop']
    with pytest.raises(ValueError): TagIndex(Path(report['path']),'wrong fingerprint')


def test_undated_and_invalid_tags_fail_explicitly(prepared):
    root,matrix_dir=prepared
    source=root/'tags.tsv'
    source.write_text('item_id\ttag\tcount\nmbid:t1\tchill\t1\n')
    with pytest.raises(ValueError,match='assigned_at'): import_tags(root,source,matrix_dir=matrix_dir)
    with pytest.raises(ValueError,match='before'): import_tags(root,source,matrix_dir=matrix_dir,metadata_as_of='2009-05-01T00:00:00Z')
    report=import_tags(root,source,matrix_dir=matrix_dir,metadata_as_of='2009-01-01T00:00:00Z')
    assert report['tagged_items']==1
    source.write_text('item_id\ttag\tcount\nmbid:t1\tchill\tNaN\n')
    with pytest.raises(ValueError,match='invalid'): import_tags(root,source,matrix_dir=matrix_dir,metadata_as_of='2009-01-01T00:00:00Z')
    source.write_text('item_id\ttag\tcount\tassigned_at\nmbid:t1\tchill\t1\t2009-01-01\n')
    with pytest.raises(ValueError,match='invalid'): import_tags(root,source,matrix_dir=matrix_dir)


def test_dormancy_threshold_and_exact_boundary(prepared):
    root,matrix_dir=prepared
    report=extract_dormant(root,matrix_dir=matrix_dir,min_historical_plays=2)
    assert report['candidates']==1 and report['users']==1
    with read_db() as db:
        rows=db.execute('SELECT user_id,item_id,historical_plays,active_plays,days_since_last_play FROM read_parquet(?)',
                        [str(Path(report['path'])/'candidates.parquet')]).fetchall()
    assert rows==[('u1','mbid:t1',2,0,119.)]
    empty=extract_dormant(root,matrix_dir=matrix_dir,min_historical_plays=100)
    assert empty['candidates']==0
    with pytest.raises(ValueError): extract_dormant(root,dataset='360k')


def test_artist_cosine_uses_actual_metadata():
    actual=artist_profile_cosine(['a','b','c',None],['a','b'],[2,2])
    np.testing.assert_allclose(actual,[1/math.sqrt(2),1/math.sqrt(2),0,0])


def test_training_reload_unseen_candidates_and_fusion(prepared):
    root,matrix_dir=prepared
    trained=train(root,matrix_dir=matrix_dir,factors=3,iterations=3,threads=1)
    model_dir=Path(trained['path'])
    first=recommend(root,'u1',model_dir=model_dir,k=3,candidates=5)
    second=recommend(root,'u1',model_dir=model_dir,k=3,candidates=5)
    assert first==second
    assert {r['item_id'] for r in first['recommendations']}=={'mbid:t3','mbid:t4','mbid:t5'}
    assert all(r['tag_score'] is None for r in first['recommendations'])
    assert first['seen_items_excluded']==2
    source=root/'tags.tsv'
    source.write_text('item_id\ttag\tcount\nmbid:t1\tdream\t1\nmbid:t3\tdream\t1\nmbid:t4\trock\t1\n')
    tags=import_tags(root,source,matrix_dir=matrix_dir,metadata_as_of='2009-01-01T00:00:00Z')
    fused=recommend(root,'u1',model_dir=model_dir,tags_dir=Path(tags['path']),k=3,candidates=5)
    assert fused['mode']=='fusion'
    assert fused['recommendations'][0]['item_id']=='mbid:t3'
    assert fused['recommendations'][0]['shared_tags']==['dream']
    with pytest.raises(ValueError,match='Unknown training user'): recommend(root,'new-user',model_dir=model_dir)
    with pytest.raises(ValueError,match='requires'): recommend(root,'u1',model_dir=model_dir,mode='fusion')
