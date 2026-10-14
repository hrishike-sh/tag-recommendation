import json
import math
from datetime import datetime, timezone
import numpy as np
import pytest
import scipy.sparse as sp
from lastfm.ingest import ingest
from lastfm.service import RecommendationService


@pytest.fixture
def prepared(tmp_path):
    source=tmp_path/'events.tsv'
    rows=[]
    for user,item,date,count in [('u','old','2008-01-01',6),('u','active','2009-04-20',2),
          ('v','old','2008-02-01',5),('v','new','2009-04-10',4),('v','other','2009-03-01',3),
          ('u','old','2009-05-03',1),('u','new','2009-05-04',1),
          ('u','future','2009-06-01',1),('unknown','new','2009-05-10',1),
          ('v','active','2009-05-03',1)]:
        rows += [f'{user}\t{date}T00:00:{i:02d}Z\tartist\tFixture Artist\t{item}\t{item}' for i in range(count)]
    source.write_text('\n'.join(rows)+'\n')
    ingest(tmp_path,source,'1k')
    tags=tmp_path/'data/raw/hetrec2011-lastfm-2k';tags.mkdir(parents=True)
    (tags/'artists.dat').write_text('id\tname\n1\tFixture Artist\n')
    (tags/'tags.dat').write_text('tagID\ttagValue\n1\trock\n2\tfuturetag\n')
    (tags/'user_taggedartists.dat').write_text('userID\tartistID\ttagID\tday\tmonth\tyear\n'+'1\t1\t1\t1\t1\t2008\n'*6+'1\t1\t2\t1\t6\t2009\n'*6)
    return RecommendationService.open(tmp_path,factors=3,iterations=2)


def test_service_roundtrip_and_no_future_leakage(prepared,monkeypatch):
    service=prepared
    assert not any('future' in item for item in service.items)
    assert service.tags.shape[1]==1
    original=service.recommend('u',alpha=.5)
    assert {r['kind'] for r in original['recommendations']}=={'nostalgia','novelty'}
    json.dumps(original,allow_nan=False)
    for row in original['recommendations']:
        assert sum(row['stream_contributions'].values())==pytest.approx(row['score'])
    import lastfm.service as module
    monkeypatch.setattr(module.ImplicitMSVD,'fit',lambda *a,**kw:pytest.fail('cached request retrained'))
    loaded=RecommendationService.open(service.root,factors=3,iterations=2)
    assert loaded.recommend('u',alpha=.5)['recommendations']==original['recommendations']
    with pytest.raises(ValueError,match='Unknown user'): loaded.recommend('missing')
    with pytest.raises(ValueError): loaded.recommend('u',alpha=float('nan'))


def test_metric_values_and_missing_evidence():
    pytest.importorskip('sklearn')
    from lastfm.service_evaluation import ranking_metrics
    tags=sp.csr_matrix([[1,0],[0,1],[0,0]])
    out=ranking_metrics([0,1],{1,2},{1},np.array([1,3,0]),3,tags,2)
    assert out['ndcg']==pytest.approx((1/math.log2(3))/(1+1/math.log2(3)))
    assert out['novelty_bits']==pytest.approx(.5)
    assert out['diversity']==pytest.approx(1)
    assert out['nostalgia_recall']==1
    assert ranking_metrics([2],{0},set(),np.zeros(3),3,tags,2)['diversity'] is None
    empty=ranking_metrics([],set(),set(),np.zeros(3),3,tags,2)
    assert empty['ndcg'] is None and empty['nostalgia_recall'] is None
    assert empty['novelty_bits'] is None
    assert ranking_metrics([0],{0,1},set(),np.zeros(3),3,tags,2)['ndcg']==pytest.approx(1/(1+1/math.log2(3)))
    with pytest.raises(ValueError):ranking_metrics([0,0],{0},set(),np.zeros(3),3,tags,2)


def test_evaluation_and_ann(prepared):
    pytest.importorskip('faiss')
    pytest.importorskip('sklearn')
    from lastfm.service_evaluation import evaluate_service,benchmark_service,holdout
    service=prepared
    report=evaluate_service(service,k_list=[1,5],max_users=None)
    assert report['evaluated_users']==2
    assert report['exclusions']['excluded_unknown_item_pairs']==1
    assert report['exclusions']['excluded_unknown_user_pairs']==1
    assert report['metrics']['5']['nostalgia_recall']==1
    assert report['metrics']['5']['nostalgia_recall_users']==1
    assert report['metrics']['5']['diversity']==pytest.approx(0)
    json.dumps(report,allow_nan=False)
    with pytest.raises(ValueError):holdout(service,'2008-01-01T00:00:00Z')
    ann=service.recommend('u',retrieval='hnsw',alpha=.5)
    assert len({r['item_id'] for r in ann['recommendations']})==len(ann['recommendations'])
    assert any(r['kind']=='nostalgia' for r in ann['recommendations'])
    bench=benchmark_service(service,users=['u','v'],k=2,repeats=1)
    assert 0<=bench['overlap_at_k']<=1
    assert (service.folder/'hnsw.faiss').exists()
    json.dumps(bench,allow_nan=False)


def test_hnsw_inner_product_and_exclusion(tmp_path):
    pytest.importorskip('faiss')
    from lastfm.retrieval import CandidateIndex
    vectors=np.array([[10,0],[1,1],[0,1]],dtype=np.float32)
    index=CandidateIndex(vectors,tmp_path/'test.faiss')
    assert index.search(np.array([1,0]),2)==[0,1]
    assert index.search(np.array([1,0]),2,{0})==[1,2]
    assert CandidateIndex(vectors,tmp_path/'test.faiss').search(np.array([1,0]),1)==[0]
