import json
import threading
import time
from pathlib import Path
import duckdb
from lastfm.web import create_app


def fixture(tmp_path):
    reports=tmp_path/'artifacts/reports';reports.mkdir(parents=True)
    snapshot=tmp_path/'snapshot';(snapshot/'events').mkdir(parents=True)
    matrix=tmp_path/'matrix';matrix.mkdir()
    (reports/'ingest-1k.json').write_text(json.dumps({'snapshot':str(snapshot),'clean_rows':6,'users':1,'items':2}))
    (reports/'matrix-1k.json').write_text(json.dumps({'path':str(matrix)}))
    with duckdb.connect() as db:
        db.execute("COPY (SELECT 'u' AS user_id) TO ? (FORMAT PARQUET)",[str(matrix/'users.parquet')])
        db.execute("COPY (SELECT 'old' AS item_id,'Artist' AS artist_name,'Old song' AS track_name UNION ALL SELECT 'recent','Artist','Recent song') TO ? (FORMAT PARQUET)",[str(snapshot/'track_metadata.parquet')])
        db.execute("CREATE TABLE events(user_id VARCHAR,item_id VARCHAR,played_at TIMESTAMPTZ)")
        db.executemany('INSERT INTO events VALUES (?,?,?)',[('u','old','2008-01-01T00:00:00Z')]*5+[('u','recent','2009-04-20T00:00:00Z')])
        db.execute('COPY events TO ? (FORMAT PARQUET)',[str(snapshot/'events/e.parquet')])
    return create_app(tmp_path).test_client()


def test_web_queries_and_controls(tmp_path):
    c=fixture(tmp_path)
    assert c.get('/').status_code==200
    assert c.get('/assets/app.js').status_code==200
    assert c.get('/api/status',headers={'Host':'external.test'}).status_code==403
    assert c.get('/api/users').json['users']==[{'user_id':'u'}]
    history=c.get('/api/history?user=u').json
    assert history['total_plays']==6 and history['rows'][0]['track']=='Old song'
    dormant=c.get('/api/dormant?user=u').json
    assert dormant['total']==1 and dormant['rows'][0]['historical_plays']==5
    assert c.get('/api/dormant?user=u&threshold=6').json['total']==0
    assert c.get('/api/history?user=missing').status_code==400
    assert c.get('/api/dormant?dataset=360k&user=u').status_code==400
    assert c.post('/api/jobs',json={'user':'u'}).status_code==403
    token=c.get('/api/status').json['token']
    assert c.post('/api/jobs',headers={'X-Local-Token':token},json={'user':'u','alpha':2}).status_code==400
    assert c.get('/api/jobs/missing/result').status_code==404


def test_job_lifecycle_and_result(tmp_path,monkeypatch):
    import lastfm.web as web
    c=fixture(tmp_path);gate=threading.Event()
    class Process:
        def __init__(self,command,**kwargs):
            self.folder=Path(command[-1])
            assert command[1:4]==['-u','-m','lastfm.web_worker']
        def wait(self):
            gate.wait(5)
            (self.folder/'result.json').write_text(json.dumps({'recommendations':[]}))
            return 0
    monkeypatch.setattr(web.subprocess,'Popen',Process)
    headers={'X-Local-Token':c.get('/api/status').json['token']}
    r=c.post('/api/jobs',headers=headers,json={'user':'u','alpha':.6});assert r.status_code==202
    assert c.post('/api/jobs',headers=headers,json={'user':'u'}).status_code==409
    gate.set()
    for _ in range(100):
        data=c.get('/api/jobs').json['jobs'][0]
        if data['status']=='completed':break
        time.sleep(.01)
    assert data['status']=='completed' and data['result_available']
    assert c.get('/api/jobs/'+r.json['id']+'/result').json=={'recommendations':[]}
