"""Local REPRISE dashboard over immutable Last.fm snapshots."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import uuid

import duckdb
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
from .ingest import literal

CUTOFF = datetime(2009, 5, 1, tzinfo=timezone.utc)


def create_app(root):
    root = Path(root).resolve()
    app = Flask(__name__, static_folder='ui', static_url_path='/assets')
    app.config['MAX_CONTENT_LENGTH'] = 16384
    token = secrets.token_urlsafe(32)
    jobs = {}; lock = threading.Lock()

    def report(kind, dataset):
        path = root / f'artifacts/reports/{kind}-{dataset}.json'
        return json.loads(path.read_text()) if path.exists() else None

    def selected():
        dataset = request.args.get('dataset', '1k')
        if dataset not in {'1k','360k'}: raise ValueError('Unknown dataset')
        return dataset

    def source(kind, dataset):
        meta = report(kind,dataset)
        if not meta: raise ValueError('Prepare the dataset and matrix first')
        path = Path(meta['snapshot' if kind=='ingest' else 'path'])
        if not path.is_absolute(): path = root/path
        if not path.is_dir(): raise ValueError('Local snapshot is missing; rebuild the dataset')
        return path

    def query(sql, args=()):
        with duckdb.connect() as db:
            db.execute("SET TimeZone='UTC'")
            db.execute("SET threads=2")
            db.execute("SET memory_limit='1GB'")
            res = db.execute(sql,args)
            cols = [c[0] for c in res.description]
            return [dict(zip(cols, (v.isoformat() if isinstance(v,datetime) else v for v in r))) for r in res.fetchall()]

    @app.before_request
    def protect():
        if request.host.split(':')[0] not in {'127.0.0.1','localhost'}: return jsonify(error='Local access only'),403
        if request.method=='POST' and request.headers.get('X-Local-Token')!=token:
            return jsonify(error='Refresh the page before starting a run'),403

    @app.after_request
    def headers(response):
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Content-Security-Policy']="default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        return response

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc,HTTPException): return jsonify(error=exc.description),exc.code
        if isinstance(exc,(ValueError,FileNotFoundError)): return jsonify(error=str(exc)),400
        app.logger.exception('Dashboard error')
        return jsonify(error='Unable to complete this request. Check the server log.'),500

    @app.get('/')
    def index(): return app.send_static_file('index.html')

    @app.get('/api/status')
    def status():
        return jsonify(token=token,cutoff=CUTOFF.isoformat(),datasets={d:{
            'ingest':report('ingest',d),'matrix':report('matrix',d)} for d in ('1k','360k')},
            tags_available=(root/'data/raw/hetrec2011-lastfm-2k/user_taggedartists.dat').exists())

    @app.get('/api/users')
    def users():
        path=source('matrix',selected())/'users.parquet'
        return jsonify(users=query(f'SELECT user_id FROM read_parquet({literal(path.as_posix())}) WHERE starts_with(user_id,?) ORDER BY user_id LIMIT 40',[request.args.get('q','')]))

    def history_data():
        dataset=selected(); user=request.args.get('user','').strip()
        if not user: raise ValueError('Choose a listener first')
        matrix=source('matrix',dataset)
        exists=query(f'SELECT user_id FROM read_parquet({literal((matrix/"users.parquet").as_posix())}) WHERE user_id=?',[user])
        if not exists: raise ValueError('Listener not found in this training snapshot')
        source_path=source('ingest',dataset)
        if dataset=='1k':
            recent=int(request.args.get('days',90))
            if not 1<=recent<=3650: raise ValueError('Use a window between 1 and 3650 days')
            events=literal((source_path/'events/**/*.parquet').as_posix())
            rows=query(f'''SELECT item_id,count(*) AS play_count,
                count(*) FILTER(WHERE played_at < ?) AS historical_plays,
                count(*) FILTER(WHERE played_at >= ?) AS active_plays,
                min(played_at) AS first_played_at,max(played_at) AS last_played_at
                FROM read_parquet({events},hive_partitioning=true)
                WHERE user_id=? AND played_at < ? GROUP BY item_id''',
                [CUTOFF-timedelta(days=recent),CUTOFF-timedelta(days=recent),user,CUTOFF])
            meta=source_path/'track_metadata.parquet'
        else:
            rows=query(f'SELECT item_id,play_count FROM read_parquet({literal((matrix/"edges.parquet").as_posix())}) WHERE user_id=?',[user])
            meta=source_path/'artist_metadata.parquet'
        # Restrict metadata lookup to this listener's items and choose aliases deterministically.
        with duckdb.connect() as db:
            db.execute('CREATE TEMP TABLE chosen(item_id VARCHAR)')
            if rows: db.executemany('INSERT INTO chosen VALUES (?)',[(r['item_id'],) for r in rows])
            track='track_name' if dataset=='1k' else 'NULL AS track_name'
            metadata=db.execute(f'''SELECT item_id,artist_name,{track} FROM read_parquet({literal(meta.as_posix())})
                JOIN chosen USING(item_id) ORDER BY item_id,artist_name''').fetchall()
        lookup={r[0]:r[1:] for r in reversed(metadata)}
        for row in rows:
            row['artist'],row['track']=lookup.get(row['item_id'],(None,None))
        return rows

    @app.get('/api/history')
    def history():
        rows=history_data()
        return jsonify(total_items=len(rows),total_plays=sum(r['play_count'] for r in rows),
                       rows=sorted(rows,key=lambda r:(-r['play_count'],r['item_id']))[:100])

    @app.get('/api/dormant')
    def dormant():
        if selected()!='1k': raise ValueError('360K has no listening timestamps for dormancy')
        threshold=int(request.args.get('threshold',5))
        if not 1<=threshold<=10000: raise ValueError('Minimum plays must be between 1 and 10000')
        rows=[r for r in history_data() if r['historical_plays']>=threshold and r['active_plays']==0]
        for row in rows:
            row['days_dormant']=round((CUTOFF-datetime.fromisoformat(row['last_played_at'])).total_seconds()/86400,1)
        return jsonify(total=len(rows),rows=sorted(rows,key=lambda r:(-r['historical_plays'],r['item_id']))[:100])

    def run(job_id, options):
        folder=root/'tmp/reprise-jobs'/job_id; folder.mkdir(parents=True)
        (folder/'request.json').write_text(json.dumps(options))
        try:
            with (folder/'run.log').open('w',encoding='utf-8') as log:
                process=subprocess.Popen([sys.executable,'-u','-m','lastfm.web_worker',str(root),str(folder)],
                    cwd=root,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                code=process.wait()
            with lock: jobs[job_id]['status']='completed' if code==0 else 'failed'
        except Exception as exc:
            with lock: jobs[job_id].update(status='failed',error=str(exc))

    @app.post('/api/jobs')
    def start():
        body=request.get_json(silent=True)
        if not isinstance(body,dict): raise ValueError('Expected job settings')
        user=body.get('user','')
        if not isinstance(user,str) or not user.strip(): raise ValueError('Choose a listener')
        options={'user':user.strip()}
        for name,default,low,high in [('k',10,1,50),('recent_days',90,1,3650),('min_historical_plays',5,1,10000),('half_life_days',90,1,3650)]:
            value=body.get(name,default)
            if type(value)!=int or not low<=value<=high: raise ValueError(f'Invalid {name}')
            options[name]=value
        alpha=body.get('alpha')
        if alpha is not None and (type(alpha) not in (int,float) or not math.isfinite(alpha) or not 0<=alpha<=1): raise ValueError('Alpha must be between zero and one')
        options['alpha']=alpha
        matrix=source('matrix','1k')
        if not query(f'SELECT user_id FROM read_parquet({literal((matrix/"users.parquet").as_posix())}) WHERE user_id=?',[user]): raise ValueError('Unknown listener')
        with lock:
            if any(j['status']=='running' for j in jobs.values()): return jsonify(error='A run is already in progress'),409
            job_id=uuid.uuid4().hex[:12]
            jobs[job_id]=dict(id=job_id,user=user,status='running',created=datetime.now(timezone.utc).isoformat(),settings=options)
        threading.Thread(target=run,args=(job_id,options),daemon=True).start()
        return jsonify(id=job_id),202

    @app.get('/api/jobs')
    def list_jobs():
        with lock: records=[dict(j) for j in jobs.values()][::-1]
        for job in records:
            folder=root/'tmp/reprise-jobs'/job['id']; log=folder/'run.log'
            job['log']=log.read_text(encoding='utf-8',errors='replace')[-8000:] if log.exists() else 'Starting worker…'
            job['result_available']=(folder/'result.json').exists() and job['status']=='completed'
        return jsonify(jobs=records)

    @app.get('/api/jobs/<job_id>/result')
    def result(job_id):
        with lock: job=jobs.get(job_id)
        if not job or job['status']!='completed': return jsonify(error='Result is not ready'),404
        return jsonify(json.loads((root/'tmp/reprise-jobs'/job_id/'result.json').read_text()))

    return app


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    create_app(args.root).run(host='127.0.0.1',port=args.port,debug=False)

if __name__=='__main__': main()
