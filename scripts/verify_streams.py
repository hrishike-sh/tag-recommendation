"""Verify the prepared model and dormant candidates without claiming accuracy."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.sparse import load_npz

from lastfm.discovery import recommend
from lastfm.snapshots import dump, fingerprint, read_db

ROOT=Path(__file__).resolve().parents[1]


def main():
    checks={}
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition:
            raise AssertionError(name)
    report_dir=ROOT/'artifacts/reports'
    model=json.loads((report_dir/'model-1k.json').read_text())
    dormant=json.loads((report_dir/'dormancy-1k.json').read_text())
    model_dir=Path(model['path'])
    matrix_dir=Path(model['matrix_path'])
    check('model bound to immutable source matrix',fingerprint(matrix_dir)==model['matrix_fingerprint'])
    check('dormancy and model use same snapshot',dormant['matrix_fingerprint']==model['matrix_fingerprint'])
    counts=load_npz(model_dir/'counts.npz')
    check('training count shape',list(counts.shape)==model['shape'])
    check('training observed pairs',counts.nnz==model['observed_pairs'])
    for name,rows in [('user',counts.shape[0]),('item',counts.shape[1])]:
        factors=np.load(model_dir/f'{name}_factors.npy',mmap_mode='r',allow_pickle=False)
        check(f'{name} factor shape',factors.shape==(rows,model['factors']))
        check(f'{name} factors finite',np.isfinite(factors).all())
    losses=[step['library_training_loss'] for step in model['training_history']]
    check('finite loss history',len(losses)==model['iterations'] and np.isfinite(losses).all())
    check('training loss decreased',losses[-1]<losses[0])
    with read_db() as db:
        samples=db.execute('SELECT user_id,user_index FROM read_parquet(?) ORDER BY user_index LIMIT 3',
                           [str(model_dir/'users.parquet')]).fetchall()
        for user_id,user_index in samples:
            result=recommend(ROOT,user_id,model_dir=model_dir,k=10)
            rows=result['recommendations']
            ids=[r['item_index'] for r in rows]
            seen=set(counts.getrow(user_index).indices)
            check(f'{user_id}: 10 distinct unseen items',len(ids)==10 and len(set(ids))==10 and not seen.intersection(ids))
            check(f'{user_id}: finite scores and no fabricated tags',all(np.isfinite(r['score']) and r['tag_score'] is None for r in rows))
        db.execute('CREATE TEMP TABLE candidates AS SELECT * FROM read_parquet(?)',
                   [str(Path(dormant['path'])/'candidates.parquet')])
        actual,bad=db.execute('''SELECT count(*), count(*) FILTER (WHERE historical_plays < ?
            OR active_plays != 0 OR days_since_last_play <= ? OR last_played_at >= ?::TIMESTAMPTZ)
            FROM candidates''',[dormant['min_historical_plays'],dormant['recent_days'],dormant['cutoff']]).fetchone()
        check('all dormant candidates satisfy criterion',actual==dormant['candidates'] and bad==0)
        expected=db.execute('''SELECT count(*) FROM read_parquet(?)
            WHERE historical_plays>=? AND active_plays=0''',
            [str(matrix_dir/'edges.parquet'),dormant['min_historical_plays']]).fetchone()[0]
        check('dormancy extraction complete',actual==expected)
        mismatches=db.execute('''SELECT count(*) FROM candidates c LEFT JOIN read_parquet(?) e
            USING(user_index,item_index) WHERE e.play_count IS NULL OR c.play_count != e.play_count
            OR c.last_played_at != e.last_played_at''',[str(matrix_dir/'edges.parquet')]).fetchone()[0]
        check('dormant records retain source counts and times',mismatches==0)
    commands={}
    for name,command in [('python_tests',[sys.executable,'-m','pytest','-q']),
                         ('go_tests',['go','test','./...']),('go_vet',['go','vet','./...'])]:
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        check(name,result.returncode==0)
        commands[name]={'exit_code':result.returncode,'output':result.stdout+result.stderr}
    report={'verified_at':datetime.now(timezone.utc).isoformat(),'checks':checks,'commands':commands,
            'model':{k:v for k,v in model.items() if k not in ('path','matrix_path')},
            'dormancy':{k:v for k,v in dormant.items() if k!='path'},
            'model_path':model_dir.relative_to(ROOT).as_posix(),
            'dormancy_path':Path(dormant['path']).relative_to(ROOT).as_posix(),
            'tag_status':'Implemented and tested with synthetic fixtures; real corpus not supplied',
            'quality_evaluation':'No held-out relevance metrics have been measured'}
    dump(report_dir/'streams-verification.json',report)
    dump(ROOT/'docs/streams-verified-state.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
