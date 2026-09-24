"""Verify the prepared full dataset artifacts and record reproducible evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import duckdb
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]


def main():
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)
    db = duckdb.connect(str(ROOT / 'data/lastfm.duckdb'), read_only=True)
    db.execute("SET TimeZone='UTC'")
    db.execute("SET memory_limit='2GB'")
    db.execute("SET threads=4")
    for dataset in ('1k','360k'):
        report = json.loads((ROOT / f'artifacts/reports/ingest-{dataset}.json').read_text())
        matrix = json.loads((ROOT / f'artifacts/reports/matrix-{dataset}.json').read_text())
        merged = report.get('duplicates_removed',report.get('merged_user_artist_rows',0))
        check(f'{dataset}: ingestion conservation',
              report['raw_parsed_rows'] == report['validation_rejected_rows'] + report['clean_rows'] + merged)
        expected = 19150868 if dataset=='1k' else 17559530
        check(f'{dataset}: source line total',report['raw_parsed_rows']+report['parse_rejected_rows']==expected)
        count, bad, total = db.execute(f"""SELECT count(*),count(*) FILTER (
            WHERE play_count <= 0 OR preference != 1
            OR abs(confidence-(1+{matrix['kappa']}*ln(1+play_count))) > 1e-9
            OR abs(confidence_delta-(confidence-1)) > 1e-9),sum(play_count)
            FROM recsys.edges_{dataset}""").fetchone()
        check(f'{dataset}: matrix edges and confidence',count==matrix['observed_pairs'] and bad==0)
        for kind, size in zip(('user','item'),matrix['shape']):
            observed = db.execute(f"SELECT count(*),min({kind}_index),max({kind}_index) FROM recsys.{kind}s_{dataset}").fetchone()
            check(f'{dataset}: contiguous {kind} mapping',observed==(size,0,size-1))
        if dataset=='1k':
            cutoff=matrix['exclusive_cutoff_utc']
            expected_total=db.execute("SELECT count(*) FROM recsys.events_1k WHERE played_at < ?::TIMESTAMPTZ",[cutoff]).fetchone()[0]
            bad_temporal=db.execute("""SELECT count(*) FROM recsys.edges_1k WHERE
                active_plays+historical_plays != play_count OR last_played_at >= ?::TIMESTAMPTZ""",[cutoff]).fetchone()[0]
            check('1k: no future leakage and temporal count conservation',bad_temporal==0 and total==expected_total)
        else:
            expected_total=db.execute('SELECT sum(play_count) FROM recsys.artist_plays_360k').fetchone()[0]
            check('360k: aggregate play conservation',total==expected_total)
            bad_ids=db.execute("SELECT count(*) FROM recsys.artist_plays_360k WHERE NOT regexp_full_match(user_id,'[0-9a-f]{40}')").fetchone()[0]
            check('360k: only valid source user IDs',bad_ids==0)
        if matrix['npz_exported']:
            for name in ('counts','preferences','confidence_delta'):
                sparse=load_npz(Path(matrix['path'])/f'{name}.npz')
                check(f'{dataset}: {name} CSR shape',list(sparse.shape)==matrix['shape'])
                if name=='counts':
                    check(f'{dataset}: CSR count conservation',int(sparse.sum())==total and sparse.nnz==count)
                if name=='preferences':
                    check(f'{dataset}: CSR binary preferences',sparse.nnz==count and bool((sparse.data==1).all()))
    plan=db.execute('EXPLAIN SELECT count(*) FROM recsys.events_1k WHERE year=2009 AND month=4').fetchone()[1]
    check('1k: physical file pruning','File Filters' in plan and 'Scanning Files:' in plan)
    db.close()
    commands={}
    for name, command in {
        'python_tests':[sys.executable,'-m','pytest','-q'],
        'go_tests':['go','test','./...'],
        'go_vet':['go','vet','./...'],
        'go_smoke':['go','run','./cmd/recommender','-count','9','-kappa','40'],
    }.items():
        result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
        commands[name]={'exit_code':result.returncode,'output':result.stdout+result.stderr}
        check(name,result.returncode==0)
    go=json.loads(commands['go_smoke']['output'])
    check('Go/Python formula agreement',go['preference']==1 and abs(go['confidence']-93.10340371976183)<1e-10)
    output={'verified_at':datetime.now(timezone.utc).isoformat(),'checks':checks,
            'commands':commands,'partition_pruning_plan':plan}
    (ROOT/'artifacts/reports/verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    portable={'verified_at':output['verified_at'],'checks':checks,'datasets':{}}
    for dataset in ('1k','360k'):
        ingestion=json.loads((ROOT/f'artifacts/reports/ingest-{dataset}.json').read_text())
        matrix_report=json.loads((ROOT/f'artifacts/reports/matrix-{dataset}.json').read_text())
        manifest=json.loads((ROOT/f'data/raw/{dataset}.manifest.json').read_text())
        portable['datasets'][dataset]={
            'source_url':manifest['url'],'archive_bytes':manifest['archive_bytes'],
            'archive_sha256':manifest['archive_sha256'],
            'ingestion':{k:v for k,v in ingestion.items() if k not in ('source','snapshot')},
            'matrix':{k:v for k,v in matrix_report.items() if k not in ('source','path')},
            'snapshot_path':Path(ingestion['snapshot']).relative_to(ROOT).as_posix(),
            'matrix_path':Path(matrix_report['path']).relative_to(ROOT).as_posix()}
    (ROOT/'docs/verified-state.json').write_text(json.dumps(portable,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
