"""Stream B phase 1: extract eligible dormant favorites without future data."""
from pathlib import Path
import uuid

from .ingest import literal
from .snapshots import dump, fingerprint, read_db, resolve_matrix


def extract_dormant(root: Path, dataset="1k", matrix_dir=None, min_historical_plays=5):
    if dataset != '1k':
        raise ValueError("Dormancy needs track timestamps; 360k cannot support it")
    if not isinstance(min_historical_plays,int) or isinstance(min_historical_plays,bool) or min_historical_plays < 1:
        raise ValueError("min_historical_plays must be a positive integer")
    matrix_dir,meta=resolve_matrix(root,dataset,matrix_dir)
    if not meta['exclusive_cutoff_utc'] or not meta['recent_days']:
        raise ValueError("Matrix must define a cutoff and recent window")
    out=root/'artifacts/dormancy'/dataset/uuid.uuid4().hex[:12]
    out.mkdir(parents=True)
    end=f"TIMESTAMPTZ {literal(meta['exclusive_cutoff_utc'])}"
    with read_db() as db:
        db.execute(f"""CREATE TEMP TABLE dormant AS SELECT *,
            epoch({end}-last_played_at)/86400.0 AS days_since_last_play,
            {end} AS as_of, {int(meta['recent_days'])} AS recent_days,
            {int(min_historical_plays)} AS min_historical_plays
            FROM read_parquet({literal(matrix_dir/'edges.parquet')})
            WHERE historical_plays >= {int(min_historical_plays)} AND active_plays=0
            AND last_played_at < {end} - INTERVAL '{int(meta['recent_days'])} days'""")
        db.execute(f"COPY dormant TO {literal(out/'candidates.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
        count,users=db.execute('SELECT count(*),count(DISTINCT user_id) FROM dormant').fetchone()
        minimum=db.execute('SELECT min(days_since_last_play) FROM dormant').fetchone()[0]
    report={"schema_version":1,"dataset":dataset,"matrix_fingerprint":fingerprint(matrix_dir),
            "cutoff":meta['exclusive_cutoff_utc'],"recent_days":meta['recent_days'],
            "min_historical_plays":min_historical_plays,"candidates":count,"users":users,
            "min_days_since_last_play":minimum,"path":str(out.resolve()),
            "criterion":"historical_plays >= threshold AND active_plays = 0",
            "scope":"Eligibility only; no contextual nostalgia score or Stream A/B blending yet"}
    dump(out/'dormancy.json',report)
    dump(root/f'artifacts/reports/dormancy-{dataset}.json',report)
    return report
