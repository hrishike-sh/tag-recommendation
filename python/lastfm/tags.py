"""TF-IDF tag artifacts and two explicitly named overlap measures."""
from collections import defaultdict
import json
import math
from pathlib import Path
import uuid

import numpy as np
from scipy.sparse import coo_matrix, diags, load_npz, save_npz

from .acquire import sha256
from .ingest import literal
from .snapshots import dump, fingerprint, read_db, resolve_matrix, utc


def tfidf(counts):
    counts = counts.astype(np.float64).tocsr(copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("Tag frequencies must be nonnegative and finite")
    totals = np.asarray(counts.sum(axis=1)).ravel()
    tf = diags(np.divide(1., totals, out=np.zeros_like(totals), where=totals > 0)) @ counts
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    # N is the whole training catalog, including untagged items; no smoothing.
    idf = np.zeros(len(df), dtype=np.float64)
    np.log(np.divide(counts.shape[0], df, out=np.ones(len(df)), where=df > 0), out=idf)
    relevance = (tf @ diags(idf)).tocsr()
    relevance.eliminate_zeros()
    masses = np.asarray(relevance.sum(axis=1)).ravel()
    normalized = (diags(np.divide(1., masses, out=np.zeros_like(masses), where=masses > 0)) @ relevance).tocsr()
    return relevance, normalized, idf


def weighted_jaccard(a: dict, b: dict) -> float:
    """Symmetric sum(min(weight))/sum(max(weight)), zero for empty/zero mass."""
    if any(not math.isfinite(v) or v < 0 for v in [*a.values(), *b.values()]):
        raise ValueError("Weights must be finite and nonnegative")
    union = a.keys() | b.keys()
    denominator = sum(max(a.get(t,0),b.get(t,0)) for t in union)
    return sum(min(a.get(t,0),b.get(t,0)) for t in union)/denominator if denominator else 0.


def proposal_overlap(candidate: dict, profile: dict) -> float:
    """Jaccard(present tags) * mean shared candidate relevance (asymmetric)."""
    if any(not math.isfinite(v) or v < 0 for v in [*candidate.values(), *profile.values()]):
        raise ValueError("Weights must be finite and nonnegative")
    common = candidate.keys() & profile.keys()
    union = candidate.keys() | profile.keys()
    if not common or not union:
        return 0.
    return len(common)/len(union) * sum(candidate[t] for t in common)/len(common)


def import_tags(root: Path, source: Path, dataset="1k", matrix_dir=None, metadata_as_of=None):
    """TSV: item_id, tag, count; optional assigned_at timestamp per row."""
    matrix_dir, matrix_meta = resolve_matrix(root,dataset,matrix_dir)
    cutoff = matrix_meta["exclusive_cutoff_utc"]
    if metadata_as_of:
        as_of = utc(metadata_as_of)
        if cutoff and as_of >= utc(cutoff):
            raise ValueError("metadata_as_of must be strictly before the training cutoff")
        metadata_as_of = as_of.isoformat()
    with read_db() as db:
        db.execute("CREATE TEMP TABLE raw AS SELECT * FROM read_csv(?,delim='\t',header=true,all_varchar=true,quote='',escape='')",[str(source.resolve())])
        columns = {row[0] for row in db.execute("DESCRIBE raw").fetchall()}
        if not {"item_id","tag","count"} <= columns:
            raise ValueError("Tag TSV requires header columns item_id, tag, count")
        if cutoff and "assigned_at" not in columns and not metadata_as_of:
            raise ValueError("Historical training requires assigned_at or an explicit metadata_as_of assertion")
        timestamp = "try_cast(assigned_at AS TIMESTAMPTZ)" if "assigned_at" in columns else "NULL::TIMESTAMPTZ"
        fallback = f"TIMESTAMPTZ {literal(metadata_as_of)}" if metadata_as_of else "NULL::TIMESTAMPTZ"
        bad_timestamp = ("assigned_at IS NOT NULL AND trim(assigned_at) != '' AND ("
                         + timestamp + " IS NULL OR NOT regexp_matches(trim(assigned_at),"
                         "'(Z|z|[+-][0-9]{2}:[0-9]{2})$'))") if 'assigned_at' in columns else 'false'
        db.execute(f"""CREATE TEMP TABLE normalized AS SELECT trim(item_id) AS item_id,
            lower(regexp_replace(trim(tag),'\\s+',' ','g')) AS tag,
            try_cast(count AS DOUBLE) AS frequency,
            coalesce({timestamp},{fallback}) AS available_at,
            {bad_timestamp} AS bad_timestamp
            FROM raw""")
        invalid = db.execute("""SELECT count(*) FROM normalized WHERE
            item_id IS NULL OR item_id='' OR tag IS NULL OR tag='' OR frequency IS NULL
            OR NOT isfinite(frequency) OR frequency<=0 OR bad_timestamp""").fetchone()[0]
        if invalid:
            raise ValueError(f"{invalid} invalid tag rows; fix the source instead of silently dropping them")
        if cutoff:
            missing = db.execute("SELECT count(*) FROM normalized WHERE available_at IS NULL").fetchone()[0]
            if missing:
                raise ValueError("Missing tag timestamps require metadata_as_of")
        total = db.execute("SELECT count(*) FROM normalized").fetchone()[0]
        condition = f"available_at < TIMESTAMPTZ {literal(cutoff)}" if cutoff else "true"
        future = db.execute(f"SELECT count(*) FROM normalized WHERE NOT ({condition})").fetchone()[0]
        db.execute(f"CREATE TEMP TABLE eligible AS SELECT * FROM normalized WHERE {condition}")
        db.execute("CREATE TEMP TABLE mapping AS SELECT * FROM read_parquet(?)",[str(matrix_dir/'items.parquet')])
        unknown = db.execute("SELECT count(*) FROM eligible e ANTI JOIN mapping m USING(item_id)").fetchone()[0]
        db.execute("""CREATE TEMP TABLE joined AS SELECT item_index,tag,sum(frequency) AS frequency
            FROM eligible JOIN mapping USING(item_id) GROUP BY item_index,tag""")
        db.execute("CREATE TEMP TABLE vocabulary AS SELECT tag,row_number() OVER (ORDER BY tag)-1 AS tag_index FROM (SELECT DISTINCT tag FROM joined)")
        vocabulary = [r[0] for r in db.execute("SELECT tag FROM vocabulary ORDER BY tag_index").fetchall()]
        if not vocabulary:
            raise ValueError("No tags match the training items before the cutoff")
        data = db.execute("SELECT item_index,tag_index,frequency FROM joined JOIN vocabulary USING(tag)").fetchnumpy()
    counts = coo_matrix((data['frequency'],(data['item_index'],data['tag_index'])),
                         shape=(matrix_meta['shape'][1],len(vocabulary))).tocsr()
    relevance, normalized, idf = tfidf(counts)
    out = root/'artifacts/tags'/dataset/uuid.uuid4().hex[:12]
    out.mkdir(parents=True)
    for name, value in (("counts",counts),("tfidf",relevance),("relevance",normalized)):
        save_npz(out/f'{name}.npz',value)
    np.save(out/'idf.npy',idf,allow_pickle=False)
    (out/'vocabulary.json').write_text(json.dumps(vocabulary,ensure_ascii=False),encoding='utf-8')
    covered = int(np.count_nonzero(np.diff(counts.indptr)))
    report = {"schema_version":1,"dataset":dataset,"matrix_fingerprint":fingerprint(matrix_dir),
              "source":str(source.resolve()),"source_sha256":sha256(source),
              "cutoff":cutoff,"metadata_as_of_assertion":metadata_as_of,
              "source_rows":total,"future_rows_excluded":future,"unknown_item_rows_excluded":unknown,
              "catalog_items":counts.shape[0],"tagged_items":covered,"coverage":covered/counts.shape[0],
              "vocabulary_size":len(vocabulary),"item_tag_pairs":counts.nnz,
              "tfidf":"(tag count / item tag count total) * ln(training catalog size / document frequency)",
              "normalization":"L1 per item for fusion; raw TF-IDF also retained",
              "path":str(out.resolve())}
    dump(out/'tags.json',report)
    dump(root/f'artifacts/reports/tags-{dataset}.json',report)
    return report


class TagIndex:
    def __init__(self, path: Path, matrix_fingerprint: str):
        self.metadata = json.loads((path/'tags.json').read_text())
        if self.metadata['matrix_fingerprint'] != matrix_fingerprint:
            raise ValueError("Tag artifact belongs to a different matrix/cutoff/mapping")
        self.counts=load_npz(path/'counts.npz').tocsr()
        self.relevance=load_npz(path/'relevance.npz').tocsr()
        self.vocabulary=json.loads((path/'vocabulary.json').read_text(encoding='utf-8'))

    def item(self, item_index):
        start,end=self.counts.indptr[item_index:item_index+2]
        tags={int(t):0. for t in self.counts.indices[start:end]}
        row=self.relevance.getrow(item_index)
        tags.update({int(t):float(w) for t,w in zip(row.indices,row.data)})
        return tags

    def profile(self, item_indices, plays):
        result=defaultdict(float)
        for item,weight in zip(item_indices,np.log1p(plays)):
            for tag,relevance in self.item(int(item)).items():
                result[tag] += float(weight)*relevance
        mass=sum(result.values())
        return {tag:weight/mass if mass else 0. for tag,weight in result.items()}
