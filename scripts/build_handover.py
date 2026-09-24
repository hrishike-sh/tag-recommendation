"""Build the team handover from verified run reports (requires reportlab)."""
from datetime import datetime
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Preformatted

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'artifacts/reports'
I={d:json.loads((REPORTS/f'ingest-{d}.json').read_text()) for d in ('1k','360k')}
M={d:json.loads((REPORTS/f'matrix-{d}.json').read_text()) for d in ('1k','360k')}
V=json.loads((REPORTS/'verification.json').read_text())
assert all(V['checks'].values()), 'Do not publish a handover with failed verification'
A={d:json.loads((ROOT/f'data/raw/{d}.manifest.json').read_text()) for d in ('1k','360k')}
OUT=ROOT/'output/pdf/LastFM_Team_Handover.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
navy=colors.HexColor('#12304A'); teal=colors.HexColor('#087E8B'); gray=colors.HexColor('#526273')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleLocal',fontName='Helvetica-Bold',fontSize=27,leading=32,textColor=navy,spaceAfter=14))
styles.add(ParagraphStyle(name='HeadingLocal',fontName='Helvetica-Bold',fontSize=16,leading=20,textColor=navy,spaceAfter=12))
styles.add(ParagraphStyle(name='SubLocal',fontName='Helvetica-Bold',fontSize=11,leading=15,textColor=teal,spaceBefore=12,spaceAfter=6))
styles.add(ParagraphStyle(name='BodyLocal',fontName='Helvetica',fontSize=9.5,leading=14,textColor=navy,spaceAfter=8))
styles.add(ParagraphStyle(name='SmallLocal',fontName='Helvetica',fontSize=8,leading=11,textColor=gray,spaceAfter=5))
styles.add(ParagraphStyle(name='CellLocal',fontName='Helvetica',fontSize=8.5,leading=12,textColor=navy))
styles.add(ParagraphStyle(name='CodeLocal',fontName='Courier-Bold',fontSize=8.5,leading=11.5,textColor=navy,backColor=colors.HexColor('#F1F5F8'),borderPadding=8,spaceAfter=10))
story=[]
def p(text,style='BodyLocal'):
    story.append(Paragraph(text,styles[style]))
def h(text):p(text,'SubLocal')
def page(n,title):
    if story:story.append(PageBreak())
    p(f'ENGINEERING HANDOVER / {n:02d}','SmallLocal')
    p(title,'HeadingLocal')
def code(text):story.append(Preformatted(text,styles['CodeLocal']))
def table(rows,widths):
    cells=[[Paragraph(escape(str(c)),styles['CellLocal']) for c in row] for row in rows]
    t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DCECF0')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F4F7F9')]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),
        ('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),
        ('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),0.7,teal)]))
    story.append(t);story.append(Spacer(1,10))
def fmt(x):return f'{x:,}'

page(1,'Last.fm recommendation project')
p('Data foundation<br/>Team handover','TitleLocal')
p('Prepared 15 September 2026 | Design Experience | Milestone: acquisition, ingestion and confidence matrices','SmallLocal')
p('<b>Status:</b> Both source datasets have been acquired, ingested and converted to sparse preference artifacts. The Python pipeline and Go scaffold have passed automated checks. This is a working data foundation; recommendation training and ranking remain next-stage work.')
table([['Delivered','What teammates can use now'],
 ['Source acquisition','Original HTTPS archives, extracted TSVs, source READMEs and SHA-256 manifests.'],
 ['DuckDB + Parquet','A persistent catalog over cleaned snapshots; 1K events partitioned by UTC year/month.'],
 ['Training artifacts','User/item mappings, sparse edge tables and CSR NPZ files for both datasets.'],
 ['Go + Python repo','Installable Python CLI, locked dependencies, Go confidence CLI and tests.'],
 ['Handover evidence','Data contract, operations guide, verification report and this PDF.']],[119,380])
h('Why the project exists')
p('The larger proposal extends a movie recommender to music. A discovery stream will suggest unfamiliar tracks; a nostalgia stream will resurface previously favored tracks that have gone dormant but match current listening. This milestone supplies the historical data and implicit-feedback representation needed by both.')
h('Current scale')
table([['Metric','1K: tracks','360K: artists'],
 ['Clean records',fmt(I['1k']['clean_rows']),fmt(I['360k']['clean_rows'])],
 ['Users',fmt(I['1k']['users']),fmt(I['360k']['users'])],
 ['Canonical items',fmt(I['1k']['items']),fmt(I['360k']['items'])],
 ['Time partitioning',str(I['1k']['partitions'])+' UTC months','Not available in source']],[159,170,170])
p('Repository: C:/Users/rushi/Desktop/s5/Projects/Design Experience<br/>Start with README.md, then docs/data-contract.md and docs/operations.md.','SmallLocal')

page(2,'Data sources and ingestion architecture')
table([['Source','Semantics','Important boundary'],
 ['Last.fm 1K','User, timestamp, artist ID/name, track ID/name.','Individual track plays; supports chronology.'],
 ['Last.fm 360K','User SHA-1 ID, artist ID/name, total plays.','Artist totals; no listening timestamps.']],[103,201,195])
p('360K signup dates belong to user profiles and are not listening dates. The two datasets remain separate because their item units and user identifiers differ. Original profiles are retained in raw storage, but demographic tables are not part of the training pipeline. Tags and audio features are not supplied by these archives.')
h('Physical flow')
code('MTG HTTPS archives -> data/raw/ + provenance manifests\n  1K TSV   -> validate + deduplicate -> year=/month= Parquet\n  360K TSV -> validate + aggregate  -> artist-play Parquet\n                         |\n               data/lastfm.duckdb\n               recsys.* catalog views\n                         |\n          cutoff/counts -> mappings -> sparse matrices')
h('Canonical tables')
table([['View','Columns / role'],
 ['recsys.events_1k','user_id, played_at (TIMESTAMPTZ), artist_id, item_id, year, month'],
 ['recsys.artist_plays_360k','user_id, item_id, play_count (BIGINT)'],
 ['recsys.edges_1k / edges_360k','Indexed observed pairs, counts, preferences and confidence weights.'],
 ['recsys.users_* / items_*','Zero-based index to source/canonical ID mappings.']],[180,319])
h('Identity and publication policy')
p('IDs prefer MusicBrainz IDs. Missing artist IDs use a hash of the normalized name; missing track IDs use a hash of the artist identity and track name. Source names and IDs remain in metadata Parquet files. These fallbacks do not solve aliases or merge all name-only records with MBID records.')
p('Each ingestion creates a new snapshot directory. After successful writes, the catalog switches to it. Reruns replace the active view and do not append duplicate data. Old and failed snapshots are retained for audit. Matrix mappings and edge views are published together in a transaction.')
p('Partitioning is implemented as Hive-style Parquet directories queried by DuckDB, not native partitioned DuckDB tables. Include year/month filters plus exact played_at bounds for effective file pruning.','SmallLocal')

page(3,'Confidence-weighted preference matrix')
p('A listen is an implicit interest signal, not a star rating. Repetition increases confidence in a positive preference. A missing listen has low-confidence unknown preference; it is not an explicit dislike.')
code('n_ui = observed play count for user u and item i\nP_ui = 1[n_ui > 0]\nC_ui = 1 + kappa * ln(1 + n_ui)\nD_ui = C_ui - 1\n\nMissing pair: P_ui = 0, C_ui = 1, D_ui = 0')
table([['Plays','Preference','Confidence (kappa=40)'],['0','0','1.000000'],['1','1','28.725887'],['9','1','93.103404']],[120,130,249])
p('The natural logarithm limits the effect of very high play counts. Kappa defaults to 40 and is configurable; it has not been tuned. Kappa is separate from the future discovery/nostalgia mixture parameter alpha.')
h('Temporal contract for 1K')
p('Prepared training cutoff: <b>2009-05-01 00:00:00 UTC, exclusive</b>. Only earlier events contribute to counts, indices and last-play times. Active plays are in [cutoff - 90 days, cutoff); historical plays are earlier. Active + historical must equal the total. One ingested user has no pre-cutoff history, so the prepared training matrix has fewer users than the full lake.')
table([['Prepared matrix','Users x items','Observed pairs'],
 ['1K / track',f"{fmt(M['1k']['shape'][0])} x {fmt(M['1k']['shape'][1])}",fmt(M['1k']['observed_pairs'])],
 ['360K / artist',f"{fmt(M['360k']['shape'][0])} x {fmt(M['360k']['shape'][1])}",fmt(M['360k']['observed_pairs'])]],[109,260,130])
h('Sparse representation and solver boundary')
p('Each snapshot contains edges.parquet, users.parquet, items.parquet and matrix.json. CSR exports are counts.npz, preferences.npz and confidence_delta.npz. Missing sparse entries are zero, so actual confidence is conceptually an all-ones baseline plus the sparse delta. Do not allocate the dense confidence matrix.')
code('Objective for a future implicit factorizer:\nsum_ui C_ui * (P_ui - dot(X_u, Y_i))^2\n  + lambda_reg * (sum_u ||X_u||^2 + sum_i ||Y_i||^2)')
p('The objective includes missing pairs with confidence 1. Training only stored positive edges, or taking plain SVD of C*P, is different. Confirm the chosen solver\'s sparse-confidence convention. No factorizer or MSVD-specific regularizers are implemented yet.','SmallLocal')

page(4,'Runbook for teammates')
p('Run commands from the repository root. Install Python 3.12+, uv, curl and Go 1.24+. The prepared workspace has a Python 3.12 virtual environment. Go currently uses only its standard library.')
h('Install and verify the scaffold')
code('uv sync --frozen --extra dev\nuv run pytest -q\ngo test ./...\ngo vet ./...\ngo run ./cmd/recommender -count 9 -kappa 40')
h('Acquire and rebuild the data')
code('uv run lastfm acquire --dataset 1k\nuv run lastfm acquire --dataset 360k\nuv run lastfm ingest --dataset 1k\nuv run lastfm ingest --dataset 360k\nuv run lastfm matrix --dataset 1k \\\n  --cutoff 2009-05-01T00:00:00Z --npz\nuv run lastfm matrix --dataset 360k --npz\nuv run lastfm inspect\nuv run python scripts/verify.py')
p('The backslash above indicates a wrapped command; enter the 1K matrix command on one line in PowerShell. Alternatively use .venv/Scripts/python.exe -m lastfm.cli in this Windows workspace. The CLI also accepts --root before the subcommand.','SmallLocal')
h('Explore the prepared data')
code("import duckdb\ndb = duckdb.connect('data/lastfm.duckdb', read_only=True)\ndb.execute(\"SET TimeZone='UTC'\")\nprint(db.sql('SELECT * FROM recsys.edges_1k LIMIT 5'))")
h('Operational constraints')
p('Allow at least 15 GB free for downloads, extracted data, Parquet, exports and spill files. DuckDB uses a 2 GB memory limit and four threads; NumPy/SciPy exports require additional process memory. Run writers sequentially. This is not a multi-process serving database.')
p('The DuckDB catalog uses absolute paths to external Parquet. Rebuild views/data after moving the repo; copying the .duckdb file alone is insufficient. Archive and derived-data folders are ignored by Git. Teammates can reacquire the data using the CLI.')
p('Acquisition uses HTTPS with certificate verification, retries and resume. Existing manifests are checked against files before reuse. Archive scripts are not executed. SHA-256 values are local integrity fingerprints, not publisher-signed checksums.')

page(5,'Verification results and source anomalies')
p(f"<b>{len(V['checks'])} automated verification checks passed</b> on {datetime.fromisoformat(V['verified_at']).strftime('%d %B %Y at %H:%M UTC')}. Full evidence is stored in artifacts/reports/verification.json; the summary below refers to the prepared full-data run.")
table([['Check group','Evidence'],
 ['Source totals','All original interaction lines accounted for by parsed and quarantined records.'],
 ['Ingestion conservation','Accepted + validation rejects + merged/duplicate rows reconcile to parsed input.'],
 ['Preference artifacts','Positive counts, binary preferences, formula agreement and contiguous mappings.'],
 ['Temporal correctness','No post-cutoff last-play times; active + historical = total; counts match filtered events.'],
 ['Sparse export','CSR shapes and count sums match the corresponding DuckDB matrix tables.'],
 ['Partition pruning','EXPLAIN confirms year/month file filters and a reduced file scan.'],
 ['Code checks','Python integration/boundary tests, Go tests, go vet and cross-language formula smoke test.']],[138,361])
h('Cleaning outcomes')
table([['Outcome','1K','360K'],
 ['Original records',fmt(I['1k']['raw_parsed_rows']),fmt(I['360k']['raw_parsed_rows'])],
 ['Parser rejects',fmt(I['1k']['parse_rejected_rows']),fmt(I['360k']['parse_rejected_rows'])],
 ['Validation rejects',fmt(I['1k']['validation_rejected_rows']),fmt(I['360k']['validation_rejected_rows'])],
 ['Collapsed / merged rows',fmt(I['1k']['duplicates_removed']),fmt(I['360k']['merged_user_artist_rows'])],
 ['Clean records',fmt(I['1k']['clean_rows']),fmt(I['360k']['clean_rows'])]],[229,135,135])
p('1K contains two out-of-window timestamps and one exact canonical duplicate. Accepted events span '+escape(I['1k']['timestamp_min'])+' to '+escape(I['1k']['timestamp_max'])+'.')
p('360K contains 86 rows with date strings in the user-ID field, plus an item without a usable identity and a zero-count row. They are quarantined. Repeated canonical user-artist rows are summed; the report counts how many rows were merged.')
h('Interpretation limits')
p('These checks establish data integrity and pipeline behavior, not recommendation accuracy. There is no trained model, NDCG/Recall result, latency benchmark or demonstrated nostalgia improvement. The source chapter\'s roughly 83% personalization figure is not a result of this music implementation.')

page(6,'Ownership, next work and references')
table([['Workstream','Next concrete deliverable','Acceptance condition'],
 ['Data / tags','Measure tag coverage and audit identity aliases.','Report missing tags and fallback policy without inventing acoustic data.'],
 ['Modeling','Implement an implicit-feedback baseline.','Use the confidence contract and save factors with the exact ID mappings.'],
 ['Evaluation','Define chronological validation and test windows.','Report Recall/NDCG, popularity baseline and cold-start exclusions.'],
 ['Nostalgia','Retrieve dormant historical favorites.','Use pre-cutoff historical/active counts; compare with simple repeat baselines.'],
 ['Go service','Load versioned model artifacts and expose retrieval.','Model version, mappings and API contract agree with offline output.']],[89,193,217])
h('Decisions the team must make')
p('Agree on kappa tuning, validation/test dates, dormant-play thresholds, missing-tag behavior, and how discovery and nostalgia scores will be normalized and mixed. Do not assume a 0.7 mixture weight guarantees seven discovery items in a list of ten. The four-week proposal schedule is a plan, not a measured completion estimate for the remaining work.')
h('Files to read first')
p('<b>README.md</b> - commands and repository map.<br/><b>docs/data-contract.md</b> - schemas, identity, temporal and confidence semantics.<br/><b>docs/operations.md</b> - resource limits, reruns, portability and ownership.<br/><b>docs/verified-state.json</b> - portable result summary.<br/><b>artifacts/reports/</b> - full local run evidence.<br/><b>python/lastfm/</b> and <b>internal/preference/</b> - implementation.')
h('Source and technical references')
for title,url in [
 ('UPF / Last.fm 360K dataset description','https://www.upf.edu/web/mtg/lastfm360k'),
 ('Original 1K archive','https://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz'),
 ('Original 360K archive','https://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-360K.tar.gz'),
 ('DuckDB partitioned writes','https://duckdb.org/docs/stable/data/partitioning/partitioned_writes.html')]:
    p(f'<link href="{url}" color="#087E8B">{title}</link>','SmallLocal')
p('Read the README.txt retained with each dataset for provenance and the non-commercial-use condition. The proposal and source chapter supplied in the project root explain the broader research idea. This handover documents the code and data actually prepared in this milestone.','SmallLocal')
h('Handoff boundary')
p('The repository is initialized locally. No remote repository or deployment has been created. Raw data is excluded from Git and has not been published. Start the next work from a named matrix snapshot and retain its mappings, cutoff and kappa with all trained artifacts.')

def footer(canvas,doc):
    canvas.setStrokeColor(teal);canvas.setLineWidth(0.6);canvas.line(48,42,A4[0]-48,42)
    canvas.setFont('Helvetica',8);canvas.setFillColor(gray)
    canvas.drawString(48,29,'DESIGN EXPERIENCE  |  LAST.FM DATA FOUNDATION  |  15 SEP 2026')
    canvas.drawRightString(A4[0]-48,29,str(doc.page))

doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=56,
                      title='Last.fm Data Foundation - Team Handover',author='Design Experience')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
