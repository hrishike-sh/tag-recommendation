import argparse
import json
from pathlib import Path
from .acquire import acquire, FILES
from .ingest import ingest, connect
from .matrix import matrix


def main():
    parser = argparse.ArgumentParser(description="Last.fm offline data pipeline")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("acquire", "ingest", "matrix"):
        p = sub.add_parser(command)
        p.add_argument("--dataset", choices=["1k", "360k"], required=True)
        if command == "ingest":
            p.add_argument("--source", type=Path)
        if command == "matrix":
            p.add_argument("--cutoff")
            p.add_argument("--kappa", type=float, default=40)
            p.add_argument("--recent-days", type=int, default=90)
            p.add_argument("--npz", action="store_true")
    sub.add_parser("inspect")
    p = sub.add_parser("train", help="Train the weighted implicit ALS core of Stream A")
    p.add_argument("--dataset", choices=["1k", "360k"], default="1k")
    p.add_argument("--matrix-dir", type=Path)
    p.add_argument("--factors", type=int, default=32)
    p.add_argument("--regularization", type=float, default=0.1)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("tags", help="Import canonical item-tag counts and compute TF-IDF")
    p.add_argument("--dataset", choices=["1k", "360k"], default="1k")
    p.add_argument("--matrix-dir", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--metadata-as-of")
    p = sub.add_parser("recommend", help="Retrieve unseen candidates and optionally apply fusion")
    p.add_argument("--dataset", choices=["1k", "360k"], default="1k")
    p.add_argument("--user-id", required=True)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--tags-dir", type=Path)
    p.add_argument("--mode", choices=["als", "fusion"])
    p.add_argument("--tag-metric", choices=["proposal", "weighted-jaccard"], default="proposal")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--candidates", type=int, default=200)
    p.add_argument("--favorites", type=int, default=50)
    p.add_argument("--content-weight", type=float, default=1.)
    p.add_argument("--tag-weight", type=float, default=1.)
    p = sub.add_parser("dormant", help="Extract Stream B dormant favorites from a temporal matrix")
    p.add_argument("--dataset", choices=["1k", "360k"], default="1k")
    p.add_argument("--matrix-dir", type=Path)
    p.add_argument("--min-historical-plays", type=int, default=5)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "acquire":
        result = acquire(root,args.dataset)
    elif args.command == "ingest":
        label = "1K" if args.dataset == "1k" else "360K"
        source = args.source or root / "data/raw" / f"lastfm-dataset-{label}" / FILES[args.dataset]
        result = ingest(root,source,args.dataset)
    elif args.command == "matrix":
        result = matrix(root,args.dataset,args.cutoff,args.kappa,args.recent_days,args.npz)
    elif args.command == "train":
        from .factorization import train
        result = train(root,args.dataset,args.matrix_dir,args.factors,args.regularization,
                       args.iterations,args.seed,args.threads)
    elif args.command == "tags":
        from .tags import import_tags
        result = import_tags(root,args.source,args.dataset,args.matrix_dir,args.metadata_as_of)
    elif args.command == "recommend":
        from .discovery import recommend
        result = recommend(root,args.user_id,args.dataset,args.model_dir,args.tags_dir,args.k,
                           args.candidates,args.favorites,args.mode,args.tag_metric,
                           args.content_weight,args.tag_weight)
    elif args.command == "dormant":
        from .dormancy import extract_dormant
        result = extract_dormant(root,args.dataset,args.matrix_dir,args.min_historical_plays)
    else:
        with connect(root) as db:
            result = db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='recsys'").fetchall()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
