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
    else:
        with connect(root) as db:
            result = db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='recsys'").fetchall()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
