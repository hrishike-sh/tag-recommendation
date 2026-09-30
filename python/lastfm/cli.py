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
    for command in ("acquire", "ingest", "matrix", "evaluate", "recommend"):
        p = sub.add_parser(command)
        if command in ("acquire", "ingest", "matrix"):
            p.add_argument("--dataset", choices=["1k", "360k"], required=True)
        else:
            p.add_argument("--dataset", choices=["1k"], default="1k")
        if command == "ingest":
            p.add_argument("--source", type=Path)
        if command == "matrix":
            p.add_argument("--cutoff")
            p.add_argument("--kappa", type=float, default=40)
            p.add_argument("--recent-days", type=int, default=90)
            p.add_argument("--npz", action="store_true")
        if command == "evaluate":
            p.add_argument("--model", choices=["als", "popularity"], default="als")
            p.add_argument("--cutoff", default="2009-05-01T00:00:00Z")
            p.add_argument("--factors", type=int, default=32)
            p.add_argument("--reg", type=float, default=0.05)
            p.add_argument("--epochs", type=int, default=15)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
            p.add_argument("--out", type=Path)
        if command == "recommend":
            p.add_argument("--mode", choices=["dual", "global"], default="dual")
            p.add_argument("--alpha", type=float, help="Discovery weight [0,1]; omitted = dynamic")
            p.add_argument("--recent-days", type=int, default=90)
            p.add_argument("--min-historical-plays", type=int, default=5)
            p.add_argument("--half-life-days", type=float, default=90)
            p.add_argument("--user", required=True, help="User ID (e.g. user_000001 or 0)")
            p.add_argument("--k", type=int, default=10, help="Number of items to recommend")
            p.add_argument("--weights", nargs=3, type=float, default=[0.20, 0.30, 0.50], help="Global mode only: weights for [MSVD, Tag, Temporal]")
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
    elif args.command == "evaluate":
        from .evaluate import run_experiment
        result = run_experiment(
            root,
            model_type=args.model,
            dataset=args.dataset,
            train_cutoff=args.cutoff,
            factors=args.factors,
            regularization=args.reg,
            iterations=args.epochs,
            seed=args.seed,
            k_list=args.k,
            output_dir=args.out,
        )
    elif args.command == "recommend":
        from .recommend_engine import get_recommendations_with_explanations
        result = get_recommendations_with_explanations(
            root,
            user_query=args.user,
            k=args.k,
            weights=args.weights,
            mode=args.mode, alpha=args.alpha, recent_days=args.recent_days,
            min_historical_plays=args.min_historical_plays, half_life_days=args.half_life_days,
        )
    else:
        with connect(root) as db:
            result = db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='recsys'").fetchall()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
