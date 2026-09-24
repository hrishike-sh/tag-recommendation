"""Execute Phase 4 hyperparameter validation protocol and final test run on ingested Last.fm dataset."""

from pathlib import Path
from lastfm.protocol import run_phase4_protocol


def main():
    root = Path.cwd()
    print("=== Running Phase 4 Protocol (Validation Search -> Test Evaluation) ===")
    
    results = run_phase4_protocol(
        root=root,
        dataset="1k",
        train_cutoff="2009-04-01T00:00:00Z",
        val_cutoff="2009-05-01T00:00:00Z",
        test_end="2009-07-01T00:00:00Z",
        factors_grid=[32, 64, 128],
        reg_grid=[0.01, 0.05, 0.1, 0.5],
        iterations_grid=[10, 15, 20],
        seed=42,
        kappa=40.0,
    )
    
    print("\nPhase 4 Protocol Finished Successfully!")
    print(f"Selected Configuration: {results['selected_config']}")
    print(f"Popularity Test Metrics: {results['popularity_metrics']}")
    print(f"ImplicitMSVD Test Metrics: {results['als_metrics']}")


if __name__ == "__main__":
    main()
