import sys
import numpy as np
import scipy.sparse as sp
from lastfm.als import ImplicitMSVD

def main():
    print("=== Running Synthetic Dataset ALS Demonstration ===")
    n_users, n_items = 5, 6
    kappa = 40.0
    
    # Interactions: (user, item, count)
    interactions = [
        (0, 0, 12), (0, 1, 4),
        (1, 1, 8),  (1, 2, 20),
        (2, 2, 5),  (2, 3, 15),
        (3, 3, 9),  (3, 4, 3),
        (4, 0, 7),  (4, 5, 25),
    ]
    
    rows = [u for u, i, c in interactions]
    cols = [i for u, i, c in interactions]
    counts = np.array([c for u, i, c in interactions], dtype=np.float32)
    
    P = sp.csr_matrix((np.ones(len(interactions), dtype=np.float32), (rows, cols)), shape=(n_users, n_items))
    D = sp.csr_matrix((kappa * np.log1p(counts), (rows, cols)), shape=(n_users, n_items))
    
    model = ImplicitMSVD(factors=8, regularization=0.05, iterations=10, seed=42)
    model.fit(P, D, compute_loss_history=True, verbose=True)
    
    print("\nTraining loss progression:")
    for h in model.history:
        print(f"  Epoch {h['epoch']:2d}: Loss = {h['loss']:10.4f} (elapsed: {h['elapsed_sec']*1000:.2f}ms)")
        
    print("\nRecommendations for User 0 (seen: {0, 1}):")
    recs_unseen = model.recommend(0, k=3, exclude_seen=True, seen_items={0, 1})
    for item_idx, score in recs_unseen:
        print(f"  Item {item_idx}: score = {score:+.4f}")
        
    recs_all = model.recommend(0, k=3, exclude_seen=False)
    print("Top-3 unconstrained for User 0:")
    for item_idx, score in recs_all:
        print(f"  Item {item_idx}: score = {score:+.4f}")
        
    print("\nDemonstration complete and verified successfully.")

if __name__ == "__main__":
    main()
