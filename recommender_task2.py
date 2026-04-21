import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless
import matplotlib.pyplot as plt

from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate, KFold

#********************************* TASK 2 *********************************
# -------------------------------------------------------------------------
# (a)  Load data with line format: userID movieID rating timestamp
# -------------------------------------------------------------------------
def load_data(path):
    # ratings_small.csv has header: userId,movieId,rating,timestamp
    df = pd.read_csv(path)
    # Normalize column names in case they differ
    cols = {c.lower(): c for c in df.columns}
    u, m, r = cols['userid'], cols['movieid'], cols['rating']
    df = df[[u, m, r]].rename(columns={u: 'user', m: 'item', r: 'rating'})

    reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
    data   = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)
    print(f"Loaded {len(df):>7} ratings, "
          f"{df['user'].nunique()} users, "
          f"{df['item'].nunique()} items, "
          f"scale {df['rating'].min()}–{df['rating'].max()}")
    return data


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def run_cv(algo, data, cv=5, label=""):
    """Run 5-fold CV, return (mean_rmse, mean_mae, all_rmse, all_mae)."""
    t0 = time.time()
    res = cross_validate(algo, data,
                         measures=['RMSE', 'MAE'],
                         cv=cv, verbose=False)
    rmse = res['test_rmse']
    mae  = res['test_mae']
    print(f"  {label:<28}  "
          f"RMSE = {rmse.mean():.4f} ± {rmse.std():.4f}   "
          f"MAE = {mae.mean():.4f} ± {mae.std():.4f}   "
          f"[{time.time()-t0:.1f}s]")
    return rmse.mean(), mae.mean(), rmse, mae


# -------------------------------------------------------------------------
# (c) + (d)  Compare PMF, User-CF, Item-CF under 5-fold CV
# -------------------------------------------------------------------------
def part_cd(data):
    print("\n" + "=" * 72)
    print("(c)/(d)  PMF  vs  User-based CF  vs  Item-based CF   [5-fold CV]")
    print("=" * 72)

    results = {}

    # PMF ≡ unbiased SVD with no regularisation bias term.
    # Surprise's SVD with biased=False matches standard PMF (Salakhutdinov & Mnih).
    pmf = SVD(biased=False, n_factors=50, n_epochs=20, random_state=42)
    results['PMF'] = run_cv(pmf, data, label="PMF (SVD biased=False)")

    user_cf = KNNBasic(sim_options={'name': 'MSD', 'user_based': True},
                       verbose=False)
    results['User-CF'] = run_cv(user_cf, data, label="User-based CF (MSD)")

    item_cf = KNNBasic(sim_options={'name': 'MSD', 'user_based': False},
                       verbose=False)
    results['Item-CF'] = run_cv(item_cf, data, label="Item-based CF (MSD)")

    # Summary
    print("\n  Summary of mean performance:")
    print(f"  {'Method':<12}{'mean RMSE':>12}{'mean MAE':>12}")
    for name in ['PMF', 'User-CF', 'Item-CF']:
        r, m, _, _ = results[name]
        print(f"  {name:<12}{r:>12.4f}{m:>12.4f}")

    best = min(results, key=lambda n: results[n][0])
    print(f"\n  Best (lowest RMSE): {best}")

    return results


# -------------------------------------------------------------------------
# (e)  Impact of similarity metric on User-CF vs Item-CF
# -------------------------------------------------------------------------
def part_e(data, out_png='part_e_similarity.png'):
    print("\n" + "=" * 72)
    print("(e)  Impact of similarity metric on User-CF and Item-CF")
    print("=" * 72)

    similarities = ['cosine', 'MSD', 'pearson']
    grid = {'User-CF': {}, 'Item-CF': {}}

    for sim in similarities:
        print(f"\n  -- similarity = {sim} --")
        for label, user_based in [('User-CF', True), ('Item-CF', False)]:
            algo = KNNBasic(sim_options={'name': sim, 'user_based': user_based},
                            verbose=False)
            rmse, mae, _, _ = run_cv(algo, data, label=f"{label} ({sim})")
            grid[label][sim] = (rmse, mae)

    # Plot: grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x   = np.arange(len(similarities))
    w   = 0.35

    for ax, metric, idx in [(axes[0], 'RMSE', 0), (axes[1], 'MAE', 1)]:
        user_vals = [grid['User-CF'][s][idx] for s in similarities]
        item_vals = [grid['Item-CF'][s][idx] for s in similarities]
        ax.bar(x - w/2, user_vals, w, label='User-CF', color='#4e79a7')
        ax.bar(x + w/2, item_vals, w, label='Item-CF', color='#f28e2b')
        ax.set_xticks(x)
        ax.set_xticklabels(similarities)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by similarity metric')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        # Zoom y-axis so the small differences between bars are visible.
        lo = min(user_vals + item_vals)
        hi = max(user_vals + item_vals)
        pad = (hi - lo) * 0.25
        ax.set_ylim(lo - pad, hi + pad)
        # annotate bars with values
        offset = (hi - lo) * 0.03
        for xi, v in zip(x - w/2, user_vals):
            ax.text(xi, v + offset, f'{v:.3f}', ha='center', fontsize=8)
        for xi, v in zip(x + w/2, item_vals):
            ax.text(xi, v + offset, f'{v:.3f}', ha='center', fontsize=8)

    fig.suptitle('(e) Similarity metric impact — User-CF vs Item-CF')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved plot -> {out_png}")

    # Consistency check
    best_user = min(similarities, key=lambda s: grid['User-CF'][s][0])
    best_item = min(similarities, key=lambda s: grid['Item-CF'][s][0])
    print(f"\n  Best similarity for User-CF (by RMSE): {best_user}")
    print(f"  Best similarity for Item-CF (by RMSE): {best_item}")
    consistent = (best_user == best_item)
    print(f"  Ranking consistent across User-CF and Item-CF? {consistent}")

    return grid


# -------------------------------------------------------------------------
# (f) + (g)  Impact of K (number of neighbors) and best K
# -------------------------------------------------------------------------
def part_fg(data, k_values=None, out_png='part_f_neighbors.png'):
    print("\n" + "=" * 72)
    print("(f)  Impact of number of neighbors K")
    print("=" * 72)

    if k_values is None:
        k_values = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]

    # Use MSD (Surprise default) for both.
    rmse_user, mae_user = [], []
    rmse_item, mae_item = [], []

    for K in k_values:
        print(f"\n  -- K = {K} --")
        for label, store_r, store_m, user_based in [
                ('User-CF', rmse_user, mae_user, True),
                ('Item-CF', rmse_item, mae_item, False)]:
            algo = KNNBasic(k=K,
                            sim_options={'name': 'MSD', 'user_based': user_based},
                            verbose=False)
            rmse, mae, _, _ = run_cv(algo, data, label=f"{label} (K={K})")
            store_r.append(rmse)
            store_m.append(mae)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(k_values, rmse_user, 'o-', label='User-CF', color='#4e79a7')
    axes[0].plot(k_values, rmse_item, 's-', label='Item-CF', color='#f28e2b')
    axes[0].set_xlabel('K (number of neighbors)')
    axes[0].set_ylabel('RMSE (5-fold CV mean)')
    axes[0].set_title('RMSE vs K')
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    rlo, rhi = min(rmse_user + rmse_item), max(rmse_user + rmse_item)
    rpad = (rhi - rlo) * 0.08
    axes[0].set_ylim(rlo - rpad, rhi + rpad)

    axes[1].plot(k_values, mae_user, 'o-', label='User-CF', color='#4e79a7')
    axes[1].plot(k_values, mae_item, 's-', label='Item-CF', color='#f28e2b')
    axes[1].set_xlabel('K (number of neighbors)')
    axes[1].set_ylabel('MAE (5-fold CV mean)')
    axes[1].set_title('MAE vs K')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    mlo, mhi = min(mae_user + mae_item), max(mae_user + mae_item)
    mpad = (mhi - mlo) * 0.08
    axes[1].set_ylim(mlo - mpad, mhi + mpad)

    # Mark best K on RMSE subplot
    bi_u = int(np.argmin(rmse_user)); bi_i = int(np.argmin(rmse_item))
    axes[0].axvline(k_values[bi_u], color='#4e79a7', ls='--', alpha=0.4)
    axes[0].axvline(k_values[bi_i], color='#f28e2b', ls='--', alpha=0.4)
    axes[0].annotate(f'best User-CF K={k_values[bi_u]}',
                     xy=(k_values[bi_u], rmse_user[bi_u]),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, color='#4e79a7')
    axes[0].annotate(f'best Item-CF K={k_values[bi_i]}',
                     xy=(k_values[bi_i], rmse_item[bi_i]),
                     xytext=(10, -15), textcoords='offset points',
                     fontsize=9, color='#f28e2b')

    fig.suptitle('(f) Neighbors impact')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved plot -> {out_png}")

    # (g) Best K
    best_k_user = k_values[bi_u]
    best_k_item = k_values[bi_i]
    print(f"\n  Best K for User-CF (min RMSE) : {best_k_user}  "
          f"(RMSE = {rmse_user[bi_u]:.4f})")
    print(f"  Best K for Item-CF (min RMSE) : {best_k_item}  "
          f"(RMSE = {rmse_item[bi_i]:.4f})")
    print(f"  Same best K? {best_k_user == best_k_item}")

    return {
        'k_values'  : k_values,
        'rmse_user' : rmse_user, 'mae_user' : mae_user,
        'rmse_item' : rmse_item, 'mae_item' : mae_item,
        'best_k_user' : best_k_user, 'best_k_item' : best_k_item,
    }


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ratings', default='archive/ratings_small.csv',
                   help='path to ratings_small.csv')
    args = p.parse_args()

    np.random.seed(42)
    data = load_data(args.ratings)

    results_cd = part_cd(data)
    results_e  = part_e(data)
    results_fg = part_fg(data)

    print("\n" + "=" * 72)
    print("All experiments complete.")
    print("Plots written to:  part_e_similarity.png,  part_f_neighbors.png")
    print("=" * 72)
