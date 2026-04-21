import numpy as np
import pandas as pd
import time

#********************************* TASK 1 *********************************

# =========================================================================
# 1.  DISTANCE FUNCTIONS   (each returns an (N, K) matrix)
# =========================================================================
def euclidean_distance(data, centroids):
    """Squared Euclidean distance.
       Using squared form is fine for argmin (monotonic)
       and corresponds directly to the standard Euclidean SSE."""
    data_sq = np.sum(data * data, axis=1, keepdims=True)        # (N, 1)
    cent_sq = np.sum(centroids * centroids, axis=1)             # (K,)
    cross   = data @ centroids.T                                # (N, K)
    return np.maximum(data_sq - 2.0 * cross + cent_sq, 0.0)


def cosine_distance(data, centroids):
    """1 - cosine_similarity."""
    data_norm = np.linalg.norm(data, axis=1, keepdims=True) + 1e-12
    cent_norm = np.linalg.norm(centroids, axis=1) + 1e-12
    sim = (data @ centroids.T) / (data_norm * cent_norm)
    return 1.0 - sim


def jaccard_distance(data, centroids):
    """1 - Generalized Jaccard similarity
       J(x, y) = sum_i min(x_i, y_i) / sum_i max(x_i, y_i)
       Chunked over data rows to keep memory reasonable."""
    N, D = data.shape
    K    = centroids.shape[0]
    out  = np.empty((N, K))
    chunk = 500
    for s in range(0, N, chunk):
        d  = data[s:s + chunk]                                  # (c, D)
        mn = np.minimum(d[:, None, :], centroids[None, :, :]).sum(axis=2)
        mx = np.maximum(d[:, None, :], centroids[None, :, :]).sum(axis=2)
        out[s:s + chunk] = 1.0 - np.where(mx > 0, mn / mx, 0.0)
    return out


# =========================================================================
# 2.  K-MEANS CORE  (stop criteria are toggleable for Q4)
# =========================================================================
def kmeans(data, K, distance_fn,
           max_iter=500,
           stop_no_change=True,
           stop_sse_increase=True,
           seed=42):
    """Generic K-means.  Centroids are always updated to the mean of the
       points assigned to each cluster (as in classical K-means).

       Returns a dict with centroids, assignments, final SSE, iteration
       count, SSE history, and wall-clock runtime."""

    rng = np.random.RandomState(seed)
    N   = data.shape[0]

    # Initialize centroids by randomly picking K distinct samples
    init_idx  = rng.choice(N, K, replace=False)
    centroids = data[init_idx].astype(np.float64).copy()

    sse_history    = []
    prev_centroids = centroids.copy()
    iterations     = 0
    t0 = time.time()

    for it in range(1, max_iter + 1):
        iterations = it

        # ----- assignment step -----
        dists       = distance_fn(data, centroids)
        assignments = np.argmin(dists, axis=1)
        min_d       = dists[np.arange(N), assignments]

        # SSE under the chosen metric.
        # For Euclidean, `dists` are already squared -> no extra square.
        if distance_fn is euclidean_distance:
            sse = float(np.sum(min_d))
        else:
            sse = float(np.sum(min_d ** 2))

        # ----- stop: SSE increased vs previous iteration -----
        if stop_sse_increase and sse_history and sse > sse_history[-1]:
            # revert to previous centroids and recompute final metrics
            centroids = prev_centroids
            dists       = distance_fn(data, centroids)
            assignments = np.argmin(dists, axis=1)
            min_d       = dists[np.arange(N), assignments]
            sse = float(np.sum(min_d)) if distance_fn is euclidean_distance \
                  else float(np.sum(min_d ** 2))
            break

        sse_history.append(sse)

        # ----- update step -----
        new_centroids = centroids.copy()
        for k in range(K):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = data[mask].mean(axis=0)
            # empty-cluster safeguard: keep old centroid

        # ----- stop: no change in centroid position -----
        if stop_no_change and np.allclose(centroids, new_centroids, atol=1e-8):
            centroids = new_centroids
            break

        prev_centroids = centroids
        centroids      = new_centroids

    runtime = time.time() - t0
    return {
        'centroids'   : centroids,
        'assignments' : assignments,
        'sse'         : sse,
        'iterations'  : iterations,
        'sse_history' : sse_history,
        'runtime'     : runtime,
    }


# =========================================================================
# 3.  MAJORITY-VOTE ACCURACY
# =========================================================================
def majority_vote_accuracy(assignments, labels, K):
    """For each cluster, label it with the majority class of the points
       assigned to it.  Return predictive accuracy."""
    correct = 0
    for k in range(K):
        mask = assignments == k
        if not mask.any():
            continue
        cluster_labels = labels[mask]
        majority       = np.bincount(cluster_labels).argmax()
        correct       += int((cluster_labels == majority).sum())
    return correct / len(labels)


# =========================================================================
# 4.  MAIN EXPERIMENT  (answers Q1–Q5)
# =========================================================================
def main(data_path, label_path):
    print("Loading data ...")
    data   = pd.read_csv(data_path,  header=None).values.astype(np.float64)
    labels = pd.read_csv(label_path, header=None).values.flatten().astype(int)
    K      = len(np.unique(labels))
    print(f"  data   = {data.shape}")
    print(f"  labels = {labels.shape}, K = {K}\n")

    metrics = [
        ('Euclidean', euclidean_distance),
        ('Cosine',    cosine_distance),
        ('Jaccard',   jaccard_distance),
    ]

    # -------------------------------------------------------------
    # Q1 + Q2 + Q3  :  combined stop criterion
    # (no centroid change  OR  SSE increases  OR  max_iter = 500)
    # -------------------------------------------------------------
    print("=" * 70)
    print("Q1 / Q2 / Q3 — combined stop criteria")
    print("=" * 70)

    q123 = {}
    for name, fn in metrics:
        r = kmeans(data, K, fn, max_iter=500,
                   stop_no_change=True, stop_sse_increase=True, seed=42)
        r['accuracy'] = majority_vote_accuracy(r['assignments'], labels, K)
        q123[name]    = r
        print(f"{name:<10}  SSE = {r['sse']:>14.4f}   "
              f"acc = {r['accuracy']*100:5.2f}%   "
              f"iters = {r['iterations']:>3}   "
              f"time = {r['runtime']:5.1f}s")

    # -------------------------------------------------------------
    # Q4 : each stop criterion in isolation
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Q4 — each stop condition individually")
    print("=" * 70)

    conditions = [
        ('(a) no-change only',
             dict(stop_no_change=True,  stop_sse_increase=False, max_iter=500)),
        ('(b) SSE-increase only',
             dict(stop_no_change=False, stop_sse_increase=True,  max_iter=500)),
        ('(c) max_iter=100 only',
             dict(stop_no_change=False, stop_sse_increase=False, max_iter=100)),
    ]

    q4 = {n: {} for n, _ in metrics}
    for cname, kw in conditions:
        print(f"\n-- {cname} --")
        for name, fn in metrics:
            r = kmeans(data, K, fn, seed=42, **kw)
            q4[name][cname] = r
            print(f"  {name:<10}  SSE = {r['sse']:>14.4f}   "
                  f"iters = {r['iterations']:>3}   "
                  f"time = {r['runtime']:5.1f}s")

    # -------------------------------------------------------------
    # Summary tables
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nQ1 — SSE comparison")
    for n in ['Euclidean', 'Cosine', 'Jaccard']:
        print(f"  {n:<10}  SSE = {q123[n]['sse']:.4f}")

    print("\nQ2 — Accuracy comparison")
    for n in ['Euclidean', 'Cosine', 'Jaccard']:
        print(f"  {n:<10}  Acc = {q123[n]['accuracy']*100:.2f}%")

    print("\nQ3 — Iterations / time to converge")
    print(f"  {'Method':<10} {'Iterations':>10} {'Runtime(s)':>12}")
    for n in ['Euclidean', 'Cosine', 'Jaccard']:
        print(f"  {n:<10} {q123[n]['iterations']:>10} {q123[n]['runtime']:>12.2f}")

    print("\nQ4 — SSE under each stop condition")
    header = f"  {'Method':<10}" + \
             "".join(f"{c:>26}" for c, _ in conditions)
    print(header)
    for n in ['Euclidean', 'Cosine', 'Jaccard']:
        row = f"  {n:<10}"
        for cname, _ in conditions:
            row += f"{q4[n][cname]['sse']:>26.4f}"
        print(row)

    return q123, q4


if __name__ == '__main__':
    main(data_path='kmeans_data/data.csv', label_path='kmeans_data/label.csv')
