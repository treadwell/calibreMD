#!/usr/bin/env python3
"""
XML-ridge style one-vs-all classifier using pooled chunk embeddings from calibregpt.db.

Key features:
- Uses pooled book embeddings from chunk embeddings (model-specific when available).
- Solves ridge in closed form: W = (X^T X + lambda I)^(-1) X^T Y.
- Uses inverse propensity weights parameterized by A and B.
- Tunes (lambda, A, B) with pure-NumPy Gaussian-process Bayesian optimization.
- Evaluates on held-out split with propensity-scored precision@k.
- Prints top-10 predicted labels and ground-truth labels for random books.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def decode_embedding_blob(blob: bytes, expected_dim: int) -> np.ndarray | None:
    if blob is None:
        return None
    if len(blob) % 8 != 0:
        return None
    vec = np.frombuffer(blob, dtype="<f8")
    if expected_dim > 0 and vec.size != expected_dim:
        return None
    if not np.all(np.isfinite(vec)):
        return None
    return vec.astype(np.float32, copy=True)


@dataclass
class EmbeddingLoadResult:
    book_ids: np.ndarray
    X: np.ndarray
    titles: Dict[int, str]
    n_chunks_total: int
    n_chunks_model_specific: int
    n_chunks_fallback: int
    n_chunks_skipped: int


def load_pooled_book_embeddings(
    library_dir: str,
    model: str,
    embedding_dim: int,
) -> EmbeddingLoadResult:
    db_path = os.path.join(library_dir, "calibregpt.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"calibregpt.db not found at {db_path}")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    query = """
    SELECT
      b.id AS book_id,
      b.title AS title,
      bc.sequence AS sequence,
      ce.embedding AS emb_model,
      bc.embedding AS emb_fallback
    FROM books b
    JOIN book_chunks bc ON bc.id_book = b.id
    LEFT JOIN chunk_embeddings ce
      ON ce.id_chunk = bc.id
     AND ce.model = ?
    ORDER BY b.id, bc.sequence
    """

    cur = con.execute(query, (model,))

    book_ids: List[int] = []
    pooled: List[np.ndarray] = []
    titles: Dict[int, str] = {}

    n_chunks_total = 0
    n_chunks_model_specific = 0
    n_chunks_fallback = 0
    n_chunks_skipped = 0

    current_book = None
    current_sum = None
    current_count = 0

    def flush_current() -> None:
        nonlocal current_book, current_sum, current_count
        if current_book is None:
            return
        if current_count > 0:
            book_ids.append(int(current_book))
            pooled.append((current_sum / float(current_count)).astype(np.float32, copy=False))
        current_book = None
        current_sum = None
        current_count = 0

    for row in cur:
        n_chunks_total += 1
        book_id = int(row["book_id"])
        titles[book_id] = row["title"]

        if current_book is None:
            current_book = book_id
            current_sum = np.zeros(embedding_dim, dtype=np.float64)
            current_count = 0
        elif book_id != current_book:
            flush_current()
            current_book = book_id
            current_sum = np.zeros(embedding_dim, dtype=np.float64)
            current_count = 0

        blob = row["emb_model"]
        used_model_specific = blob is not None
        if blob is None:
            blob = row["emb_fallback"]

        vec = decode_embedding_blob(blob, embedding_dim)
        if vec is None:
            n_chunks_skipped += 1
            continue

        current_sum += vec
        current_count += 1
        if used_model_specific:
            n_chunks_model_specific += 1
        else:
            n_chunks_fallback += 1

    flush_current()
    con.close()

    if not pooled:
        raise RuntimeError("No usable pooled embeddings were found.")

    return EmbeddingLoadResult(
        book_ids=np.asarray(book_ids, dtype=np.int64),
        X=np.vstack(pooled).astype(np.float32, copy=False),
        titles=titles,
        n_chunks_total=n_chunks_total,
        n_chunks_model_specific=n_chunks_model_specific,
        n_chunks_fallback=n_chunks_fallback,
        n_chunks_skipped=n_chunks_skipped,
    )


@dataclass
class LabelData:
    label_names: List[str]
    book_true_labels: List[List[int]]
    label_pos_rows: List[np.ndarray]
    label_freq: np.ndarray


def load_labels_from_metadata(
    library_dir: str,
    ordered_book_ids: np.ndarray,
    min_label_freq: int,
) -> LabelData:
    md_path = os.path.join(library_dir, "metadata.db")
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"metadata.db not found at {md_path}")

    con = sqlite3.connect(md_path)
    con.row_factory = sqlite3.Row
    cur = con.execute(
        """
        SELECT l.book AS book_id, t.name AS tag_name
        FROM books_tags_link l
        JOIN tags t ON t.id = l.tag
        """
    )

    allowed = set(int(x) for x in ordered_book_ids.tolist())
    raw_book_to_tags: Dict[int, set[str]] = {}
    tag_freq: Dict[str, int] = {}
    for row in cur:
        book_id = int(row["book_id"])
        if book_id not in allowed:
            continue
        tag = str(row["tag_name"])
        tags = raw_book_to_tags.setdefault(book_id, set())
        if tag not in tags:
            tags.add(tag)
            tag_freq[tag] = tag_freq.get(tag, 0) + 1
    con.close()

    kept_tags = sorted([t for t, f in tag_freq.items() if f >= min_label_freq])
    if not kept_tags:
        raise RuntimeError("No labels remain after min_label_freq filtering.")

    label_to_col = {t: i for i, t in enumerate(kept_tags)}
    label_names = kept_tags

    book_true_labels: List[List[int]] = []
    label_pos_rows_acc: List[List[int]] = [[] for _ in range(len(label_names))]
    for row_idx, book_id in enumerate(ordered_book_ids.tolist()):
        tags = raw_book_to_tags.get(int(book_id), set())
        lbls = sorted(label_to_col[t] for t in tags if t in label_to_col)
        book_true_labels.append(lbls)
        for lbl in lbls:
            label_pos_rows_acc[lbl].append(row_idx)

    label_pos_rows = [np.asarray(v, dtype=np.int64) for v in label_pos_rows_acc]
    label_freq = np.asarray([len(v) for v in label_pos_rows_acc], dtype=np.float64)

    return LabelData(
        label_names=label_names,
        book_true_labels=book_true_labels,
        label_pos_rows=label_pos_rows,
        label_freq=label_freq,
    )


def inverse_propensity(
    label_freq: np.ndarray,
    n_train: int,
    prop_a: float,
    prop_b: float,
) -> np.ndarray:
    c = (math.log(max(2, n_train)) - 1.0) * ((prop_b + 1.0) ** prop_a)
    inv = 1.0 + c * np.power(label_freq + prop_b, -prop_a)
    return inv.astype(np.float64, copy=False)


def build_B_base(X: np.ndarray, label_pos_rows: Sequence[np.ndarray]) -> np.ndarray:
    d = X.shape[1]
    n_labels = len(label_pos_rows)
    B = np.zeros((d, n_labels), dtype=np.float64)
    for j, rows in enumerate(label_pos_rows):
        if rows.size == 0:
            continue
        B[:, j] = X[rows].sum(axis=0, dtype=np.float64)
    return B


def fit_xml_ridge_from_B(
    XtX: np.ndarray,
    B_base: np.ndarray,
    lam: float,
    inv_prop: np.ndarray,
    train_propensity_weighting: bool = True,
) -> np.ndarray:
    d = XtX.shape[0]
    A = XtX + lam * np.eye(d, dtype=np.float64)
    if train_propensity_weighting:
        # Mild scaling keeps very rare labels from dominating too aggressively.
        weights = np.sqrt(inv_prop)
    else:
        weights = np.ones_like(inv_prop)
    B = B_base * weights[np.newaxis, :]
    W = np.linalg.solve(A, B)
    return W


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(np.int64, copy=False)


def oracle_cutoff_for_sample(scores_row: np.ndarray, true_labels: Sequence[int]) -> Tuple[int, float]:
    true_set = set(true_labels)
    m = len(true_set)
    if m == 0:
        return 0, 1.0

    rank = np.argsort(-scores_row)
    tp = 0
    best_f1 = 0.0
    best_t = 0
    for t_i, lbl in enumerate(rank.tolist(), start=1):
        if lbl in true_set:
            tp += 1
        f1_i = 2.0 * tp / float(m + t_i)
        if f1_i > best_f1:
            best_f1 = f1_i
            best_t = t_i
    return best_t, best_f1


def fit_topk_sigmoid_calibrator(
    scores: np.ndarray,
    book_true_labels: Sequence[Sequence[int]],
    k: int,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> Tuple[float, float]:
    # Fit p(y=1|s) = sigmoid(a*s + b) on top-k score-label pairs from held-out data.
    xs: List[float] = []
    ys: List[float] = []
    n = scores.shape[0]
    for i in range(n):
        idx = topk_indices(scores[i], k).tolist()
        true_set = set(book_true_labels[i])
        for j in idx:
            xs.append(float(scores[i, j]))
            ys.append(1.0 if j in true_set else 0.0)

    if not xs:
        return 1.0, 0.0

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)

    a = 1.0
    b = 0.0
    for _ in range(max_iter):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        w = p * (1.0 - p)
        # Gradient
        g_a = np.sum((p - y) * x)
        g_b = np.sum(p - y)
        # Hessian (2x2)
        h_aa = np.sum(w * x * x) + 1e-9
        h_ab = np.sum(w * x)
        h_bb = np.sum(w) + 1e-9

        det = h_aa * h_bb - h_ab * h_ab
        if det <= 1e-18:
            break

        # Newton step: H * delta = g
        d_a = (h_bb * g_a - h_ab * g_b) / det
        d_b = (-h_ab * g_a + h_aa * g_b) / det
        a_new = a - d_a
        b_new = b - d_b

        if abs(a_new - a) < tol and abs(b_new - b) < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new

    return float(a), float(b)


def gfm_cutoff_from_topk_probs(topk_probs_desc: np.ndarray) -> Tuple[int, float]:
    """
    Approximate GFM cutoff over top-k ranked labels using marginal probabilities.
    For t in [0..k], maximize estimated expected F1:
      E[F1_t] ≈ 2 * E[TP_t] / (E[|Y|] + t)
    where E[TP_t] = sum_{i<=t} p_i and E[|Y|] ≈ sum_{i<=k} p_i.
    """
    if topk_probs_desc.size == 0:
        return 0, 0.0
    p = np.clip(topk_probs_desc.astype(np.float64, copy=False), 0.0, 1.0)
    k = p.size
    m_hat = float(np.sum(p))
    pref = np.cumsum(p)
    best_t = 0
    best_val = 0.0
    for t in range(1, k + 1):
        val = 2.0 * float(pref[t - 1]) / float(m_hat + t) if (m_hat + t) > 0 else 0.0
        if val > best_val:
            best_val = val
            best_t = t
    return best_t, best_val


def evaluate_metrics(
    X: np.ndarray,
    book_true_labels: Sequence[Sequence[int]],
    W: np.ndarray,
    k: int,
    cal_params: Tuple[float, float] | None = None,
) -> Dict[str, float]:
    scores = X @ W
    f1_at_k_total = 0.0
    oracle_f1_total = 0.0
    gfm_f1_total = 0.0
    n = X.shape[0]

    for i in range(n):
        pred = topk_indices(scores[i], k)
        true = book_true_labels[i]
        true_set = set(true)
        pred_list = pred.tolist()
        hits = sum(1 for l in pred_list if l in true_set)

        # Macro F1 per sample at fixed k (top-k predictions).
        m = len(true_set)
        t = len(pred_list)
        if m == 0 and t == 0:
            f1_at_k = 1.0
        elif m == 0 or t == 0:
            f1_at_k = 0.0
        else:
            f1_at_k = 2.0 * hits / float(m + t)
        f1_at_k_total += f1_at_k

        # Oracle macro F1 per sample: best F1 over all score cutoffs for this sample.
        # Includes the empty-prediction option (t=0).
        scores_i = scores[i]
        rank = np.argsort(-scores_i)
        if m == 0:
            best_f1 = 1.0
        else:
            tp = 0
            best_f1 = 0.0
            for t_i, lbl in enumerate(rank.tolist(), start=1):
                if lbl in true_set:
                    tp += 1
                f1_i = 2.0 * tp / float(m + t_i)
                if f1_i > best_f1:
                    best_f1 = f1_i
        oracle_f1_total += best_f1

        # GFM-style cutoff from within top-k candidate list using calibrated probabilities.
        if cal_params is not None:
            cal_a, cal_b = cal_params
            pred_idx = pred.tolist()
            if pred_idx:
                pred_scores = scores[i, pred_idx]
                pred_probs = 1.0 / (1.0 + np.exp(-np.clip(cal_a * pred_scores + cal_b, -40.0, 40.0)))
                gfm_t, _ = gfm_cutoff_from_topk_probs(pred_probs)
                pred_gfm = pred_idx[:gfm_t]
            else:
                gfm_t = 0
                pred_gfm = []

            hits_g = sum(1 for l in pred_gfm if l in true_set)
            tg = len(pred_gfm)
            if m == 0 and tg == 0:
                f1_g = 1.0
            elif m == 0 or tg == 0:
                f1_g = 0.0
            else:
                f1_g = 2.0 * hits_g / float(m + tg)
            gfm_f1_total += f1_g

    out = {
        "macro_f1_per_sample_at_k": f1_at_k_total / float(n),
        "oracle_macro_f1_per_sample": oracle_f1_total / float(n),
    }
    if cal_params is not None:
        out["macro_f1_per_sample_gfm"] = gfm_f1_total / float(n)
    return out


def print_metric_curves(
    title: str,
    X: np.ndarray,
    book_true_labels: Sequence[Sequence[int]],
    W: np.ndarray,
    max_k: int,
) -> None:
    print(title)
    print("  k\tF1@k(macro-per-sample)")
    for kk in range(1, max_k + 1):
        m = evaluate_metrics(
            X=X,
            book_true_labels=book_true_labels,
            W=W,
            k=kk,
        )
        print(
            f"  {kk}\t"
            f"{m['macro_f1_per_sample_at_k']:.6f}"
        )


def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _Phi(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float, var: float) -> np.ndarray:
    X1_sq = np.sum(X1 * X1, axis=1, keepdims=True)
    X2_sq = np.sum(X2 * X2, axis=1, keepdims=True).T
    d2 = np.maximum(0.0, X1_sq + X2_sq - 2.0 * (X1 @ X2.T))
    return var * np.exp(-0.5 * d2 / (length_scale * length_scale))


def gp_predict(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    X_cand: np.ndarray,
    noise: float = 1e-6,
    length_scale: float = 0.35,
    var: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    K = rbf_kernel(X_obs, X_obs, length_scale=length_scale, var=var)
    K[np.diag_indices_from(K)] += noise
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))

    K_s = rbf_kernel(X_obs, X_cand, length_scale=length_scale, var=var)
    mu = K_s.T @ alpha

    v = np.linalg.solve(L, K_s)
    var_pred = np.maximum(var - np.sum(v * v, axis=0), 1e-12)
    return mu, var_pred


def expected_improvement(mu: np.ndarray, var: np.ndarray, best: float, xi: float = 1e-3) -> np.ndarray:
    sigma = np.sqrt(np.maximum(var, 1e-12))
    z = (mu - best - xi) / sigma
    ei = (mu - best - xi) * _Phi(z) + sigma * _phi(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def run_gp_bo(
    eval_fn,
    bounds: Sequence[Tuple[float, float]],
    n_init: int,
    n_iter: int,
    n_candidates: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    dim = len(bounds)

    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)

    def sample_uniform(n: int) -> np.ndarray:
        u = rng.random((n, dim))
        return lo + (hi - lo) * u

    X_obs = []
    y_obs = []

    for _ in range(n_init):
        x = sample_uniform(1)[0]
        y = eval_fn(x)
        X_obs.append(x)
        y_obs.append(y)

    while len(X_obs) < n_iter:
        Xo = np.asarray(X_obs, dtype=np.float64)
        yo = np.asarray(y_obs, dtype=np.float64)
        y_mean = yo.mean()
        y_std = yo.std() + 1e-12
        yo_norm = (yo - y_mean) / y_std

        X_cand = sample_uniform(n_candidates)
        try:
            mu, var = gp_predict(Xo, yo_norm, X_cand)
            ei = expected_improvement(mu, var, best=float(np.max(yo_norm)))
            x_next = X_cand[int(np.argmax(ei))]
        except np.linalg.LinAlgError:
            x_next = sample_uniform(1)[0]

        y_next = eval_fn(x_next)
        X_obs.append(x_next)
        y_obs.append(y_next)

    Xo = np.asarray(X_obs, dtype=np.float64)
    yo = np.asarray(y_obs, dtype=np.float64)
    best_idx = int(np.argmax(yo))
    return Xo, yo, float(yo[best_idx]), Xo[best_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="XML-ridge with pooled chunk embeddings (ada-002 focus).")
    parser.add_argument("--library-dir", required=True, help="Calibre library directory containing calibregpt.db + metadata.db")
    parser.add_argument("--model", default="text-embedding-ada-002", help="Embedding model name in chunk_embeddings")
    parser.add_argument("--embedding-dim", type=int, default=1536, help="Expected embedding dimension")
    parser.add_argument("--min-label-freq", type=int, default=5, help="Drop labels with fewer positives")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--k",
        type=int,
        default=32,
        help="Candidate top-k list; GFM selects cutoff <= k from this list",
    )
    parser.add_argument("--sample-size", type=int, default=10, help="How many random books to print")
    parser.add_argument("--no-train-propensity-weighting", action="store_true", help="Disable propensity weighting in training")
    parser.add_argument("--bo-init", type=int, default=8, help="Initial random BO trials")
    parser.add_argument("--bo-iters", type=int, default=20, help="Total BO trials")
    parser.add_argument("--bo-candidates", type=int, default=1500, help="Candidate points per BO iteration")
    parser.add_argument("--lambda-log10-min", type=float, default=-6.0)
    parser.add_argument("--lambda-log10-max", type=float, default=2.0)
    parser.add_argument("--prop-a-min", type=float, default=0.2)
    parser.add_argument("--prop-a-max", type=float, default=1.5)
    parser.add_argument("--prop-b-min", type=float, default=0.5)
    parser.add_argument("--prop-b-max", type=float, default=5.0)
    args = parser.parse_args()

    library_dir = expand_path(args.library_dir)
    np.random.seed(args.seed)
    random.seed(args.seed)

    emb = load_pooled_book_embeddings(
        library_dir=library_dir,
        model=args.model,
        embedding_dim=args.embedding_dim,
    )

    lbl = load_labels_from_metadata(
        library_dir=library_dir,
        ordered_book_ids=emb.book_ids,
        min_label_freq=args.min_label_freq,
    )

    n_books, d = emb.X.shape
    n_labels = len(lbl.label_names)
    print(f"Loaded books with pooled embeddings: {n_books}")
    print(f"Embedding dims: {d}")
    print(f"Labels kept (min freq {args.min_label_freq}): {n_labels}")
    print(
        "Chunk usage: total={total}, model_specific={ms}, fallback={fb}, skipped={sk}".format(
            total=emb.n_chunks_total,
            ms=emb.n_chunks_model_specific,
            fb=emb.n_chunks_fallback,
            sk=emb.n_chunks_skipped,
        )
    )

    # In-sample-only flow: BO and final reporting are all on full data.
    X_all = emb.X.astype(np.float64, copy=False)
    y_all = lbl.book_true_labels
    XtX_all = X_all.T @ X_all
    B_base_all = build_B_base(X_all, lbl.label_pos_rows)
    label_freq_all = lbl.label_freq

    def eval_candidate(x: np.ndarray) -> float:
        lam = 10.0 ** float(x[0])
        prop_a = float(x[1])
        prop_b = float(x[2])
        inv_prop = inverse_propensity(
            label_freq=label_freq_all,
            n_train=X_all.shape[0],
            prop_a=prop_a,
            prop_b=prop_b,
        )
        W = fit_xml_ridge_from_B(
            XtX=XtX_all,
            B_base=B_base_all,
            lam=lam,
            inv_prop=inv_prop,
            train_propensity_weighting=not args.no_train_propensity_weighting,
        )
        metrics = evaluate_metrics(
            X=X_all,
            book_true_labels=y_all,
            W=W,
            k=args.k,
        )
        return float(metrics["oracle_macro_f1_per_sample"])

    bounds = [
        (args.lambda_log10_min, args.lambda_log10_max),
        (args.prop_a_min, args.prop_a_max),
        (args.prop_b_min, args.prop_b_max),
    ]

    X_trials, y_trials, best_score, best_x = run_gp_bo(
        eval_fn=eval_candidate,
        bounds=bounds,
        n_init=args.bo_init,
        n_iter=args.bo_iters,
        n_candidates=args.bo_candidates,
        seed=args.seed,
    )

    best_lambda = 10.0 ** float(best_x[0])
    best_a = float(best_x[1])
    best_b = float(best_x[2])
    print("\nBest BO params:")
    print(f"  lambda={best_lambda:.6g} (log10={best_x[0]:.4f})")
    print(f"  propensity_a={best_a:.6g}")
    print(f"  propensity_b={best_b:.6g}")
    print(f"  best_in_sample_oracle_macro_f1={best_score:.6f}")

    # Fit best in-sample model and report in-sample metrics.
    inv_prop_all = inverse_propensity(
        label_freq=label_freq_all,
        n_train=X_all.shape[0],
        prop_a=best_a,
        prop_b=best_b,
    )
    W_all = fit_xml_ridge_from_B(
        XtX=XtX_all,
        B_base=B_base_all,
        lam=best_lambda,
        inv_prop=inv_prop_all,
        train_propensity_weighting=not args.no_train_propensity_weighting,
    )
    scores_all = X_all @ W_all
    cal_a_all, cal_b_all = fit_topk_sigmoid_calibrator(scores_all, y_all, k=args.k)
    all_metrics = evaluate_metrics(
        X=X_all,
        book_true_labels=y_all,
        W=W_all,
        k=args.k,
        cal_params=(cal_a_all, cal_b_all),
    )
    print(f"  gfm_calibrator: sigmoid(a*s+b), a={cal_a_all:.6f}, b={cal_b_all:.6f}")
    print(
        "In-sample metrics: "
        f"F1@{args.k} (macro-per-sample)={all_metrics['macro_f1_per_sample_at_k']:.6f}, "
        f"GFM Macro F1 (per-sample, cutoff<= {args.k})={all_metrics['macro_f1_per_sample_gfm']:.6f}, "
        f"Oracle F1={all_metrics['oracle_macro_f1_per_sample']:.6f}"
    )
    print_metric_curves(
        title=f"In-sample curves (k=1..{args.k})",
        X=X_all,
        book_true_labels=y_all,
        W=W_all,
        max_k=args.k,
    )
    rng = np.random.default_rng(args.seed + 7)
    sample_n = min(args.sample_size, X_all.shape[0])
    sampled_rows = rng.choice(X_all.shape[0], size=sample_n, replace=False)

    print(f"\nRandom sample predictions (top {args.k}):")
    for row in sampled_rows.tolist():
        book_id = int(emb.book_ids[row])
        title = emb.titles.get(book_id, "<unknown>")
        true_lbl_idx = lbl.book_true_labels[row]
        best_t, best_f1 = oracle_cutoff_for_sample(scores_all[row], true_lbl_idx)

        rank_full = np.argsort(-scores_all[row]).tolist()
        show_n = max(args.k, best_t)
        show_n = max(1, min(len(rank_full), show_n))
        shown_idx = rank_full[:show_n]
        shown_labels = [lbl.label_names[j] for j in shown_idx]

        # Approximate GFM cutoff using calibrated probabilities on top-k scores.
        topk_idx = rank_full[: args.k]
        topk_scores = scores_all[row, topk_idx]
        topk_probs = 1.0 / (1.0 + np.exp(-np.clip(cal_a_all * topk_scores + cal_b_all, -40.0, 40.0)))
        gfm_t, gfm_obj = gfm_cutoff_from_topk_probs(topk_probs)
        show_n = max(show_n, gfm_t)
        show_n = max(1, min(len(rank_full), show_n))
        shown_idx = rank_full[:show_n]
        shown_labels = [lbl.label_names[j] for j in shown_idx]

        marker_map: Dict[int, List[str]] = {}
        marker_map.setdefault(max(0, min(show_n, best_t)), []).append("oracle")
        marker_map.setdefault(max(0, min(show_n, gfm_t)), []).append("gfm")
        parts: List[str] = []
        for pos in range(show_n + 1):
            if pos in marker_map:
                tags = "|".join(marker_map[pos])
                parts.append(f"|{tags}|")
            if pos < show_n:
                parts.append(shown_labels[pos])
        pred_with_pipe = ", ".join(parts)

        true_labels = [lbl.label_names[j] for j in true_lbl_idx]
        if len(true_labels) > args.k:
            true_labels = true_labels[: args.k]
        print(f"\nBook {book_id}: {title}")
        print(
            f"  Pred Ranked (shown={show_n}, oracle_t={best_t}, oracle_f1={best_f1:.4f}, "
            f"gfm_t={gfm_t}, gfm_obj={gfm_obj:.4f}):",
            pred_with_pipe,
        )
        print("  True Labels:", ", ".join(true_labels) if true_labels else "<none>")

    # Print compact BO trail summary for reproducibility.
    print("\nBO trail (in_sample_oracle_macro_f1):")
    for i in range(len(y_trials)):
        lam_i = 10.0 ** float(X_trials[i, 0])
        a_i = float(X_trials[i, 1])
        b_i = float(X_trials[i, 2])
        print(
            f"  iter={i+1:02d} "
            f"lambda={lam_i:.4g} a={a_i:.4f} b={b_i:.4f} "
            f"in_sample_oracle_macro_f1={y_trials[i]:.6f}"
        )


if __name__ == "__main__":
    main()
