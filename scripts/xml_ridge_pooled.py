#!/usr/bin/env python3
"""
XMLC trainer with two feature modes:
- XML-ridge on pooled chunk embeddings from calibregpt.db
- ELM-ridge (random sigmoid hidden layer) on sparse binary unigram features from full-text-search.db

Key features:
- Selectable feature pipeline (`embeddings` or `elm`).
- Solves ridge in closed form: W = (X^T X + lambda I)^(-1) X^T Y.
- Uses inverse propensity weights parameterized by A and B.
- Tunes (lambda, A, B) with pure-NumPy Gaussian-process Bayesian optimization.
- Evaluates in-sample oracle and GFM F1 diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import re
import math
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


class Stopwatch:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.last = self.t0

    def stamp(self, label: str) -> None:
        now = time.perf_counter()
        total = now - self.t0
        delta = now - self.last
        self.last = now
        print(f"[timing] +{total:8.2f}s (Δ{delta:7.2f}s) {label}")


TOKEN_RE = re.compile(r"[a-z0-9]+")


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


@dataclass
class BowLoadResult:
    book_ids: np.ndarray
    rows: List[np.ndarray]
    vocab_size: int
    titles: Dict[int, str]
    n_rows: int
    n_books_with_text: int
    n_tokens: int
    vocab_min_df: int
    vocab_max_features: int | None


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


def load_sparse_binary_bow_from_fulltext(
    library_dir: str,
    vocab_min_df: int,
    vocab_max_features: int | None,
    min_token_len: int = 2,
) -> BowLoadResult:
    db_path = os.path.join(library_dir, "full-text-search.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"full-text-search.db not found at {db_path}")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.execute(
        """
        SELECT book AS book_id, searchable_text
        FROM books_text
        WHERE length(searchable_text) > 0
        ORDER BY book, id
        """
    )

    # Pass 1: document frequency for vocabulary construction.
    df_counter: Counter[str] = Counter()
    n_rows = 0
    n_tokens = 0

    for row in cur:
        n_rows += 1
        text = str(row["searchable_text"] or "")
        toks = [t for t in TOKEN_RE.findall(text.lower()) if len(t) >= min_token_len]
        n_tokens += len(toks)
        if toks:
            df_counter.update(set(toks))

    # Build bounded vocabulary by descending DF then lexicographic token for stability.
    items = [(tok, df) for tok, df in df_counter.items() if df >= vocab_min_df]
    items.sort(key=lambda x: (-x[1], x[0]))
    if vocab_max_features is not None and vocab_max_features > 0:
        items = items[:vocab_max_features]
    vocab = {tok: i for i, (tok, _) in enumerate(items)}
    vocab_size = len(vocab)
    if vocab_size == 0:
        raise RuntimeError("Vocabulary is empty. Lower --vocab-min-df or increase available text.")

    # Pass 2: construct sparse binary rows per book.
    cur = con.execute(
        """
        SELECT book AS book_id, searchable_text
        FROM books_text
        WHERE length(searchable_text) > 0
        ORDER BY book, id
        """
    )
    book_ids: List[int] = []
    rows: List[np.ndarray] = []
    current_book = None
    current_set: set[int] | None = None

    def flush_current_binary() -> None:
        nonlocal current_book, current_set
        if current_book is None or current_set is None:
            return
        if current_set:
            idx = np.asarray(sorted(current_set), dtype=np.int32)
            book_ids.append(int(current_book))
            rows.append(idx)
        current_book = None
        current_set = None

    for row in cur:
        book_id = int(row["book_id"])
        text = str(row["searchable_text"] or "")
        if current_book is None:
            current_book = book_id
            current_set = set()
        elif book_id != current_book:
            flush_current_binary()
            current_book = book_id
            current_set = set()

        for tok in TOKEN_RE.findall(text.lower()):
            if len(tok) < min_token_len:
                continue
            j = vocab.get(tok)
            if j is not None:
                current_set.add(j)

    flush_current_binary()
    con.close()

    md_path = os.path.join(library_dir, "metadata.db")
    con_md = sqlite3.connect(md_path)
    cur_md = con_md.execute("SELECT id, title FROM books")
    titles = {int(r[0]): str(r[1]) for r in cur_md.fetchall()}
    con_md.close()

    if not rows:
        raise RuntimeError("No usable full-text rows found for BOW features.")

    return BowLoadResult(
        book_ids=np.asarray(book_ids, dtype=np.int64),
        rows=rows,
        vocab_size=vocab_size,
        titles=titles,
        n_rows=n_rows,
        n_books_with_text=len(book_ids),
        n_tokens=n_tokens,
        vocab_min_df=vocab_min_df,
        vocab_max_features=vocab_max_features,
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
    oracle_f1_total = 0.0
    gfm_f1_total = 0.0
    n = X.shape[0]

    for i in range(n):
        pred = topk_indices(scores[i], k)
        true = book_true_labels[i]
        true_set = set(true)
        pred_list = pred.tolist()
        hits = sum(1 for l in pred_list if l in true_set)

        m = len(true_set)

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
        "oracle_macro_f1_per_sample": oracle_f1_total / float(n),
    }
    if cal_params is not None:
        out["macro_f1_per_sample_gfm"] = gfm_f1_total / float(n)
    return out


def inv_norm_cdf(p: np.ndarray) -> np.ndarray:
    # Acklam's approximation, vectorized for NumPy arrays.
    p = np.asarray(p, dtype=np.float64)
    a = np.array([
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ])
    b = np.array([
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ])
    c = np.array([
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ])
    d = np.array([
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ])

    plow = 0.02425
    phigh = 1.0 - plow
    x = np.empty_like(p)

    lo = p < plow
    hi = p > phigh
    mid = (~lo) & (~hi)

    if np.any(lo):
        q = np.sqrt(-2.0 * np.log(p[lo]))
        x[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if np.any(mid):
        q = p[mid] - 0.5
        r = q * q
        x[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                 (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    if np.any(hi):
        q = np.sqrt(-2.0 * np.log(1.0 - p[hi]))
        x[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                  ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return x


def compute_bns_feature_weights(
    rows: Sequence[np.ndarray],
    label_pos_rows: Sequence[np.ndarray],
    vocab_size: int,
    eps: float = 1e-6,
) -> np.ndarray:
    n_books = len(rows)
    n_labels = len(label_pos_rows)

    df = np.zeros(vocab_size, dtype=np.int32)
    counts = np.zeros((vocab_size, n_labels), dtype=np.int32)

    for i, idx in enumerate(rows):
        if idx.size == 0:
            continue
        df[idx] += 1

    # Precompute row -> labels map once.
    row_to_labels: List[List[int]] = [[] for _ in range(n_books)]
    for l, pos_rows in enumerate(label_pos_rows):
        for r in pos_rows.tolist():
            row_to_labels[r].append(l)

    for i, idx in enumerate(rows):
        if idx.size == 0:
            continue
        for l in row_to_labels[i]:
            counts[idx, l] += 1

    max_bns = np.zeros(vocab_size, dtype=np.float64)
    for l, pos_rows in enumerate(label_pos_rows):
        n_pos = int(pos_rows.size)
        n_neg = n_books - n_pos
        if n_pos <= 0 or n_neg <= 0:
            continue
        tp = counts[:, l].astype(np.float64)
        fp = (df - counts[:, l]).astype(np.float64)
        tpr = (tp + eps) / (n_pos + 2.0 * eps)
        fpr = (fp + eps) / (n_neg + 2.0 * eps)
        tpr = np.clip(tpr, eps, 1.0 - eps)
        fpr = np.clip(fpr, eps, 1.0 - eps)
        bns = np.abs(inv_norm_cdf(tpr) - inv_norm_cdf(fpr))
        max_bns = np.maximum(max_bns, bns)

    # Keep scale numerically stable while preserving feature-relative BNS weighting.
    pos = max_bns[max_bns > 0]
    if pos.size > 0:
        med = float(np.median(pos))
        if med > 0:
            max_bns = max_bns / med
    max_bns = np.clip(max_bns, 0.0, 50.0)
    return max_bns


def build_elm_hidden_features(
    rows: Sequence[np.ndarray],
    input_dim: int,
    n_hidden: int,
    seed: int,
    feature_scale: np.ndarray | None = None,
) -> np.ndarray:
    n = len(rows)
    d = input_dim
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0 / math.sqrt(max(1, d)), size=(d, n_hidden)).astype(np.float64)
    if feature_scale is not None:
        if feature_scale.shape[0] != d:
            raise ValueError("feature_scale length must match input_dim")
        W *= feature_scale[:, None]
    b = rng.normal(0.0, 1.0, size=(n_hidden,)).astype(np.float64)
    H = np.empty((n, n_hidden), dtype=np.float64)
    for i, idx in enumerate(rows):
        if idx.size == 0:
            z = b.copy()
        else:
            # Sparse binary input: xW is sum of rows of W at active feature indices.
            z = b + W[idx.astype(np.int64, copy=False)].sum(axis=0)
        H[i] = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    return H


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
    trial_callback=None,
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

    for i in range(n_init):
        t_iter = time.perf_counter()
        x = sample_uniform(1)[0]
        y = eval_fn(x)
        X_obs.append(x)
        y_obs.append(y)
        if trial_callback is not None:
            trial_callback(
                iter_idx=i + 1,
                total_iters=n_iter,
                phase="init",
                x=x,
                y=y,
                elapsed_sec=time.perf_counter() - t_iter,
            )

    while len(X_obs) < n_iter:
        t_iter = time.perf_counter()
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
        if trial_callback is not None:
            trial_callback(
                iter_idx=len(X_obs),
                total_iters=n_iter,
                phase="bo",
                x=x_next,
                y=y_next,
                elapsed_sec=time.perf_counter() - t_iter,
            )

    Xo = np.asarray(X_obs, dtype=np.float64)
    yo = np.asarray(y_obs, dtype=np.float64)
    best_idx = int(np.argmax(yo))
    return Xo, yo, float(yo[best_idx]), Xo[best_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="XML-ridge / ELM-ridge for XMLC on Calibre data.")
    parser.add_argument("--library-dir", required=True, help="Calibre library directory containing calibregpt.db + metadata.db")
    parser.add_argument("--feature-mode", choices=["embeddings", "elm"], default="embeddings", help="Feature pipeline")
    parser.add_argument("--model", default="text-embedding-ada-002", help="Embedding model name in chunk_embeddings")
    parser.add_argument("--embedding-dim", type=int, default=1536, help="Expected embedding dimension")
    parser.add_argument("--vocab-min-df", type=int, default=5, help="Minimum document frequency for unigram vocabulary in ELM mode")
    parser.add_argument("--vocab-max-features", type=int, default=50000, help="Maximum unigram vocabulary size in ELM mode")
    parser.add_argument("--bow-min-token-len", type=int, default=2, help="Minimum token length for BOW mode")
    parser.add_argument("--elm-hidden-nodes", type=int, default=8192, help="Hidden units for random sigmoid layer in ELM mode")
    parser.add_argument("--elm-seed", type=int, default=None, help="Random seed for ELM hidden layer")
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
    sw = Stopwatch()

    library_dir = expand_path(args.library_dir)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.feature_mode == "embeddings":
        emb = load_pooled_book_embeddings(
            library_dir=library_dir,
            model=args.model,
            embedding_dim=args.embedding_dim,
        )
        book_ids = emb.book_ids
        titles = emb.titles
        X_raw = emb.X
        sw.stamp("Loaded and pooled chunk embeddings")
    else:
        vmax = None if args.vocab_max_features is not None and args.vocab_max_features <= 0 else args.vocab_max_features
        bow = load_sparse_binary_bow_from_fulltext(
            library_dir=library_dir,
            vocab_min_df=args.vocab_min_df,
            vocab_max_features=vmax,
            min_token_len=args.bow_min_token_len,
        )
        book_ids = bow.book_ids
        titles = bow.titles
        X_raw = None
        sw.stamp("Loaded sparse binary unigram features")

    lbl = load_labels_from_metadata(
        library_dir=library_dir,
        ordered_book_ids=book_ids,
        min_label_freq=args.min_label_freq,
    )
    sw.stamp("Loaded labels from metadata.db")

    if args.feature_mode == "elm":
        elm_seed = args.elm_seed if args.elm_seed is not None else args.seed
        bns_scale = compute_bns_feature_weights(
            rows=bow.rows,
            label_pos_rows=lbl.label_pos_rows,
            vocab_size=bow.vocab_size,
        )
        X_raw = build_elm_hidden_features(
            rows=bow.rows,
            input_dim=bow.vocab_size,
            n_hidden=args.elm_hidden_nodes,
            seed=elm_seed,
            feature_scale=bns_scale,
        )
        sw.stamp("Computed max-pooled BNS feature weights and built BNS-scaled ELM hidden features")

    n_books, d = X_raw.shape
    n_labels = len(lbl.label_names)
    print(f"Feature mode: {args.feature_mode}")
    print(f"Loaded books: {n_books}")
    print(f"Feature dims: {d}")
    print(f"Labels kept (min freq {args.min_label_freq}): {n_labels}")
    if args.feature_mode == "embeddings":
        print(
            "Chunk usage: total={total}, model_specific={ms}, fallback={fb}, skipped={sk}".format(
                total=emb.n_chunks_total,
                ms=emb.n_chunks_model_specific,
                fb=emb.n_chunks_fallback,
                sk=emb.n_chunks_skipped,
            )
        )
    else:
        bns_nonzero = int(np.sum(bns_scale > 0))
        print(
            "BOW usage: rows={rows}, books_with_text={books}, tokens={tokens}, vocab_size={vsz}, min_df={mdf}, max_features={mxf}, elm_hidden={hidden}, bns_nonzero={bnsnz}".format(
                rows=bow.n_rows,
                books=bow.n_books_with_text,
                tokens=bow.n_tokens,
                vsz=bow.vocab_size,
                mdf=bow.vocab_min_df,
                mxf=(bow.vocab_max_features if bow.vocab_max_features is not None else "None"),
                hidden=args.elm_hidden_nodes,
                bnsnz=bns_nonzero,
            )
        )

    # In-sample-only flow: BO and final reporting are all on full data.
    X_all = X_raw.astype(np.float64, copy=False)
    y_all = lbl.book_true_labels
    XtX_all = X_all.T @ X_all
    B_base_all = build_B_base(X_all, lbl.label_pos_rows)
    label_freq_all = lbl.label_freq
    sw.stamp("Prepared in-sample matrices (XtX, X^T Y base)")

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

    def on_bo_trial(iter_idx, total_iters, phase, x, y, elapsed_sec) -> None:
        lam = 10.0 ** float(x[0])
        a = float(x[1])
        b = float(x[2])
        print(
            f"[timing] BO {iter_idx:03d}/{total_iters:03d} "
            f"phase={phase} took {elapsed_sec:7.2f}s "
            f"lambda={lam:.4g} a={a:.4f} b={b:.4f} "
            f"oracle_f1={float(y):.6f}"
        )

    X_trials, y_trials, best_score, best_x = run_gp_bo(
        eval_fn=eval_candidate,
        bounds=bounds,
        n_init=args.bo_init,
        n_iter=args.bo_iters,
        n_candidates=args.bo_candidates,
        seed=args.seed,
        trial_callback=on_bo_trial,
    )
    sw.stamp("Completed GP-BO search")

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
    sw.stamp("Fit best model + calibrated GFM + computed in-sample metrics")
    print(f"  gfm_calibrator: sigmoid(a*s+b), a={cal_a_all:.6f}, b={cal_b_all:.6f}")
    print(
        "In-sample metrics: "
        f"GFM Macro F1 (per-sample, cutoff<= {args.k})={all_metrics['macro_f1_per_sample_gfm']:.6f}, "
        f"Oracle F1={all_metrics['oracle_macro_f1_per_sample']:.6f}"
    )
    rng = np.random.default_rng(args.seed + 7)
    sample_n = min(args.sample_size, X_all.shape[0])
    sampled_rows = rng.choice(X_all.shape[0], size=sample_n, replace=False)

    print(f"\nRandom sample predictions (top {args.k}):")
    for row in sampled_rows.tolist():
        book_id = int(book_ids[row])
        title = titles.get(book_id, "<unknown>")
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

    # Diagnostic: sample books with no retained ground-truth tags and show non-empty GFM recommendations.
    no_tag_rows = np.asarray([i for i, tags in enumerate(y_all) if len(tags) == 0], dtype=np.int64)
    if no_tag_rows.size > 0:
        diag_n = min(100, no_tag_rows.size)
        diag_rows = rng.choice(no_tag_rows, size=diag_n, replace=False)
        print(f"\nNo-tag diagnostic (sampled {diag_n} books with empty ground truth; showing non-empty GFM recommendations):")
        shown = 0
        for row in diag_rows.tolist():
            rank_full = np.argsort(-scores_all[row]).tolist()
            topk_idx = rank_full[: args.k]
            topk_scores = scores_all[row, topk_idx]
            topk_probs = 1.0 / (1.0 + np.exp(-np.clip(cal_a_all * topk_scores + cal_b_all, -40.0, 40.0)))
            gfm_t, gfm_obj = gfm_cutoff_from_topk_probs(topk_probs)
            if gfm_t <= 0:
                continue
            rec_idx = topk_idx[:gfm_t]
            rec_labels = [lbl.label_names[j] for j in rec_idx]
            book_id = int(book_ids[row])
            title = titles.get(book_id, "<unknown>")
            print(
                f"  Book {book_id} (gfm_t={gfm_t}, gfm_obj={gfm_obj:.4f}): "
                f"{title} -> {', '.join(rec_labels)}"
            )
            shown += 1
        if shown == 0:
            print("  None of the sampled no-tag books received non-empty GFM recommendations.")
    sw.stamp("Rendered sample predictions and no-tag diagnostic")

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
    sw.stamp("Printed BO trail and completed run")


if __name__ == "__main__":
    main()
