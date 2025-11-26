"""
src/task2/recommend_utils.py
Core content-based and hybrid recommendation utilities.

Design choices (teaching notes):
- Content-based: filter by cuisine flags and city; or compute cosine similarity on cuisine vectors.
- Hybrid: combine model predicted rating + normalized votes to compute final score.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def filter_by_cuisine_and_city(df: pd.DataFrame, cuisine_list: List[str]=None, city: Optional[str]=None):
    """
    Simple filter: if cuisine_list provided, we require at least one of the 'Cuisine_Grouped_<X>' flags.
    If cuisine flags not present, try 'Primary Cuisine' matching.
    """
    dfc = df.copy()
    if city:
        dfc = dfc[dfc["City"] == city] if "City" in dfc.columns else dfc
    if cuisine_list:
        # build mask based on one-hot columns if present
        flags = [f"Cuisine_Grouped_{c}" for c in cuisine_list]
        present_flags = [f for f in flags if f in dfc.columns]
        if present_flags:
            mask = dfc[present_flags].any(axis=1)
            dfc = dfc[mask]
        else:
            # fallback to Primary Cuisine string
            if "Primary Cuisine" in dfc.columns:
                mask = dfc["Primary Cuisine"].isin(cuisine_list)
                dfc = dfc[mask]
    return dfc

def build_cuisine_matrix(df: pd.DataFrame, cuisine_prefix="Cuisine_Grouped_"):
    """
    Build a matrix (n_restaurants x n_cuisines) containing binary values for cuisine flags.
    If flags not present, build by parsing 'Primary Cuisine' (one-hot).
    """
    cols = [c for c in df.columns if c.startswith(cuisine_prefix)]
    if len(cols) > 0:
        mat = df[cols].astype(float).values
        return mat, cols
    # fallback: try primary cuisine categories
    if "Primary Cuisine" in df.columns:
        one_hot = pd.get_dummies(df["Primary Cuisine"], prefix=cuisine_prefix)
        mat = one_hot.values
        cols = list(one_hot.columns)
        return mat, cols
    # else empty
    return None, []

def recommend_by_similarity(df: pd.DataFrame, item_idx: Optional[int]=None, user_pref_vector: Optional[np.ndarray]=None, top_k:int=10):
    """
    Content-based using cosine similarity:
    - If item_idx provided: compute similarity between that item and all others.
    - If user_pref_vector provided (same length as cuisine vector), compute similarity to user preference.
    Returns top_k candidates sorted by similarity.
    """
    mat, cols = build_cuisine_matrix(df)
    if mat is None or mat.shape[1] == 0:
        raise RuntimeError("No cuisine matrix available for similarity.")
    if item_idx is not None:
        anchor = mat[item_idx].reshape(1, -1)
        sims = cosine_similarity(anchor, mat).ravel()
    elif user_pref_vector is not None:
        # ensure shape
        up = np.asarray(user_pref_vector).reshape(1, -1)
        sims = cosine_similarity(up, mat).ravel()
    else:
        raise ValueError("Provide either item_idx or user_pref_vector.")
    dfc = df.copy()
    dfc["sim_score"] = sims
    return dfc.sort_values("sim_score", ascending=False).head(top_k)

def normalize_series(s):
    s = np.array(s, dtype=float)
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-9:
        return np.zeros_like(s)
    return (s - mn) / (mx - mn)

def hybrid_rank(df: pd.DataFrame, weight_pred=0.7, weight_votes=0.3, vote_col="Votes"):
    """
    Compute hybrid ranking score = weight_pred * (pred_rating/5) + weight_votes * normalized_votes
    Expects df to contain 'pred_rating' and vote_col.
    """
    if "pred_rating" not in df.columns:
        raise RuntimeError("pred_rating not in df; run model predictions first.")
    vnorm = normalize_series(df[vote_col].values) if vote_col in df.columns else np.zeros(len(df))
    pred_norm = df["pred_rating"].values / 5.0  # scale ratings to 0..1
    score = weight_pred * pred_norm + weight_votes * vnorm
    out = df.copy()
    out["hybrid_score"] = score
    return out.sort_values("hybrid_score", ascending=False)

def recommend_content_then_hybrid(df: pd.DataFrame, cuisine_list:List[str]=None, city:Optional[str]=None,
                                  user_pref_vector:Optional[np.ndarray]=None, item_idx:Optional[int]=None,
                                  model_predict_fn=None, top_k:int=10):
    """
    End-to-end: filter by cuisine + city, optionally use similarity (item_idx or user_pref_vector) to narrow,
    then add model predictions (if provided) and compute hybrid rank.
    Returns top_k rows with scores.
    """
    cand = filter_by_cuisine_and_city(df, cuisine_list, city)
    if cand.shape[0] == 0:
        return cand  # empty
    # if similarity step desired
    if item_idx is not None or user_pref_vector is not None:
        # if item_idx provided we need it relative to original df index; convert to position in cand if necessary.
        # simplest: run similarity on full df and then intersect with candidates
        sim_df = recommend_by_similarity(df, item_idx=item_idx, user_pref_vector=user_pref_vector, top_k=max(100, top_k*5))
        # keep only rows that are in cand
        sim_ids = sim_df.index
        cand = cand.loc[cand.index.intersection(sim_ids)]
        # re-order by sim_score descending
        cand = cand.reindex(sim_df.index).dropna(subset=[cand.columns[0]])
        # drop sim_score duplicates if any
    # apply model predict function if provided
    if model_predict_fn is not None:
        cand = model_predict_fn(cand)
    # compute hybrid ranking
    ranked = hybrid_rank(cand)
    return ranked.head(top_k)
