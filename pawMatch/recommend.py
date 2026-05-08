"""
Cat Adoption Recommender — Recommendation Algorithm Engineer
============================================================
Role:       Recommendation Algorithm Engineer
Owns:       recommendation engine
Implements: weighted Euclidean similarity + filtering rules
Tasks:      normalize dataset, feature weights, distance calculation
Algorithm:  weighted Euclidean distance  d(x,y) = sqrt(sum_i w_i*(x_i-y_i)^2)

Usage
-----
from recommend import recommend

results = recommend(
    user_answers={
        "activity":               5,   # 1-7  → energy_level
        "affection_need":         6,   # 1-7  → affection
        "maintenance_willingness":4,   # 1-7  → grooming
        "noise_tolerance":        3,   # 1-7  → vocal
        "time_away":              2,   # 1-7  → independence
        # optional direct overrides (1-7):
        # "child_friendly": 6,
        # "pet_friendly":   4,
    },
    top_k=5,
    # optional filters:
    # sex_filter="Female",
    # max_neuroticism=4,
    # min_child_friendly=5,
    # min_pet_friendly=3,
    # weights_override={"affection": 0.30, "vocal": 0.20},
)
print(results)
"""

import pandas as pd
import numpy as np
from typing import Optional

# ── Schema constants ──────────────────────────────────────────────────────────
TRAIT_COLS = [
    "neuroticism", "energy_level", "affection", "child_friendly",
    "pet_friendly", "vocal", "trainability", "grooming",
    "independence", "dominance",
]

LIKERT_MIN, LIKERT_MAX = 1, 7

# ── User-trait → Cat-trait mapping ────────────────────────────────────────────
USER_TO_CAT = {
    "activity":                "energy_level",
    "affection_need":          "affection",
    "maintenance_willingness": "grooming",
    "noise_tolerance":         "vocal",
    "time_away":               "independence",
}

# ── Default feature weights (must sum to 1.0) ─────────────────────────────────
DEFAULT_WEIGHTS = {
    "neuroticism":    0.05,
    "energy_level":   0.15,
    "affection":      0.20,
    "child_friendly": 0.10,
    "pet_friendly":   0.05,
    "vocal":          0.15,
    "trainability":   0.05,
    "grooming":       0.10,
    "independence":   0.10,
    "dominance":      0.05,
}

# ── Module-level dataset (loaded once on import) ──────────────────────────────
_df_clean: Optional[pd.DataFrame] = None
_df_norm:  Optional[pd.DataFrame] = None

def _load_data(path: str = "cats_clean.xlsx") -> None:
    global _df_clean, _df_norm
    _df_clean = pd.read_excel(path, sheet_name="cats_clean")
    _df_norm  = _df_clean[["cat_id", "Cat_sex"]].copy()
    for c in TRAIT_COLS:
        _df_norm[c] = (_df_clean[c] - LIKERT_MIN) / (LIKERT_MAX - LIKERT_MIN)


# ── Step 1: Build user target profile ─────────────────────────────────────────

def build_user_profile(
    user_answers:    dict,
    weights_override: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert user lifestyle answers (1-7 scale) into a normalized target vector.

    Returns
    -------
    target_vector : np.ndarray (10,) — normalized [0, 1]
    weight_vector : np.ndarray (10,) — per-trait importance weights
    """
    weights = {**DEFAULT_WEIGHTS, **(weights_override or {})}
    target_raw = {trait: 0.5 for trait in TRAIT_COLS}  # neutral midpoint default

    for user_key, cat_trait in USER_TO_CAT.items():
        if user_key in user_answers:
            target_raw[cat_trait] = (user_answers[user_key] - LIKERT_MIN) / (LIKERT_MAX - LIKERT_MIN)

    # Allow direct cat-trait overrides
    for trait in TRAIT_COLS:
        if trait in user_answers:
            target_raw[trait] = (user_answers[trait] - LIKERT_MIN) / (LIKERT_MAX - LIKERT_MIN)

    target_vector = np.array([target_raw[t] for t in TRAIT_COLS])
    weight_vector = np.array([weights[t]    for t in TRAIT_COLS])
    weight_vector = weight_vector / weight_vector.sum()  # normalize to sum=1
    return target_vector, weight_vector


# ── Step 2: Weighted Euclidean distance ───────────────────────────────────────

def weighted_euclidean_distance(
    cat_matrix: np.ndarray,  # shape (N, 10)
    target:     np.ndarray,  # shape (10,)
    weights:    np.ndarray,  # shape (10,)
) -> np.ndarray:
    """d(x, y) = sqrt( sum_i  w_i * (x_i - y_i)^2 )"""
    diff         = cat_matrix - target
    weighted_sq  = weights * (diff ** 2)
    return np.sqrt(weighted_sq.sum(axis=1))


# ── Step 3: Filtering rules ───────────────────────────────────────────────────

def apply_filters(
    df:                 pd.DataFrame,
    sex_filter:         Optional[str] = None,
    max_neuroticism:    Optional[int] = None,
    min_child_friendly: Optional[int] = None,
    min_pet_friendly:   Optional[int] = None,
) -> pd.DataFrame:
    """Hard-filter cats before distance ranking (operates on normalized df)."""
    out = df.copy()
    if sex_filter:
        out = out[out["Cat_sex"] == sex_filter]
    if max_neuroticism is not None:
        out = out[out["neuroticism"] <= (max_neuroticism - LIKERT_MIN) / (LIKERT_MAX - LIKERT_MIN)]
    if min_child_friendly is not None:
        out = out[out["child_friendly"] >= (min_child_friendly - LIKERT_MIN) / (LIKERT_MAX - LIKERT_MIN)]
    if min_pet_friendly is not None:
        out = out[out["pet_friendly"] >= (min_pet_friendly - LIKERT_MIN) / (LIKERT_MAX - LIKERT_MIN)]
    return out


# ── Step 4: Rank & return top-K ───────────────────────────────────────────────

def rank_cats(
    df_norm_filtered: pd.DataFrame,
    target_vector:    np.ndarray,
    weight_vector:    np.ndarray,
    top_k:            int = 5,
) -> pd.DataFrame:
    """Sort cats by ascending distance, return top-k with similarity score."""
    cat_matrix  = df_norm_filtered[TRAIT_COLS].values
    distances   = weighted_euclidean_distance(cat_matrix, target_vector, weight_vector)
    result      = df_norm_filtered[["cat_id", "Cat_sex"]].copy().reset_index(drop=True)
    result["distance"]   = distances
    max_possible         = float(np.sqrt((weight_vector * 1.0 ** 2).sum()))
    result["similarity"] = ((1 - distances / max_possible) * 100).clip(0, 100).round(2)
    result = result.sort_values("distance").head(top_k).reset_index(drop=True)
    result.index += 1
    result.index.name = "rank"
    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────

def recommend(
    user_answers:       dict,
    top_k:              int = 5,
    weights_override:   Optional[dict] = None,
    sex_filter:         Optional[str] = None,
    max_neuroticism:    Optional[int] = None,
    min_child_friendly: Optional[int] = None,
    min_pet_friendly:   Optional[int] = None,
    data_path:          str = "cats_clean.xlsx",
) -> pd.DataFrame:
    """
    Full recommendation pipeline.

    Parameters (user_answers — all values 1-7 Likert scale)
    --------------------------------------------------------
    activity               : desired cat energy/playfulness
    affection_need         : desired cat affection level
    maintenance_willingness: grooming effort user can provide
    noise_tolerance        : tolerance for cat vocalisation
    time_away              : hours/frequency away (→ cat independence need)
    child_friendly         : (optional direct trait override)
    pet_friendly           : (optional direct trait override)
    + any other TRAIT_COL key for direct override

    Returns
    -------
    pd.DataFrame  rank | cat_id | Cat_sex | distance | similarity | <all traits>
    """
    global _df_clean, _df_norm
    if _df_clean is None:
        _load_data(data_path)

    target_vector, weight_vector = build_user_profile(user_answers, weights_override)
    df_filtered = apply_filters(
        _df_norm,
        sex_filter=sex_filter,
        max_neuroticism=max_neuroticism,
        min_child_friendly=min_child_friendly,
        min_pet_friendly=min_pet_friendly,
    )
    top_cats = rank_cats(df_filtered, target_vector, weight_vector, top_k=top_k)
    enriched = top_cats.merge(_df_clean[["cat_id"] + TRAIT_COLS], on="cat_id", how="left")
    enriched.index = top_cats.index
    return enriched


# ── CLI quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = recommend(
        user_answers={
            "activity":               4,
            "affection_need":         6,
            "maintenance_willingness":3,
            "noise_tolerance":        3,
            "time_away":              4,
        },
        top_k=5,
    )
    print(results.to_string())
