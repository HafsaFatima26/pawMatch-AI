"""
Cat Adoption Recommender — Data Engineer / Knowledge Base Pipeline
==================================================================
Owns: dataset quality + structured cat data repository.

Pipeline stages:
    1. load_raw          — read source xlsx
    2. clean             — drop duplicates, strip whitespace
    3. handle_missing    — impute Cat_sex NaN/"Unsure" -> "Unknown"
    4. encode            — one-hot encode Cat_sex (nominal feature)
    5. add_schema        — add cat_id primary key, enforce dtypes
    6. validate          — assert trait ranges (1-7 Likert), no NaN
    7. normalize_traits  — scale traits to 0-1 for recommender (separate sheet)
    8. save              — write multi-sheet xlsx repository
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---- Schema constants ----------------------------------------------------
TRAIT_COLS = [
    'neuroticism', 'energy_level', 'affection', 'child_friendly',
    'pet_friendly', 'vocal', 'trainability', 'grooming',
    'independence', 'dominance'
]
LIKERT_MIN, LIKERT_MAX = 1, 7
SEX_CATEGORIES = ['Female', 'Male', 'Unknown']


# ---- Stage 1: Load -------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"[load] {len(df)} rows, {len(df.columns)} cols")
    return df


# ---- Stage 2: Clean ------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df['Cat_sex'] = df['Cat_sex'].astype(str).str.strip()
    print(f"[clean] removed {before - len(df)} duplicate rows")
    return df


# ---- Stage 3: Missing values --------------------------------------------
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Cat_sex: NaN and 'Unsure' -> 'Unknown' (preserves the row).
       Trait cols: assert no NaN (none in source; fail loud if that changes)."""
    df['Cat_sex'] = df['Cat_sex'].replace({'nan': 'Unknown', 'Unsure': 'Unknown'})
    df.loc[~df['Cat_sex'].isin(SEX_CATEGORIES), 'Cat_sex'] = 'Unknown'

    trait_nans = df[TRAIT_COLS].isnull().sum().sum()
    assert trait_nans == 0, f"Unexpected NaN in trait columns: {trait_nans}"
    print(f"[missing] Cat_sex categories: {df['Cat_sex'].value_counts().to_dict()}")
    return df


# ---- Stage 4: Encoding ---------------------------------------------------
def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Cat_sex (nominal). Keep original column for readability."""
    dummies = pd.get_dummies(df['Cat_sex'], prefix='sex').astype(int)
    for cat in SEX_CATEGORIES:
        col = f'sex_{cat}'
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f'sex_{c}' for c in SEX_CATEGORIES]]
    df = pd.concat([df, dummies], axis=1)
    print(f"[encode] added one-hot cols: {list(dummies.columns)}")
    return df


# ---- Stage 5: Schema -----------------------------------------------------
def add_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Primary key + canonical column order."""
    df.insert(0, 'cat_id', [f'CAT_{i:05d}' for i in range(1, len(df) + 1)])
    ordered = (['cat_id', 'Cat_sex'] + TRAIT_COLS
               + [f'sex_{c}' for c in SEX_CATEGORIES])
    df = df[ordered]
    for c in TRAIT_COLS:
        df[c] = df[c].astype(int)
    print(f"[schema] {len(df)} rows, primary key=cat_id")
    return df


# ---- Stage 6: Validation -------------------------------------------------
def validate(df: pd.DataFrame) -> pd.DataFrame:
    assert df['cat_id'].is_unique, "cat_id must be unique"
    for c in TRAIT_COLS:
        bad = ~df[c].between(LIKERT_MIN, LIKERT_MAX)
        assert bad.sum() == 0, f"{c} has {bad.sum()} out-of-range values"
    assert df['Cat_sex'].isin(SEX_CATEGORIES).all(), "Cat_sex has unknown values"
    print(f"[validate] all checks passed")
    return df


# ---- Stage 7: Normalize for recommender ---------------------------------
def normalize_traits(df: pd.DataFrame) -> pd.DataFrame:
    """Scale each trait from 1-7 -> 0-1 so the recommender's distance metric
       is not dominated by any single column. Returns a SEPARATE frame."""
    out = df[['cat_id']].copy()
    for c in TRAIT_COLS:
        out[c] = (df[c] - LIKERT_MIN) / (LIKERT_MAX - LIKERT_MIN)
    print(f"[normalize] traits scaled to [0, 1]")
    return out


# ---- Stage 8: Save -------------------------------------------------------
def save(df_clean: pd.DataFrame, df_norm: pd.DataFrame, out_path: str):
    """Multi-sheet repository:
        - cats_clean       : cleaned, encoded data on the original 1-7 scale
        - cats_normalized  : 0-1 scaled traits for the recommender
        - schema           : column dictionary
    """
    schema = pd.DataFrame([
        ('cat_id',         'string', 'Primary key, format CAT_#####'),
        ('Cat_sex',        'string', 'Female | Male | Unknown'),
        ('neuroticism',    'int 1-7', 'Likert: anxious / fearful tendency'),
        ('energy_level',   'int 1-7', 'Likert: activity / playfulness'),
        ('affection',      'int 1-7', 'Likert: bonding with humans'),
        ('child_friendly', 'int 1-7', 'Likert: tolerance of children'),
        ('pet_friendly',   'int 1-7', 'Likert: tolerance of other pets'),
        ('vocal',          'int 1-7', 'Likert: meowing / vocalisation'),
        ('trainability',   'int 1-7', 'Likert: responsiveness to training'),
        ('grooming',       'int 1-7', 'Likert: grooming need'),
        ('independence',   'int 1-7', 'Likert: comfort being alone'),
        ('dominance',      'int 1-7', 'Likert: assertiveness over other cats'),
        ('sex_Female',     'int 0/1', 'One-hot: female'),
        ('sex_Male',       'int 0/1', 'One-hot: male'),
        ('sex_Unknown',    'int 0/1', 'One-hot: unknown / unsure'),
    ], columns=['column', 'dtype', 'description'])

    with pd.ExcelWriter(out_path, engine='openpyxl') as w:
        df_clean.to_excel(w, sheet_name='cats_clean', index=False)
        df_norm.to_excel(w, sheet_name='cats_normalized', index=False)
        schema.to_excel(w, sheet_name='schema', index=False)
    print(f"[save] wrote {out_path}")


# ---- Pipeline orchestrator ----------------------------------------------
def run(input_path: str, output_path: str):
    df = load_raw(input_path)
    df = clean(df)
    df = handle_missing(df)
    df = encode(df)
    df = add_schema(df)
    df = validate(df)
    df_norm = normalize_traits(df)
    save(df, df_norm, output_path)
    return df, df_norm


if __name__ == '__main__':
    run('cats_10_traits.xlsx', 'cats_clean.xlsx')
    
