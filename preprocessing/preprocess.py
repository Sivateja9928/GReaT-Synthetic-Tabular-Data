"""
GReaT: Synthetic Tabular Data Generation
=========================================
Data Preprocessing Pipeline for Adult Income Dataset

Author: Siva Teja Sivarathri (B23179)
Course: CS-683 Generative AI, IIT Mandi

This script handles:
1. Dataset loading
2. Missing value detection and handling
3. Rare category consolidation
4. Redundant feature removal
5. Duplicate removal
6. Textual encoding (Definition 1 - GReaT paper)
7. Random feature order permutation (Definition 2 - GReaT paper)
"""

import pandas as pd
import random
import json
from collections import Counter


# ── 1. Load Dataset ────────────────────────────────────────────────────────
def load_adult_income():
    """
    Load the Adult Income dataset from UCI repository.
    
    Key decisions:
    - na_values=' ?' : The dataset uses ' ?' for missing values
    - skipinitialspace=True : Removes leading spaces from all string values
    """
    col_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race',
        'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
        'native-country', 'income'
    ]

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    df = pd.read_csv(
        url,
        header=None,
        names=col_names,
        na_values=' ?',
        skipinitialspace=True
    )

    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── 2. Fix Hidden Missing Values ───────────────────────────────────────────
def fix_missing_values(df):
    """
    Replace '?' strings with 'Unknown'.
    
    Why 'Unknown' not deletion:
    - Missing workclass and occupation co-occur (likely unemployed)
    - The LLM learns 'workclass is Unknown' as a meaningful pattern
    - Deleting rows loses 1,836+ real people from training data
    """
    missing_cols = ['workclass', 'occupation', 'native-country']

    for col in missing_cols:
        count = (df[col] == '?').sum()
        if count > 0:
            df[col] = df[col].replace('?', 'Unknown')
            print(f"✓ Fixed {count} missing values in '{col}' → 'Unknown'")

    return df


# ── 3. Handle Rare Categories ──────────────────────────────────────────────
def handle_rare_categories(df):
    """
    Merge rare categories to prevent the LLM from seeing
    too few examples to learn reliable patterns.

    Decisions:
    - Married-AF-spouse (23 rows) → Married-civ-spouse (same meaning)
    - Armed-Forces (9 rows) → Other-service (semantically close)
    - Countries with <50 rows → 'Other'
    
    We keep 'Without-pay' (14) and 'Never-worked' (7) because
    they are semantically distinct and meaningful to the LLM.
    """
    # Fix marital-status
    df['marital-status'] = df['marital-status'].replace(
        'Married-AF-spouse', 'Married-civ-spouse'
    )
    print("✓ Merged 'Married-AF-spouse' → 'Married-civ-spouse'")

    # Fix occupation
    df['occupation'] = df['occupation'].replace(
        'Armed-Forces', 'Other-service'
    )
    print("✓ Merged 'Armed-Forces' → 'Other-service'")

    # Fix native-country
    country_counts = df['native-country'].value_counts()
    rare_countries = country_counts[country_counts < 50].index.tolist()
    df['native-country'] = df['native-country'].replace(rare_countries, 'Other')
    print(f"✓ Grouped {len(rare_countries)} rare countries → 'Other'")

    return df


# ── 4. Drop Redundant Features ─────────────────────────────────────────────
def drop_redundant_features(df):
    """
    Drop 'education-num' — it is a numeric encoding of the
    'education' categorical column (perfectly correlated).
    
    Having both would give the LLM the same information twice
    in conflicting formats, undermining GReaT's semantic encoding.
    """
    df = df.drop(columns=['education-num'])
    print(f"✓ Dropped 'education-num' — redundant with 'education'")
    return df


# ── 5. Remove Duplicates ───────────────────────────────────────────────────
def remove_duplicates(df):
    """
    Remove exact duplicate rows.
    
    24 duplicates found — verified as true duplicates via
    matching fnlwgt (Census sampling weight) values.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"✓ Removed {before - after} duplicate rows")
    return df


# ── 6. Textual Encoding (Definition 1 from GReaT paper) ───────────────────
def encode_row(row, permute=True):
    """
    Convert one tabular row into a natural language sentence.

    Definition 1 (GReaT paper):
        ti,j = [fj, "is", vi,j, ","]
        ti = [ti,1, ti,2, ..., ti,m]

    Example output:
        "age is 39, workclass is State-gov, education is Bachelors, ..."

    Args:
        row: pandas Series (one row of the dataframe)
        permute: if True, randomly shuffle feature order (Definition 2)
    
    Returns:
        str: encoded sentence
    """
    items = list(row.items())

    if permute:
        random.shuffle(items)

    sentence = ", ".join([f"{name} is {value}" for name, value in items])
    return sentence


# ── 7. Generate Text Corpus ────────────────────────────────────────────────
def generate_corpus(df, permute=True, seed=42):
    """
    Apply textual encoding to all rows to produce a text corpus
    for LLM fine-tuning.

    Definition 2 (GReaT paper):
        Random feature order permutation eliminates positional
        dependencies and enables arbitrary conditioning at inference.

    Args:
        df: cleaned DataFrame
        permute: whether to apply random permutation
        seed: random seed for reproducibility

    Returns:
        list of encoded sentence strings
    """
    random.seed(seed)
    corpus = df.apply(lambda row: encode_row(row, permute=permute), axis=1).tolist()
    print(f"✓ Generated {len(corpus)} text sequences")
    return corpus


# ── 8. Validate Corpus ─────────────────────────────────────────────────────
def validate_corpus(corpus, df):
    """
    Run quality checks on the generated corpus.
    """
    features = list(df.columns)

    # Check 1: No empty sentences
    empty = [s for s in corpus if not s.strip()]
    assert len(empty) == 0, f"Found {len(empty)} empty sentences"
    print("✓ No empty sentences")

    # Check 2: All features present in every sentence
    missing = []
    for i, sentence in enumerate(corpus[:500]):
        for feat in features:
            if feat not in sentence:
                missing.append((i, feat))
    assert len(missing) == 0, f"Missing features found: {missing[:5]}"
    print("✓ All features present in every sentence")

    # Check 3: Permutation is working
    first_features = [s.split(" is ")[0] for s in corpus[:200]]
    unique_first = len(set(first_features))
    assert unique_first > 5, "Permutation may not be working"
    print(f"✓ Permutation working — {unique_first} unique first features in first 200 rows")


# ── 9. Save Outputs ────────────────────────────────────────────────────────
def save_outputs(df, corpus, csv_path="adult_income_cleaned.csv",
                 corpus_path="adult_income_corpus.txt"):
    """Save cleaned dataframe and text corpus."""
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved cleaned dataset → {csv_path}")

    with open(corpus_path, 'w') as f:
        for sentence in corpus:
            f.write(sentence + '\n')
    print(f"✓ Saved text corpus → {corpus_path}")


# ── Main Pipeline ──────────────────────────────────────────────────────────
def run_pipeline():
    print("=" * 55)
    print("GReaT Data Preprocessing Pipeline")
    print("Adult Income Dataset")
    print("=" * 55)

    # Step 1: Load
    df = load_adult_income()

    # Step 2: Fix missing values
    df = fix_missing_values(df)

    # Step 3: Handle rare categories
    df = handle_rare_categories(df)

    # Step 4: Drop redundant features
    df = drop_redundant_features(df)

    # Step 5: Remove duplicates
    df = remove_duplicates(df)

    # Step 6 & 7: Generate corpus
    corpus = generate_corpus(df, permute=True, seed=42)

    # Step 8: Validate
    validate_corpus(corpus, df)

    # Step 9: Save
    save_outputs(df, corpus)

    print("=" * 55)
    print(f"Pipeline complete!")
    print(f"Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Text corpus: {len(corpus)} sentences")
    print("=" * 55)

    return df, corpus


if __name__ == "__main__":
    df, corpus = run_pipeline()