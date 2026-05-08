"""
setup_data.py
=============
Generates and saves 3 datasets for Lab 5 Q2 (Decision Trees).
Run once: python setup_data.py

Datasets created (inside data/<name>/):
  A) moons     — Two Moons (non-linear boundary), N=500, D=2
  B) iris      — Multi-class Iris (3 classes), N=150, D=4
  C) circles   — Concentric Circles (non-linear), N=500, D=2

Each sub-directory gets: train.csv, test.csv

DO NOT MODIFY this script.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, load_iris, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_DIR = "data"


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    out_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    standardise: bool = True,
) -> None:
    """Split and save a single dataset."""
    os.makedirs(out_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    if standardise:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["target"] = y_train

    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["target"] = y_test

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print(f"  Train: {train_df.shape}  →  {out_dir}/train.csv")
    print(f"  Test:  {test_df.shape}  →  {out_dir}/test.csv")

    # Sanity check
    print(f"  [Check] Feature-0 train mean={train_df.iloc[:,0].mean():.4f}, "
          f"std={train_df.iloc[:,0].std():.4f}")
    print(f"  [Check] Class distribution (train): {np.bincount(y_train.astype(int))}")


# ------------------------------------------------------------------
# Dataset generators
# ------------------------------------------------------------------

def generate_moons() -> None:
    """Dataset A — Two Moons (non-linear boundary), D=2."""
    print("\n--- Dataset A: Two Moons ---")
    X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
    feature_names = ["x1", "x2"]
    print(f"  Original: {X.shape}, class dist: {np.bincount(y)}")
    _save_dataset(X, y, feature_names, os.path.join(DATA_DIR, "moons"),
                  standardise=False)


def generate_iris_multiclass() -> None:
    """Dataset B — Full Iris (3 classes), D=4."""
    print("\n--- Dataset B: Iris (3-class) ---")
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)
    print(f"  Original: {X.shape}, class dist: {np.bincount(y)}")
    _save_dataset(X, y, feature_names, os.path.join(DATA_DIR, "iris"))


def generate_circles() -> None:
    """Dataset C — Concentric Circles (non-linear), D=2."""
    print("\n--- Dataset C: Concentric Circles ---")
    X, y = make_circles(n_samples=500, noise=0.15, factor=0.5,
                        random_state=42)
    feature_names = ["x1", "x2"]
    print(f"  Original: {X.shape}, class dist: {np.bincount(y)}")
    _save_dataset(X, y, feature_names, os.path.join(DATA_DIR, "circles"),
                  standardise=False)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def generate_lab_data() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    generate_moons()
    generate_iris_multiclass()
    generate_circles()
    print("\n✓ All datasets generated successfully.\n")


if __name__ == "__main__":
    generate_lab_data()
