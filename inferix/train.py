"""
Inferix — Model Training Script

Generates synthetic training data, extracts features, trains an XGBoost
classifier, and saves the model + label encoder to inferix/model/.

Usage:
    python -m inferix.train

This will:
    1. Generate 3000 synthetic labelled columns (150 per type × 20 types)
    2. Extract 50 features per column
    3. Train an XGBoost classifier with 80/20 stratified split
    4. Print a classification report
    5. Save the model and label encoder
"""

import os
import pickle
import warnings

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb

from inferix.data_generator import generate_training_dataset
from inferix.features import extract_features_vector, FEATURE_NAMES

warnings.filterwarnings("ignore", category=UserWarning)


def _get_model_dir() -> str:
    """Return the path to the model directory."""
    return os.path.join(os.path.dirname(__file__), "model")


def train(
    columns_per_type: int = 200,
    rows_per_column: int = 200,
    test_size: float = 0.2,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Train the Inferix XGBoost model on synthetic data.

    Parameters
    ----------
    columns_per_type : int
        Number of synthetic columns per semantic type.
    rows_per_column : int
        Number of rows per synthetic column.
    test_size : float
        Fraction of data to use for testing.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        If True, print progress and results.

    Returns
    -------
    dict
        Training results with keys: 'accuracy', 'report', 'model_path'.
    """
    if verbose:
        print("=" * 60)
        print("  INFERIX MODEL TRAINING")
        print("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # -----------------------------------------------------------------------
    if verbose:
        total = columns_per_type * 20
        print(f"\n[1/4] Generating {total} synthetic columns "
              f"({columns_per_type} per type × 20 types)...")

    columns, labels, col_names = generate_training_dataset(
        columns_per_type=columns_per_type,
        rows_per_column=rows_per_column,
        seed=seed,
    )

    if verbose:
        print(f"       Generated {len(columns)} columns successfully.")

    # -----------------------------------------------------------------------
    # Step 2: Extract features
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[2/4] Extracting {len(FEATURE_NAMES)} features per column...")

    X = np.zeros((len(columns), len(FEATURE_NAMES)), dtype=np.float64)
    for i, (series, col_name) in enumerate(zip(columns, col_names)):
        if verbose and (i + 1) % 500 == 0:
            print(f"       Processed {i + 1}/{len(columns)} columns...")
        try:
            X[i] = extract_features_vector(series, col_name)
        except Exception as e:
            if verbose:
                print(f"       Warning: Failed to extract features for "
                      f"column {i} ({col_name}): {e}")
            # Leave as zeros — the model will learn to handle it

    # Replace any NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        print(f"       Feature matrix shape: {X.shape}")

    # -----------------------------------------------------------------------
    # Step 3: Train XGBoost model
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[3/4] Training XGBoost classifier...")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    if verbose:
        print(f"       Classes: {n_classes}")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    # XGBoost parameters — tuned for CPU, 8GB RAM
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=seed,
        n_jobs=-1,  # Use all CPU cores
        tree_method="hist",  # Fast on CPU
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
    )
    report_str = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
    )

    accuracy = report["accuracy"]

    if verbose:
        print(f"\n       Test Accuracy: {accuracy:.4f}")
        print(f"\n       Classification Report:")
        print(report_str)

    # -----------------------------------------------------------------------
    # Step 4: Save model and label encoder
    # -----------------------------------------------------------------------
    model_dir = _get_model_dir()
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "inferix_model.json")
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")

    model.save_model(model_path)
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    if verbose:
        print(f"\n[4/4] Model saved!")
        print(f"       Model:   {model_path}")
        print(f"       Encoder: {encoder_path}")
        print(f"\n{'=' * 60}")
        print(f"  TRAINING COMPLETE — Accuracy: {accuracy:.2%}")
        print(f"{'=' * 60}")

    return {
        "accuracy": accuracy,
        "report": report,
        "model_path": model_path,
    }


if __name__ == "__main__":
    train()
