"""
Inferix — Main Inference API (Layer 5)

The public API for Inferix. Provides the `infer()` function that takes
a pandas DataFrame and returns predicted semantic types for each column
with confidence scores and evidence.

Usage:
    import pandas as pd
    from inferix import infer

    df = pd.read_csv("your_data.csv")
    results = infer(df)
    print(results)
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from inferix.features import extract_features, extract_features_vector, FEATURE_NAMES

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Model loading (cached at module level for performance)
# ---------------------------------------------------------------------------

_cached_model = None
_cached_encoder = None


def _get_model_dir() -> str:
    """Return the path to the model directory."""
    return os.path.join(os.path.dirname(__file__), "model")


def _load_model() -> tuple:
    """
    Load and cache the trained XGBoost model and LabelEncoder.

    Returns
    -------
    tuple of (xgb.XGBClassifier, LabelEncoder)
    """
    global _cached_model, _cached_encoder

    if _cached_model is not None and _cached_encoder is not None:
        return _cached_model, _cached_encoder

    model_dir = _get_model_dir()
    model_path = os.path.join(model_dir, "inferix_model.json")
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError(
            "Inferix model not found. Please train the model first:\n"
            "    python -m inferix.train\n"
            f"Expected model at: {model_path}\n"
            f"Expected encoder at: {encoder_path}"
        )

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    _cached_model = model
    _cached_encoder = encoder
    return model, encoder


# ---------------------------------------------------------------------------
# Evidence generation
# ---------------------------------------------------------------------------

def _generate_evidence(features: dict, predicted_type: str) -> str:
    """
    Generate a human-readable evidence string explaining WHY the
    prediction was made, based on the most informative features.

    Parameters
    ----------
    features : dict
        The extracted feature dictionary for this column.
    predicted_type : str
        The predicted semantic type.

    Returns
    -------
    str
        A short evidence string like "regex=0.94, name_match=yes"
    """
    evidence_parts = []

    # Check regex scores — these are the most interpretable
    regex_keys = [k for k in features if k.startswith("regex_")]
    top_regex = sorted(regex_keys, key=lambda k: features[k], reverse=True)
    for key in top_regex[:2]:
        score = features[key]
        if score > 0.1:
            evidence_parts.append(f"{key}={score:.2f}")

    # Check column name match
    name_key = f"name_{predicted_type}"
    if features.get(name_key, 0) > 0.5:
        evidence_parts.append("name_match=yes")

    # Statistical hints
    if features.get("stat_is_all_integer", 0) == 1.0:
        evidence_parts.append("all_int=True")
    if features.get("stat_has_negatives", 0) == 1.0:
        evidence_parts.append("has_neg=True")
    if features.get("unique_ratio", 0) > 0.9:
        evidence_parts.append("high_unique")
    elif features.get("unique_ratio", 0) < 0.05:
        evidence_parts.append("low_unique")

    if not evidence_parts:
        evidence_parts.append("statistical_profile")

    return ", ".join(evidence_parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer semantic data types for all columns in a DataFrame.

    This is the main public API of Inferix. It analyzes each column
    using 4 layers of feature extraction (syntactic, pattern matching,
    statistical profiling, column name analysis) and classifies using
    a trained XGBoost model.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to analyze.

    Returns
    -------
    pd.DataFrame
        Results with columns:
        - column_name: name of the analyzed column
        - semantic_type: predicted semantic type (one of 20 types)
        - confidence: prediction confidence as percentage string
        - evidence: human-readable explanation of the prediction

    Raises
    ------
    FileNotFoundError
        If the trained model is not found. Run `python -m inferix.train` first.
    ValueError
        If the input is not a pandas DataFrame or is empty.

    Examples
    --------
    >>> import pandas as pd
    >>> from inferix import infer
    >>> df = pd.DataFrame({
    ...     "pan": ["ABCDE1234F", "FGHIJ5678K"],
    ...     "mobile": ["9876543210", "8765432109"],
    ... })
    >>> results = infer(df)
    >>> print(results)
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    model, encoder = _load_model()

    results = []

    for col_name in df.columns:
        series = df[col_name]

        # Extract features
        features = extract_features(series, col_name)
        feat_vector = np.array(
            [features.get(f, 0.0) for f in FEATURE_NAMES],
            dtype=np.float64,
        ).reshape(1, -1)

        # Replace NaN/Inf
        feat_vector = np.nan_to_num(feat_vector, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict
        pred_proba = model.predict_proba(feat_vector)[0]
        pred_class_idx = int(np.argmax(pred_proba))
        confidence = float(pred_proba[pred_class_idx])
        predicted_type = encoder.inverse_transform([pred_class_idx])[0]

        # Generate evidence
        evidence = _generate_evidence(features, predicted_type)

        results.append({
            "column_name": col_name,
            "semantic_type": predicted_type,
            "confidence": f"{confidence:.0%}",
            "evidence": evidence,
        })

    return pd.DataFrame(results)


def infer_column(series: pd.Series, col_name: str = None) -> dict:
    """
    Infer the semantic type of a single column.

    Parameters
    ----------
    series : pd.Series
        The column data to analyze.
    col_name : str, optional
        Name of the column. If None, uses series.name or "unknown".

    Returns
    -------
    dict
        Dictionary with keys: semantic_type, confidence, evidence.
    """
    if col_name is None:
        col_name = series.name if series.name else "unknown"

    df = pd.DataFrame({col_name: series})
    result = infer(df)
    return result.iloc[0].to_dict()
