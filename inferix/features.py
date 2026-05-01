"""
Inferix — Feature Extraction Pipeline (Layers 1–4)

Combines four layers of analysis into a single feature vector per column:

    Layer 1 — Syntactic Analyzer   (6 features)
    Layer 2 — Pattern Matcher      (12 features)  → from patterns.py
    Layer 3 — Statistical Profiler (12 features)
    Layer 4 — Column Name Matcher  (20 features)  → keyword-based

Total: 50 features per column.
"""

import re

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from inferix.patterns import get_all_pattern_scores

# ---------------------------------------------------------------------------
# Layer 4 — Column Name Keyword Map
# ---------------------------------------------------------------------------
# Maps each semantic type to a set of keywords commonly found in column names.
# Used for lightweight column-name-based hinting without sentence-transformers.

COLUMN_NAME_KEYWORDS = {
    "pan_number": {
        "pan", "pan_no", "pan_number", "pan_num", "pannumber",
        "permanent_account", "pancard",
    },
    "gst_number": {
        "gst", "gstin", "gst_no", "gst_number", "gstno", "gstnumber",
        "gst_code", "goods_service",
    },
    "aadhaar_number": {
        "aadhaar", "aadhar", "aadhaar_no", "aadhaar_number", "aadhar_no",
        "uid", "uidai", "aadhaarno",
    },
    "ifsc_code": {
        "ifsc", "ifsc_code", "ifsccode", "bank_code", "ifsc_no",
    },
    "indian_mobile": {
        "mobile", "mobile_no", "mobile_number", "phone", "phone_no",
        "phone_number", "contact", "contact_no", "cell", "mob",
        "mobileno", "phoneno", "telephone", "tel",
    },
    "indian_pincode": {
        "pincode", "pin_code", "pin", "zipcode", "zip_code", "zip",
        "postal_code", "postalcode", "pin_no",
    },
    "inr_currency": {
        "amount", "amt", "price", "cost", "salary", "income", "revenue",
        "payment", "fee", "charge", "total", "subtotal", "balance",
        "inr", "rupee", "rupees", "rs", "fare", "rent", "tax",
        "debit", "credit", "wages", "turnover",
    },
    "email_address": {
        "email", "email_id", "emailid", "email_address", "mail",
        "e_mail", "emailaddress",
    },
    "url": {
        "url", "website", "link", "href", "web", "site", "webpage",
        "homepage", "uri",
    },
    "date_formatted": {
        "date", "dob", "date_of_birth", "joining_date", "join_date",
        "start_date", "end_date", "created_at", "updated_at",
        "birth_date", "registration_date", "expiry_date", "due_date",
        "order_date", "delivery_date", "invoice_date",
    },
    "date_disguised": {
        "date_int", "dateint", "date_num", "datenum", "date_code",
        "datecode", "ymd", "yyyymmdd",
    },
    "timestamp": {
        "timestamp", "ts", "time", "datetime", "created_ts",
        "updated_ts", "epoch", "unix_time", "unixtime",
    },
    "percentage": {
        "percentage", "percent", "pct", "rate", "ratio_pct",
        "accuracy", "completion", "progress",
    },
    "binary_flag": {
        "flag", "is_active", "is_valid", "is_deleted", "status",
        "active", "enabled", "disabled", "boolean", "bool",
        "yes_no", "true_false", "is_verified", "approved",
    },
    "id_column": {
        "id", "uid", "uuid", "identifier", "record_id", "row_id",
        "serial", "sr_no", "sno", "s_no", "sl_no", "index",
        "customer_id", "cust_id", "user_id", "emp_id", "employee_id",
        "order_id", "transaction_id", "txn_id",
    },
    "ratio": {
        "ratio", "proportion", "fraction", "share", "weight",
        "probability", "prob", "likelihood", "score_ratio",
    },
    "count": {
        "count", "cnt", "num", "number", "qty", "quantity",
        "total_count", "frequency", "freq", "occurrences",
    },
    "age": {
        "age", "age_years", "age_yr", "years_old", "age_group",
    },
    "category_low_card": {
        "category", "cat", "type", "class", "group", "segment",
        "label", "tier", "grade", "level", "department", "dept",
        "gender", "sex", "state", "city", "region", "zone",
    },
    "free_text": {
        "text", "description", "desc", "comment", "comments",
        "note", "notes", "remark", "remarks", "address", "addr",
        "feedback", "review", "message", "msg", "narrative",
        "summary", "detail", "details", "observation",
    },
}

# Feature names — fixed order for the model
FEATURE_NAMES = (
    # Layer 1: Syntactic (6)
    ["null_ratio", "unique_ratio", "dtype_is_numeric",
     "dtype_is_string", "dtype_is_bool", "total_count"]
    # Layer 2: Pattern scores (12)
    + ["regex_pan", "regex_gst", "regex_aadhaar", "regex_ifsc",
       "regex_mobile", "regex_pincode", "regex_inr", "regex_email",
       "regex_url", "regex_date", "regex_percentage", "regex_binary"]
    # Layer 3: Statistical (12)
    + ["stat_mean", "stat_std", "stat_skewness", "stat_kurtosis",
       "stat_entropy", "stat_min", "stat_max", "stat_is_all_integer",
       "stat_has_negatives", "stat_value_range",
       "stat_avg_str_length", "stat_std_str_length"]
    # Layer 4: Column name keyword scores (20)
    + [f"name_{t}" for t in [
        "pan_number", "gst_number", "aadhaar_number", "ifsc_code",
        "indian_mobile", "indian_pincode", "inr_currency", "email_address",
        "url", "date_formatted", "date_disguised", "timestamp",
        "percentage", "binary_flag", "id_column", "ratio",
        "count", "age", "category_low_card", "free_text",
    ]]
)

NUM_FEATURES = len(FEATURE_NAMES)  # 50


# ---------------------------------------------------------------------------
# Layer 1 — Syntactic Analyzer
# ---------------------------------------------------------------------------

def _extract_syntactic_features(series: pd.Series) -> dict:
    """Extract basic syntactic features from a pandas Series."""
    total = len(series)
    null_count = series.isna().sum()
    unique_count = series.nunique(dropna=True)

    null_ratio = float(null_count) / total if total > 0 else 0.0
    unique_ratio = float(unique_count) / total if total > 0 else 0.0

    dtype = series.dtype
    is_numeric = 1.0 if pd.api.types.is_numeric_dtype(dtype) else 0.0
    is_string = 1.0 if pd.api.types.is_string_dtype(dtype) or dtype == object else 0.0
    is_bool = 1.0 if pd.api.types.is_bool_dtype(dtype) else 0.0

    return {
        "null_ratio": null_ratio,
        "unique_ratio": unique_ratio,
        "dtype_is_numeric": is_numeric,
        "dtype_is_string": is_string,
        "dtype_is_bool": is_bool,
        "total_count": float(total),
    }


# ---------------------------------------------------------------------------
# Layer 3 — Statistical Profiler
# ---------------------------------------------------------------------------

def _extract_statistical_features(series: pd.Series) -> dict:
    """Extract statistical features from a pandas Series."""
    defaults = {
        "stat_mean": 0.0,
        "stat_std": 0.0,
        "stat_skewness": 0.0,
        "stat_kurtosis": 0.0,
        "stat_entropy": 0.0,
        "stat_min": 0.0,
        "stat_max": 0.0,
        "stat_is_all_integer": 0.0,
        "stat_has_negatives": 0.0,
        "stat_value_range": 0.0,
        "stat_avg_str_length": 0.0,
        "stat_std_str_length": 0.0,
    }

    non_null = series.dropna()
    if len(non_null) == 0:
        return defaults

    # Try numeric analysis
    if pd.api.types.is_numeric_dtype(series):
        values = non_null.astype(float).values
        if len(values) == 0:
            return defaults

        defaults["stat_mean"] = float(np.mean(values))
        defaults["stat_std"] = float(np.std(values)) if len(values) > 1 else 0.0
        defaults["stat_min"] = float(np.min(values))
        defaults["stat_max"] = float(np.max(values))
        defaults["stat_value_range"] = defaults["stat_max"] - defaults["stat_min"]
        defaults["stat_has_negatives"] = 1.0 if np.any(values < 0) else 0.0

        # Check if all values are integers
        defaults["stat_is_all_integer"] = 1.0 if np.all(values == np.floor(values)) else 0.0

        # Skewness and kurtosis need at least 3 values
        if len(values) >= 3:
            defaults["stat_skewness"] = float(scipy_stats.skew(values, nan_policy="omit"))
            defaults["stat_kurtosis"] = float(scipy_stats.kurtosis(values, nan_policy="omit"))

        # Entropy of the value distribution (binned)
        if len(values) >= 2:
            # Use value counts for entropy
            _, counts = np.unique(values, return_counts=True)
            defaults["stat_entropy"] = float(scipy_stats.entropy(counts))
    else:
        # String-based features
        str_values = non_null.astype(str)
        lengths = str_values.str.len().astype(float)
        defaults["stat_avg_str_length"] = float(lengths.mean())
        defaults["stat_std_str_length"] = float(lengths.std()) if len(lengths) > 1 else 0.0

        # Entropy of string value distribution
        value_counts = str_values.value_counts().values
        if len(value_counts) > 0:
            defaults["stat_entropy"] = float(scipy_stats.entropy(value_counts))

    return defaults


# ---------------------------------------------------------------------------
# Layer 4 — Column Name Keyword Matcher
# ---------------------------------------------------------------------------

def _normalize_column_name(col_name: str) -> set:
    """
    Normalize a column name into a set of tokens for keyword matching.

    Example: "Customer_PAN_Number" → {"customer", "pan", "number"}
    Example: "customerPanNumber"   → {"customer", "pan", "number"}
    """
    name = col_name.strip()
    # Split on common separators: underscore, space, dash, dot
    tokens = re.split(r"[_\s\-\.]+", name)
    # Also split camelCase BEFORE lowercasing: "customerPan" → ["customer", "Pan"]
    expanded = []
    for token in tokens:
        # Split on camelCase boundaries (must happen before lowering)
        parts = re.findall(r"[a-z]+|[A-Z][a-z]*|[A-Z]+(?=[A-Z]|$)|[0-9]+", token)
        expanded.extend([p.lower() for p in parts])
    # Also keep the full lowered name for exact matching
    expanded.append(name.lower())
    return set(expanded)


def _extract_column_name_features(col_name: str) -> dict:
    """
    Score a column name against keyword sets for each semantic type.

    Returns a dict with 20 features (one per type), each 0.0 or 1.0.
    A score of 1.0 means at least one keyword matched.
    """
    tokens = _normalize_column_name(col_name)
    features = {}
    for sem_type, keywords in COLUMN_NAME_KEYWORDS.items():
        # Check if any keyword (or keyword tokens) match column tokens
        match = 0.0
        for kw in keywords:
            kw_tokens = set(kw.lower().split("_"))
            if kw_tokens.issubset(tokens) or kw.lower() in tokens:
                match = 1.0
                break
        features[f"name_{sem_type}"] = match
    return features


# ---------------------------------------------------------------------------
# Combined Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(series: pd.Series, col_name: str) -> dict:
    """
    Extract all 50 features for a single column.

    Parameters
    ----------
    series : pd.Series
        The column data.
    col_name : str
        The name of the column (used for keyword matching).

    Returns
    -------
    dict
        A dictionary with 50 feature name → value pairs.
    """
    features = {}

    # Layer 1: Syntactic
    features.update(_extract_syntactic_features(series))

    # Layer 2: Pattern scores
    features.update(get_all_pattern_scores(series))

    # Layer 3: Statistical
    features.update(_extract_statistical_features(series))

    # Layer 4: Column name keywords
    features.update(_extract_column_name_features(col_name))

    return features


def extract_features_vector(series: pd.Series, col_name: str) -> np.ndarray:
    """
    Extract features as a numpy array in the canonical feature order.

    Parameters
    ----------
    series : pd.Series
        The column data.
    col_name : str
        The name of the column.

    Returns
    -------
    np.ndarray
        Feature vector of shape (50,).
    """
    feat_dict = extract_features(series, col_name)
    return np.array([feat_dict.get(f, 0.0) for f in FEATURE_NAMES], dtype=np.float64)
