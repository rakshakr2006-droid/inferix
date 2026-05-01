"""
Tests for inferix.features — feature extraction pipeline.

Tests each layer independently and the combined extraction.
"""

import numpy as np
import pandas as pd
import pytest

from inferix.features import (
    extract_features,
    extract_features_vector,
    FEATURE_NAMES,
    NUM_FEATURES,
    _extract_syntactic_features,
    _extract_statistical_features,
    _extract_column_name_features,
    _normalize_column_name,
)


# ═══════════════════════════════════════════════════════════════════
# Layer 1 — Syntactic Features
# ═══════════════════════════════════════════════════════════════════

class TestSyntacticFeatures:
    def test_numeric_column(self):
        series = pd.Series([1, 2, 3, None, 5])
        features = _extract_syntactic_features(series)
        assert features["dtype_is_numeric"] == 1.0
        assert features["dtype_is_string"] == 0.0
        assert features["null_ratio"] == pytest.approx(0.2)
        assert features["total_count"] == 5.0

    def test_string_column(self):
        series = pd.Series(["a", "b", "c", "a"])
        features = _extract_syntactic_features(series)
        assert features["dtype_is_string"] == 1.0
        assert features["dtype_is_numeric"] == 0.0
        assert features["unique_ratio"] == pytest.approx(0.75)
        assert features["null_ratio"] == 0.0

    def test_all_null_column(self):
        series = pd.Series([None, None, None])
        features = _extract_syntactic_features(series)
        assert features["null_ratio"] == 1.0
        assert features["unique_ratio"] == 0.0

    def test_empty_column(self):
        series = pd.Series([], dtype=float)
        features = _extract_syntactic_features(series)
        assert features["null_ratio"] == 0.0
        assert features["total_count"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# Layer 3 — Statistical Features
# ═══════════════════════════════════════════════════════════════════

class TestStatisticalFeatures:
    def test_numeric_stats(self):
        series = pd.Series([10, 20, 30, 40, 50])
        features = _extract_statistical_features(series)
        assert features["stat_mean"] == pytest.approx(30.0)
        assert features["stat_min"] == 10.0
        assert features["stat_max"] == 50.0
        assert features["stat_value_range"] == 40.0
        assert features["stat_is_all_integer"] == 1.0
        assert features["stat_has_negatives"] == 0.0

    def test_negative_values(self):
        series = pd.Series([-5, -3, 0, 3, 5])
        features = _extract_statistical_features(series)
        assert features["stat_has_negatives"] == 1.0

    def test_string_stats(self):
        series = pd.Series(["hello", "world", "hi"])
        features = _extract_statistical_features(series)
        assert features["stat_avg_str_length"] > 0
        assert features["stat_mean"] == 0.0  # Not numeric

    def test_all_null(self):
        series = pd.Series([None, None], dtype=float)
        features = _extract_statistical_features(series)
        assert features["stat_mean"] == 0.0
        assert features["stat_entropy"] == 0.0

    def test_float_column(self):
        series = pd.Series([1.5, 2.7, 3.3])
        features = _extract_statistical_features(series)
        assert features["stat_is_all_integer"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# Layer 4 — Column Name Features
# ═══════════════════════════════════════════════════════════════════

class TestColumnNameFeatures:
    def test_pan_column_name(self):
        features = _extract_column_name_features("pan_number")
        assert features["name_pan_number"] == 1.0
        assert features["name_gst_number"] == 0.0

    def test_mobile_column_name(self):
        features = _extract_column_name_features("mobile_no")
        assert features["name_indian_mobile"] == 1.0

    def test_email_column_name(self):
        features = _extract_column_name_features("email_id")
        assert features["name_email_address"] == 1.0

    def test_unknown_column_name(self):
        features = _extract_column_name_features("xyz_abc_123")
        # Should all be 0.0 since no keywords match
        assert all(v == 0.0 for v in features.values())

    def test_returns_20_features(self):
        features = _extract_column_name_features("anything")
        assert len(features) == 20

    def test_normalize_camel_case(self):
        tokens = _normalize_column_name("customerPanNumber")
        assert "pan" in tokens
        assert "customer" in tokens
        assert "number" in tokens


# ═══════════════════════════════════════════════════════════════════
# Combined Feature Extraction
# ═══════════════════════════════════════════════════════════════════

class TestCombinedFeatures:
    def test_feature_dict_has_all_keys(self):
        series = pd.Series(["ABCDE1234F", "ZYXWV9876A"])
        features = extract_features(series, "pan_no")
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_feature_vector_shape(self):
        series = pd.Series([1, 2, 3, 4, 5])
        vec = extract_features_vector(series, "count")
        assert vec.shape == (NUM_FEATURES,)
        assert vec.dtype == np.float64

    def test_num_features_is_50(self):
        assert NUM_FEATURES == 50

    def test_no_nan_in_vector(self):
        series = pd.Series(["hello", None, "world"])
        vec = extract_features_vector(series, "text")
        assert not np.any(np.isnan(vec))
