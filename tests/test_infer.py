"""
Tests for inferix.infer — end-to-end inference tests.

These tests require a trained model. If the model doesn't exist,
training tests will be skipped.
"""

import os
import pytest
import pandas as pd

# Check if model exists
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "inferix", "model")
MODEL_EXISTS = (
    os.path.exists(os.path.join(MODEL_DIR, "inferix_model.json"))
    and os.path.exists(os.path.join(MODEL_DIR, "label_encoder.pkl"))
)

skip_no_model = pytest.mark.skipif(
    not MODEL_EXISTS,
    reason="Trained model not found. Run `python -m inferix.train` first.",
)


# ═══════════════════════════════════════════════════════════════════
# End-to-End Inference Tests
# ═══════════════════════════════════════════════════════════════════

@skip_no_model
class TestInfer:
    def test_pan_detection(self):
        from inferix import infer

        df = pd.DataFrame({
            "cust_pan": ["ABCDE1234F", "ZYXWV9876A", "MNOPQ5678R",
                         "BCDEG2345H", "KLMNO8901J"] * 20,
        })
        result = infer(df)
        assert len(result) == 1
        assert result.iloc[0]["semantic_type"] == "pan_number"

    def test_mobile_detection(self):
        from inferix import infer

        df = pd.DataFrame({
            "mobile_no": ["9876543210", "8765432109", "7654321098",
                          "6543210987", "9123456789"] * 20,
        })
        result = infer(df)
        assert result.iloc[0]["semantic_type"] == "indian_mobile"

    def test_email_detection(self):
        from inferix import infer

        df = pd.DataFrame({
            "email": [f"user_name_{i}@gmail.com" if i % 2 == 0 else f"usr{i}@yahoo.com" for i in range(100)]
        })
        result = infer(df)
        assert result.iloc[0]["semantic_type"] == "email_address"

    def test_multiple_columns(self):
        from inferix import infer

        df = pd.DataFrame({
            "pan_no": ["ABCDE1234F"] * 100,
            "phone": ["9876543210"] * 100,
            "email_id": ["user@gmail.com"] * 100,
        })
        result = infer(df)
        assert len(result) == 3
        assert set(result.columns) == {"column_name", "semantic_type",
                                        "confidence", "evidence"}

    def test_result_has_confidence(self):
        from inferix import infer

        df = pd.DataFrame({"pan": ["ABCDE1234F"] * 50})
        result = infer(df)
        conf_str = result.iloc[0]["confidence"]
        assert "%" in conf_str

    def test_empty_df_raises(self):
        from inferix import infer

        with pytest.raises(ValueError, match="empty"):
            infer(pd.DataFrame())

    def test_non_df_raises(self):
        from inferix import infer

        with pytest.raises(ValueError, match="DataFrame"):
            infer("not a dataframe")


@skip_no_model
class TestInferColumn:
    def test_single_column(self):
        from inferix.infer import infer_column

        series = pd.Series(["SBIN0001234", "HDFC0002345", "ICIC0003456"] * 30)
        result = infer_column(series, "ifsc_code")
        assert result["semantic_type"] == "ifsc_code"
        assert "%" in result["confidence"]
