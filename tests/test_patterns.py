"""
Tests for inferix.patterns — regex pattern scoring functions.

Tests each Indian-specific and general pattern with valid inputs,
invalid inputs, mixed inputs, and edge cases.
"""

import pandas as pd
import pytest

from inferix.patterns import (
    score_pan,
    score_gst,
    score_aadhaar,
    score_ifsc,
    score_indian_mobile,
    score_indian_pincode,
    score_inr_currency,
    score_email,
    score_url,
    score_date_formatted,
    score_percentage,
    score_binary_flag,
    get_all_pattern_scores,
)


# ═══════════════════════════════════════════════════════════════════
# PAN Number Tests
# ═══════════════════════════════════════════════════════════════════

class TestPANPattern:
    def test_valid_pan_numbers(self):
        series = pd.Series(["ABCDE1234F", "ZYXWV9876A", "MNOPQ5678R"])
        assert score_pan(series) == 1.0

    def test_invalid_pan_numbers(self):
        series = pd.Series(["12345ABCDE", "abcde1234f", "SHORT"])
        assert score_pan(series) == 0.0

    def test_mixed_pan_values(self):
        series = pd.Series(["ABCDE1234F", "not_a_pan", "ZYXWV9876A", "invalid"])
        score = score_pan(series)
        assert 0.4 <= score <= 0.6  # 2 out of 4

    def test_empty_series(self):
        assert score_pan(pd.Series([], dtype=str)) == 0.0

    def test_all_null(self):
        assert score_pan(pd.Series([None, None, None])) == 0.0


# ═══════════════════════════════════════════════════════════════════
# GST Number Tests
# ═══════════════════════════════════════════════════════════════════

class TestGSTPattern:
    def test_valid_gst_numbers(self):
        series = pd.Series(["22ABCDE1234F1Z5", "07ZYXWV9876A2ZB"])
        assert score_gst(series) == 1.0

    def test_invalid_gst_numbers(self):
        series = pd.Series(["INVALIDGST", "12345", "ABCDE1234F"])
        assert score_gst(series) == 0.0

    def test_empty_series(self):
        assert score_gst(pd.Series([], dtype=str)) == 0.0


# ═══════════════════════════════════════════════════════════════════
# Aadhaar Number Tests
# ═══════════════════════════════════════════════════════════════════

class TestAadhaarPattern:
    def test_valid_aadhaar(self):
        series = pd.Series(["234567890123", "912345678901", "567890123456"])
        assert score_aadhaar(series) == 1.0

    def test_invalid_aadhaar_starts_with_0_or_1(self):
        series = pd.Series(["012345678901", "112345678901"])
        assert score_aadhaar(series) == 0.0

    def test_invalid_aadhaar_wrong_length(self):
        series = pd.Series(["12345", "1234567890123456"])
        assert score_aadhaar(series) == 0.0


# ═══════════════════════════════════════════════════════════════════
# IFSC Code Tests
# ═══════════════════════════════════════════════════════════════════

class TestIFSCPattern:
    def test_valid_ifsc(self):
        series = pd.Series(["SBIN0001234", "HDFC0002345", "ICIC0003456"])
        assert score_ifsc(series) == 1.0

    def test_invalid_ifsc(self):
        series = pd.Series(["SBIN1001234", "1234ABCDE00", "SHORT"])
        assert score_ifsc(series) == 0.0


# ═══════════════════════════════════════════════════════════════════
# Indian Mobile Number Tests
# ═══════════════════════════════════════════════════════════════════

class TestMobilePattern:
    def test_valid_mobiles(self):
        series = pd.Series(["9876543210", "8765432109", "7654321098", "6543210987"])
        assert score_indian_mobile(series) == 1.0

    def test_invalid_mobiles_starting_with_low_digits(self):
        series = pd.Series(["1234567890", "0987654321", "5432109876"])
        assert score_indian_mobile(series) == 0.0

    def test_invalid_length(self):
        series = pd.Series(["98765", "987654321012345"])
        assert score_indian_mobile(series) == 0.0


# ═══════════════════════════════════════════════════════════════════
# Indian Pincode Tests
# ═══════════════════════════════════════════════════════════════════

class TestPincodePattern:
    def test_valid_pincodes(self):
        series = pd.Series(["110001", "560001", "400001", "700001"])
        assert score_indian_pincode(series) == 1.0

    def test_invalid_pincode_starts_with_0(self):
        series = pd.Series(["010001", "000000"])
        assert score_indian_pincode(series) == 0.0


# ═══════════════════════════════════════════════════════════════════
# INR Currency Tests
# ═══════════════════════════════════════════════════════════════════

class TestINRCurrencyPattern:
    def test_with_rupee_symbol(self):
        series = pd.Series(["₹1000", "₹ 5000.50", "₹10,00,000"])
        score = score_inr_currency(series)
        assert score > 0.5

    def test_without_rupee_symbol(self):
        # Should return 0.0 since regex relies on ₹ symbol
        series = pd.Series(["1000", "5000.50", "10000"])
        assert score_inr_currency(series) == 0.0


# ═══════════════════════════════════════════════════════════════════
# Email Tests
# ═══════════════════════════════════════════════════════════════════

class TestEmailPattern:
    def test_valid_emails(self):
        series = pd.Series(["user@gmail.com", "test@yahoo.co.in", "admin@company.org"])
        assert score_email(series) == 1.0

    def test_invalid_emails(self):
        series = pd.Series(["not_email", "missing@", "@nodomain"])
        assert score_email(series) == 0.0


# ═══════════════════════════════════════════════════════════════════
# URL Tests
# ═══════════════════════════════════════════════════════════════════

class TestURLPattern:
    def test_valid_urls(self):
        series = pd.Series(["https://google.com", "http://example.in/page"])
        assert score_url(series) == 1.0

    def test_invalid_urls(self):
        series = pd.Series(["just_text", "ftp://wrong", "www.no-protocol.com"])
        assert score_url(series) == 0.0


# ═══════════════════════════════════════════════════════════════════
# Date Formatted Tests
# ═══════════════════════════════════════════════════════════════════

class TestDateFormattedPattern:
    def test_dd_mm_yyyy(self):
        series = pd.Series(["15/08/2023", "26/01/2024", "01/12/1999"])
        assert score_date_formatted(series) == 1.0

    def test_yyyy_mm_dd(self):
        series = pd.Series(["2023-08-15", "2024-01-26"])
        assert score_date_formatted(series) == 1.0

    def test_invalid_dates(self):
        series = pd.Series(["not_a_date", "12345", "hello"])
        assert score_date_formatted(series) == 0.0


# ═══════════════════════════════════════════════════════════════════
# Percentage Tests
# ═══════════════════════════════════════════════════════════════════

class TestPercentagePattern:
    def test_with_percent_symbol(self):
        series = pd.Series(["45%", "99.5%", "0%", "100%"])
        assert score_percentage(series) == 1.0

    def test_without_percent_symbol(self):
        series = pd.Series(["45", "99.5", "0", "100"])
        assert score_percentage(series) == 0.0


# ═══════════════════════════════════════════════════════════════════
# Binary Flag Tests
# ═══════════════════════════════════════════════════════════════════

class TestBinaryFlagPattern:
    def test_zero_one(self):
        series = pd.Series(["0", "1", "0", "1", "1"])
        assert score_binary_flag(series) == 1.0

    def test_yes_no(self):
        series = pd.Series(["Yes", "No", "Yes", "No"])
        assert score_binary_flag(series) == 1.0

    def test_true_false(self):
        series = pd.Series(["True", "False", "True"])
        assert score_binary_flag(series) == 1.0

    def test_non_binary(self):
        series = pd.Series(["apple", "banana", "cherry"])
        assert score_binary_flag(series) == 0.0


# ═══════════════════════════════════════════════════════════════════
# Combined Score API Test
# ═══════════════════════════════════════════════════════════════════

class TestGetAllPatternScores:
    def test_returns_all_12_scores(self):
        series = pd.Series(["ABCDE1234F", "ZYXWV9876A"])
        scores = get_all_pattern_scores(series)
        assert len(scores) == 12
        assert all(isinstance(v, float) for v in scores.values())
        assert scores["regex_pan"] == 1.0  # These are valid PANs

    def test_all_scores_between_0_and_1(self):
        series = pd.Series(["hello", "world", "test"])
        scores = get_all_pattern_scores(series)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} is out of range"
