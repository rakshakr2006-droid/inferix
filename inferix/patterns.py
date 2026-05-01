"""
Inferix — Pattern Matcher (Layer 2)

Contains regex patterns and scoring functions for all regex-detectable
Indian semantic types. Each function takes a pandas Series and returns
a match score between 0.0 and 1.0 indicating what fraction of non-null
values match the pattern.

Supported regex-based types:
    pan_number, gst_number, aadhaar_number, ifsc_code, indian_mobile,
    indian_pincode, inr_currency, email_address, url, date_formatted,
    percentage, binary_flag
"""

import re

import pandas as pd


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _match_score(series: pd.Series, pattern: re.Pattern) -> float:
    """
    Calculate the fraction of non-null values in *series* that match *pattern*.

    Returns 0.0 if the series is empty or all-null.
    Values are converted to strings before matching.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0

    # Convert to string and strip whitespace for robust matching
    str_values = non_null.astype(str).str.strip()
    matches = str_values.str.fullmatch(pattern).sum()
    return float(matches) / len(non_null)


# ---------------------------------------------------------------------------
# Indian Document / ID Patterns
# ---------------------------------------------------------------------------

# PAN (Permanent Account Number): ABCDE1234F
# 5 uppercase letters + 4 digits + 1 uppercase letter
_PAN_RE = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")

def score_pan(series: pd.Series) -> float:
    """Score how well the column matches PAN number format."""
    return _match_score(series, _PAN_RE)


# GST (Goods and Services Tax Number): 22ABCDE1234F1Z5
# 2-digit state code (01-37) + 10-char PAN + 1 entity code + Z + 1 check char
_GST_RE = re.compile(
    r"[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]"
)

def score_gst(series: pd.Series) -> float:
    """Score how well the column matches GST number format."""
    return _match_score(series, _GST_RE)


# Aadhaar: 12 digits, first digit 2-9
_AADHAAR_RE = re.compile(r"[2-9][0-9]{11}")

def score_aadhaar(series: pd.Series) -> float:
    """Score how well the column matches Aadhaar number format."""
    return _match_score(series, _AADHAAR_RE)


# IFSC (Indian Financial System Code): SBIN0001234
# 4 uppercase letters + 0 + 6 alphanumeric characters
_IFSC_RE = re.compile(r"[A-Z]{4}0[A-Z0-9]{6}")

def score_ifsc(series: pd.Series) -> float:
    """Score how well the column matches IFSC code format."""
    return _match_score(series, _IFSC_RE)


# Indian Mobile: 10 digits starting with 6, 7, 8, or 9
_MOBILE_RE = re.compile(r"[6-9][0-9]{9}")

def score_indian_mobile(series: pd.Series) -> float:
    """Score how well the column matches Indian mobile number format."""
    return _match_score(series, _MOBILE_RE)


# Indian Pincode: 6 digits, first digit 1-9
_PINCODE_RE = re.compile(r"[1-9][0-9]{5}")

def score_indian_pincode(series: pd.Series) -> float:
    """Score how well the column matches Indian pincode format."""
    return _match_score(series, _PINCODE_RE)


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

# INR currency: optional ₹ symbol, optional commas, optional decimals
# Matches: ₹1,23,456.78  |  ₹ 5000  |  12345.50  |  1,00,000
_INR_RE = re.compile(
    r"₹?\s*[0-9]{1,3}(?:,?[0-9]{2,3})*(?:\.[0-9]{1,2})?"
)

def score_inr_currency(series: pd.Series) -> float:
    """
    Score how well the column matches INR currency format.

    This checks for the ₹ symbol first. If at least 10% of values
    have ₹, we use the regex. Otherwise we return 0.0 for the regex
    part (statistical features in Layer 3 may still detect currency).
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0

    str_values = non_null.astype(str).str.strip()

    # Check if ₹ symbol is present in a meaningful fraction
    has_rupee = str_values.str.contains("₹", regex=False).sum()
    rupee_fraction = float(has_rupee) / len(non_null)

    if rupee_fraction > 0.1:
        # Good signal — use full regex match
        return _match_score(series, _INR_RE)

    # Without ₹ symbol, INR currency looks like plain numbers.
    # We return 0.0 here; Layer 3 stats will handle numeric detection.
    return 0.0


# ---------------------------------------------------------------------------
# General Patterns
# ---------------------------------------------------------------------------

# Email address
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

def score_email(series: pd.Series) -> float:
    """Score how well the column matches email format."""
    return _match_score(series, _EMAIL_RE)


# URL
_URL_RE = re.compile(
    r"https?://[^\s/$.?#].[^\s]*"
)

def score_url(series: pd.Series) -> float:
    """Score how well the column matches URL format."""
    return _match_score(series, _URL_RE)


# Formatted dates — common Indian date formats
# DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, YYYY/MM/DD, DD.MM.YYYY
_DATE_RE = re.compile(
    r"(?:"
    r"[0-3]?[0-9][/\-\.][0-1]?[0-9][/\-\.][12][0-9]{3}"  # DD/MM/YYYY
    r"|"
    r"[12][0-9]{3}[/\-\.][0-1]?[0-9][/\-\.][0-3]?[0-9]"  # YYYY/MM/DD
    r"|"
    r"[0-3]?[0-9]\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+[12][0-9]{3}"  # DD Mon YYYY
    r")"
)

def score_date_formatted(series: pd.Series) -> float:
    """Score how well the column matches common date formats."""
    return _match_score(series, _DATE_RE)


# Percentage: number followed by %
_PERCENT_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?%")

def score_percentage(series: pd.Series) -> float:
    """
    Score how well the column matches percentage format (with % symbol).

    For numeric columns where values are 0-100 but without %,
    Layer 3 statistics will handle detection.
    """
    return _match_score(series, _PERCENT_RE)


# Binary flag: {0, 1, True, False, Yes, No, Y, N, true, false, yes, no}
_BINARY_VALUES = {
    "0", "1", "true", "false", "yes", "no", "y", "n",
    "True", "False", "Yes", "No", "Y", "N",
    "TRUE", "FALSE", "YES", "NO",
}

def score_binary_flag(series: pd.Series) -> float:
    """Score how well the column matches binary flag pattern."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0

    str_values = non_null.astype(str).str.strip()
    matches = str_values.isin(_BINARY_VALUES).sum()
    return float(matches) / len(non_null)


# ---------------------------------------------------------------------------
# Public API: Get all pattern scores at once
# ---------------------------------------------------------------------------

def get_all_pattern_scores(series: pd.Series) -> dict:
    """
    Run ALL pattern matchers against a single column.

    Parameters
    ----------
    series : pd.Series
        A single column from a DataFrame.

    Returns
    -------
    dict
        Keys are feature names (e.g. 'regex_pan'), values are match
        scores between 0.0 and 1.0.
    """
    return {
        "regex_pan": score_pan(series),
        "regex_gst": score_gst(series),
        "regex_aadhaar": score_aadhaar(series),
        "regex_ifsc": score_ifsc(series),
        "regex_mobile": score_indian_mobile(series),
        "regex_pincode": score_indian_pincode(series),
        "regex_inr": score_inr_currency(series),
        "regex_email": score_email(series),
        "regex_url": score_url(series),
        "regex_date": score_date_formatted(series),
        "regex_percentage": score_percentage(series),
        "regex_binary": score_binary_flag(series),
    }
