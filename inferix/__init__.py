"""
Inferix — India-aware semantic data type inferencer for CSV columns.

Detects 20 semantic types including PAN numbers, GST numbers,
Aadhaar numbers, IFSC codes, Indian mobile numbers, and more.

Usage:
    import pandas as pd
    from inferix import infer

    df = pd.read_csv("your_data.csv")
    results = infer(df)
    print(results)
"""

__version__ = "0.1.0"
__author__ = "Inferix Team"

from inferix.infer import infer  # noqa: F401

# All 20 semantic types that Inferix can detect
SEMANTIC_TYPES = [
    "pan_number",
    "gst_number",
    "aadhaar_number",
    "ifsc_code",
    "indian_mobile",
    "indian_pincode",
    "inr_currency",
    "email_address",
    "url",
    "date_formatted",
    "date_disguised",
    "timestamp",
    "percentage",
    "binary_flag",
    "id_column",
    "ratio",
    "count",
    "age",
    "category_low_card",
    "free_text",
]
