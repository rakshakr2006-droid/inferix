"""
Inferix — Synthetic Data Generator (v2 — Improved)

Generates realistic synthetic data for all 20 semantic types.
Used to create training data for the XGBoost classifier.

Key improvements over v1:
  - 50% of training columns use GENERIC column names (col_1, field_a, etc.)
    so the model learns to classify from data patterns, not just names.
  - INR currency generates BOTH ₹-prefixed and plain numeric amounts.
  - ID column is more distinctive (always sequential, always high unique ratio).
  - Pincode uses realistic Indian pincode ranges with repetition.
  - Age uses tight 0-120 range with realistic distribution.
  - Date disguised uses clear YYYYMMDD range.
  - More variation in column sizes (50-500 rows).
"""

import random
import string
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# Generic column names that DON'T match any keyword set.
# Used for 50% of training data to force model to learn from data patterns.
GENERIC_COLUMN_NAMES = [
    "col_1", "col_2", "col_3", "field_a", "field_b", "field_c",
    "var1", "var2", "var3", "data_col", "attribute", "value",
    "column_x", "column_y", "info", "detail", "entry",
    "item", "record", "param", "metric", "feature_1",
    "input_1", "output_1", "result", "raw_data", "processed",
]


def _inject_noise(series: pd.Series, noise_ratio: float = 0.05) -> pd.Series:
    """Replace a fraction of values with NaN to simulate real data."""
    s = series.copy()
    n_noise = int(len(s) * noise_ratio)
    if n_noise == 0:
        return s
    noise_indices = np.random.choice(len(s), size=n_noise, replace=False)
    for idx in noise_indices:
        s.iloc[idx] = np.nan  # Missing value only — no "INVALID" junk
    return s


# ---------------------------------------------------------------------------
# Indian Document Types
# ---------------------------------------------------------------------------

def generate_pan(n: int = 200) -> pd.Series:
    """Generate realistic PAN numbers (mix of high and low uniqueness)."""
    entity_types = list("ABCFGHLJPT")
    values = []
    
    is_high_unique = random.random() < 0.7
    if is_high_unique:
        for _ in range(n):
            pan = (
                "".join(random.choices(string.ascii_uppercase, k=2))
                + random.choice(entity_types)
                + "".join(random.choices(string.ascii_uppercase, k=2))
                + "".join(random.choices(string.digits, k=4))
                + random.choice(string.ascii_uppercase)
            )
            values.append(pan)
    else:
        pool_size = max(3, n // 10)
        pool = []
        for _ in range(pool_size):
            pan = (
                "".join(random.choices(string.ascii_uppercase, k=2))
                + random.choice(entity_types)
                + "".join(random.choices(string.ascii_uppercase, k=2))
                + "".join(random.choices(string.digits, k=4))
                + random.choice(string.ascii_uppercase)
            )
            pool.append(pan)
        for _ in range(n):
            values.append(random.choice(pool))
            
    return _inject_noise(pd.Series(values, name="pan_no"))


def generate_gst(n: int = 200) -> pd.Series:
    """Generate realistic GST numbers (mix of high and low uniqueness)."""
    state_codes = [f"{i:02d}" for i in range(1, 38)]
    values = []
    
    is_high_unique = random.random() < 0.7
    if is_high_unique:
        for _ in range(n):
            state = random.choice(state_codes)
            pan_part = (
                "".join(random.choices(string.ascii_uppercase, k=5))
                + "".join(random.choices(string.digits, k=4))
                + random.choice(string.ascii_uppercase)
            )
            entity = random.choice(string.digits[1:] + string.ascii_uppercase)
            check = random.choice(string.digits + string.ascii_uppercase)
            gst = state + pan_part + entity + "Z" + check
            values.append(gst)
    else:
        pool_size = max(3, n // 10)
        pool = []
        for _ in range(pool_size):
            state = random.choice(state_codes)
            pan_part = (
                "".join(random.choices(string.ascii_uppercase, k=5))
                + "".join(random.choices(string.digits, k=4))
                + random.choice(string.ascii_uppercase)
            )
            entity = random.choice(string.digits[1:] + string.ascii_uppercase)
            check = random.choice(string.digits + string.ascii_uppercase)
            gst = state + pan_part + entity + "Z" + check
            pool.append(gst)
        for _ in range(n):
            values.append(random.choice(pool))
            
    return _inject_noise(pd.Series(values, name="gstin"))


def generate_aadhaar(n: int = 200) -> pd.Series:
    """Generate realistic Aadhaar numbers (mix of high and low uniqueness)."""
    values = []
    
    is_high_unique = random.random() < 0.7
    if is_high_unique:
        for _ in range(n):
            first = str(random.randint(2, 9))
            rest = "".join(random.choices(string.digits, k=11))
            values.append(first + rest)
    else:
        pool_size = max(3, n // 10)
        pool = []
        for _ in range(pool_size):
            first = str(random.randint(2, 9))
            rest = "".join(random.choices(string.digits, k=11))
            pool.append(first + rest)
        for _ in range(n):
            values.append(random.choice(pool))
            
    return _inject_noise(pd.Series(values, name="aadhaar_no"))


def generate_ifsc(n: int = 200) -> pd.Series:
    """Generate realistic IFSC codes (mix of high and low uniqueness)."""
    bank_prefixes = ["SBIN", "HDFC", "ICIC", "UTIB", "PUNB", "BARB",
                     "CNRB", "BKID", "IOBA", "UBIN", "CORP", "VIJB",
                     "KKBK", "YESB", "IDIB", "ANDB"]
    values = []
    
    is_high_unique = random.random() < 0.7
    if is_high_unique:
        for _ in range(n):
            prefix = random.choice(bank_prefixes)
            suffix = "".join(random.choices(string.digits, k=6))
            values.append(prefix + "0" + suffix)
    else:
        pool_size = max(3, n // 10)
        pool = []
        for _ in range(pool_size):
            prefix = random.choice(bank_prefixes)
            suffix = "".join(random.choices(string.digits, k=6))
            pool.append(prefix + "0" + suffix)
        for _ in range(n):
            values.append(random.choice(pool))
            
    return _inject_noise(pd.Series(values, name="ifsc_code"))


def generate_indian_mobile(n: int = 200) -> pd.Series:
    """Generate realistic Indian mobile numbers (mix of high and low uniqueness)."""
    values = []
    
    is_high_unique = random.random() < 0.7
    if is_high_unique:
        for _ in range(n):
            first = str(random.randint(6, 9))
            rest = "".join(random.choices(string.digits, k=9))
            values.append(first + rest)
    else:
        pool_size = max(3, n // 10)
        pool = []
        for _ in range(pool_size):
            first = str(random.randint(6, 9))
            rest = "".join(random.choices(string.digits, k=9))
            pool.append(first + rest)
        for _ in range(n):
            values.append(random.choice(pool))
            
    return _inject_noise(pd.Series(values, name="mobile_no"))


def generate_indian_pincode(n: int = 200) -> pd.Series:
    """
    Generate realistic Indian pincodes.

    Key characteristics that differentiate from IDs:
    - Always 6 digits, range 100000-999999
    - Many REPEATED values (same pin for same area)
    - Low unique ratio (a city might have 20-50 pincodes)
    """
    # Real Indian pincode ranges by region
    pin_pools = [
        # North India: 1xxxxx - 2xxxxx
        list(range(110001, 110100)) + list(range(120001, 120050)),
        # West India: 3xxxxx - 4xxxxx
        list(range(400001, 400100)) + list(range(380001, 380050)),
        # South India: 5xxxxx - 6xxxxx
        list(range(560001, 560100)) + list(range(600001, 600050)),
        # East India: 7xxxxx
        list(range(700001, 700100)) + list(range(711101, 711150)),
        # Mixed
        list(range(110001, 110030)) + list(range(400001, 400030))
        + list(range(560001, 560030)) + list(range(700001, 700030)),
    ]
    pool = random.choice(pin_pools)
    # Sample WITH replacement to create realistic repetition
    values = [str(random.choice(pool)) for _ in range(n)]
    return _inject_noise(pd.Series(values, name="pincode"))


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

def generate_inr_currency(n: int = 200) -> pd.Series:
    """
    Generate INR currency values.

    IMPORTANT: 50% of the time generates plain numeric amounts WITHOUT ₹ symbol,
    since real CSVs often store currency as plain numbers.
    """
    use_symbol = random.random() < 0.5

    if use_symbol:
        is_high_unique = random.random() < 0.7
        if is_high_unique:
            values = []
            for _ in range(n):
                amount = round(random.uniform(100, 1000000), 2)
                if random.random() < 0.5:
                    values.append(f"₹{amount:,.2f}")
                else:
                    values.append(f"₹ {amount:.2f}")
        else:
            pool_size = max(3, n // 10)
            pool_values = []
            for _ in range(pool_size):
                amount = round(random.uniform(100, 1000000), 2)
                if random.random() < 0.5:
                    pool_values.append(f"₹{amount:,.2f}")
                else:
                    pool_values.append(f"₹ {amount:.2f}")
            values = [random.choice(pool_values) for _ in range(n)]
        return _inject_noise(pd.Series(values, name="amount"))
    else:
        # Plain numeric currency — distinguishable by range and decimals
        is_high_unique = random.random() < 0.7
        if is_high_unique:
            values = np.random.lognormal(mean=8, sigma=2, size=n).clip(50, 5000000)
            if random.random() < 0.7:
                values = np.round(values, 2)
            else:
                values = np.round(values, 0)
        else:
            pool_size = max(3, n // 10)
            pool_values = np.random.lognormal(mean=8, sigma=2, size=pool_size).clip(50, 5000000)
            if random.random() < 0.7:
                pool_values = np.round(pool_values, 2)
            else:
                pool_values = np.round(pool_values, 0)
            values = [random.choice(pool_values) for _ in range(n)]
        return _inject_noise(pd.Series(values, name="amount"))


# ---------------------------------------------------------------------------
# General Types
# ---------------------------------------------------------------------------

def generate_email(n: int = 200) -> pd.Series:
    """Generate realistic email addresses (mix of high and low uniqueness)."""
    domains = ["gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com",
               "hotmail.com", "company.co.in", "org.in", "proton.me"]
    values = []
    
    is_high_unique = random.random() < 0.7
    
    if is_high_unique:
        for _ in range(n):
            name = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 12)))
            if random.random() < 0.4:
                name += str(random.randint(1, 999))
            domain = random.choice(domains)
            values.append(f"{name}@{domain}")
    else:
        pool_size = max(3, n // 10)
        pool = []
        for _ in range(pool_size):
            name = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 12)))
            if random.random() < 0.4:
                name += str(random.randint(1, 999))
            domain = random.choice(domains)
            pool.append(f"{name}@{domain}")
        for _ in range(n):
            values.append(random.choice(pool))
        
    return _inject_noise(pd.Series(values, name="email_id"))


def generate_url(n: int = 200) -> pd.Series:
    """Generate realistic URLs (mix of high and low uniqueness)."""
    domains = ["example.com", "mysite.in", "portal.gov.in",
               "shop.co.in", "blog.org", "data.gov.in"]
    values = []
    
    is_high_unique = random.random() < 0.7
    if is_high_unique:
        for _ in range(n):
            protocol = random.choice(["http://", "https://"])
            domain = random.choice(domains)
            path = "/".join(random.choices(string.ascii_lowercase, k=random.randint(1, 3)))
            values.append(f"{protocol}{domain}/{path}")
    else:
        pool_size = max(3, n // 10)
        pool = []
        for _ in range(pool_size):
            protocol = random.choice(["http://", "https://"])
            domain = random.choice(domains)
            path = "/".join(random.choices(string.ascii_lowercase, k=random.randint(1, 3)))
            pool.append(f"{protocol}{domain}/{path}")
        for _ in range(n):
            values.append(random.choice(pool))
            
    return _inject_noise(pd.Series(values, name="website"))


def generate_date_formatted(n: int = 200) -> pd.Series:
    """Generate dates in common Indian formats."""
    start = datetime(1970, 1, 1)
    # Pick ONE format per column (real CSVs are consistent)
    fmt = random.choice(["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d",
                         "%d.%m.%Y", "%d %b %Y"])
    values = []
    for _ in range(n):
        days = random.randint(0, 20000)
        dt = start + timedelta(days=days)
        values.append(dt.strftime(fmt))
    return _inject_noise(pd.Series(values, name="date_of_birth"))


def generate_date_disguised(n: int = 200) -> pd.Series:
    """
    Generate disguised dates as integers (YYYYMMDD format).

    Key characteristics that differentiate from IDs:
    - Range: 19700101 to 20251231 (realistic date range)
    - Not sequential — random dates
    - Has statistical signature: bounded range, specific mean/std
    """
    start = datetime(1970, 1, 1)
    values = []
    for _ in range(n):
        days = random.randint(0, 20000)
        dt = start + timedelta(days=days)
        values.append(int(dt.strftime("%Y%m%d")))
    return _inject_noise(pd.Series(values, name="date_code"))


def generate_timestamp(n: int = 200) -> pd.Series:
    """Generate Unix timestamps (10-digit integers near 1.6 billion)."""
    base = int(datetime(2000, 1, 1).timestamp())
    values = []
    for _ in range(n):
        ts = base + random.randint(0, 800_000_000)
        values.append(ts)
    return _inject_noise(pd.Series(values, name="created_ts"))


def generate_percentage(n: int = 200) -> pd.Series:
    """Generate percentage values — sometimes with % symbol, sometimes without."""
    if random.random() < 0.5:
        # With symbol
        values = [f"{round(random.uniform(0, 100), 2)}%" for _ in range(n)]
    else:
        # Plain numbers in 0-100 range
        values = [round(random.uniform(0, 100), 2) for _ in range(n)]
    return _inject_noise(pd.Series(values, name="completion_pct"))


def generate_binary_flag(n: int = 200) -> pd.Series:
    """Generate binary flag values."""
    flag_sets = [
        ["0", "1"],
        ["Yes", "No"],
        ["True", "False"],
        ["Y", "N"],
        [0, 1],
    ]
    chosen = random.choice(flag_sets)
    values = [random.choice(chosen) for _ in range(n)]
    return _inject_noise(pd.Series(values, name="is_active"))


def generate_id_column(n: int = 200) -> pd.Series:
    """
    Generate ID-like values.

    Key characteristics that differentiate from other numeric types:
    - ALWAYS nearly 100% unique
    - Sequential (monotonically increasing) OR UUID strings
    - For numeric IDs: large range, no statistical pattern
    """
    style = random.choice(["sequential", "uuid_short", "uuid_full",
                            "prefixed", "large_int"])
    if style == "sequential":
        start = random.randint(1, 100000)
        values = list(range(start, start + n))
    elif style == "uuid_short":
        values = [str(uuid.uuid4())[:8].upper() for _ in range(n)]
    elif style == "uuid_full":
        values = [str(uuid.uuid4()) for _ in range(n)]
    elif style == "prefixed":
        prefix = random.choice(["ID-", "REC-", "TXN-", "ORD-", "EMP-"])
        values = [f"{prefix}{i:06d}" for i in range(1, n + 1)]
    else:  # large_int — clearly different from pincode/age
        values = [random.randint(1000000000, 9999999999) for _ in range(n)]
    return pd.Series(values, name="record_id")


def generate_ratio(n: int = 200) -> pd.Series:
    """Generate ratio values (0.0 to 1.0)."""
    values = np.random.beta(2, 5, size=n)
    values = np.round(values, 4)
    return _inject_noise(pd.Series(values, name="weight_ratio"))


def generate_count(n: int = 200) -> pd.Series:
    """
    Generate count values (non-negative integers, right-skewed).

    Key characteristics:
    - Always non-negative integers
    - Right-skewed (many small values, few large)
    - Many REPEATED values (low unique ratio)
    - Range typically 0 to ~500
    """
    values = np.random.exponential(scale=30, size=n).astype(int)
    values = np.clip(values, 0, 1000)
    return _inject_noise(pd.Series(values, name="order_count"))


def generate_age(n: int = 200) -> pd.Series:
    """
    Generate age values.

    Key characteristics that differentiate from IDs:
    - Tight range: 0 to 120
    - Normal distribution centered around 30-40
    - Many REPEATED values
    - Always non-negative integers
    """
    center = random.choice([25, 30, 35, 40])  # Vary center
    values = np.random.normal(loc=center, scale=12, size=n).astype(int)
    values = np.clip(values, 0, 100)
    return _inject_noise(pd.Series(values, name="age_years"))


def generate_category_low_card(n: int = 200) -> pd.Series:
    """Generate categorical values with low cardinality."""
    category_sets = [
        ["Male", "Female", "Other"],
        ["North", "South", "East", "West", "Central"],
        ["Bronze", "Silver", "Gold", "Platinum"],
        ["A", "B", "C", "D", "E"],
        ["Active", "Inactive", "Pending", "Suspended"],
        ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Gujarat",
         "Rajasthan", "Uttar Pradesh", "West Bengal"],
        ["Full Time", "Part Time", "Contract", "Intern"],
        ["Low", "Medium", "High"],
        ["Single", "Married", "Divorced", "Widowed"],
    ]
    chosen = random.choice(category_sets)
    values = [random.choice(chosen) for _ in range(n)]
    return _inject_noise(pd.Series(values, name="category"))


def generate_free_text(n: int = 200) -> pd.Series:
    """
    Generate free-text values with HIGH uniqueness.

    Key characteristics:
    - Long strings (avg > 20 chars)
    - Very high unique ratio (> 80%)
    - Contains spaces and varied content
    """
    subjects = [
        "Customer", "Manager", "Agent", "System", "User",
        "Employee", "Client", "Partner", "Vendor", "Team",
    ]
    actions = [
        "submitted inquiry about", "raised complaint regarding",
        "requested update for", "reported issue with",
        "completed verification of", "initiated process for",
        "approved request for", "rejected application for",
        "escalated ticket about", "processed payment for",
        "reviewed document for", "scheduled meeting about",
    ]
    objects = [
        "account balance", "loan application", "insurance claim",
        "KYC documents", "policy renewal", "fund transfer",
        "credit card", "debit mandate", "investment portfolio",
        "tax filing", "refund request", "service upgrade",
        "address change", "mobile banking", "net banking access",
    ]
    values = []
    for _ in range(n):
        subj = random.choice(subjects)
        act = random.choice(actions)
        obj = random.choice(objects)
        text = f"{subj} {act} {obj}"
        # Add unique suffix for high uniqueness
        text += f" (Ref #{random.randint(10000, 99999)})"
        if random.random() < 0.3:
            text += f" on {random.randint(1,28)}/{random.randint(1,12)}/2024"
        values.append(text)
    return _inject_noise(pd.Series(values, name="remarks"))


# ---------------------------------------------------------------------------
# Master generator: produce a full labelled dataset
# ---------------------------------------------------------------------------

GENERATORS = {
    "pan_number": generate_pan,
    "gst_number": generate_gst,
    "aadhaar_number": generate_aadhaar,
    "ifsc_code": generate_ifsc,
    "indian_mobile": generate_indian_mobile,
    "indian_pincode": generate_indian_pincode,
    "inr_currency": generate_inr_currency,
    "email_address": generate_email,
    "url": generate_url,
    "date_formatted": generate_date_formatted,
    "date_disguised": generate_date_disguised,
    "timestamp": generate_timestamp,
    "percentage": generate_percentage,
    "binary_flag": generate_binary_flag,
    "id_column": generate_id_column,
    "ratio": generate_ratio,
    "count": generate_count,
    "age": generate_age,
    "category_low_card": generate_category_low_card,
    "free_text": generate_free_text,
}

# Variant column names — mix of keyword-matching AND generic names
VARIANT_COLUMN_NAMES = {
    "pan_number": ["pan_no", "cust_pan", "PAN_Number", "pan", "pan_card",
                   "col_1", "field_a", "data_1", "attribute_x", "var1"],
    "gst_number": ["gstin", "gst_no", "GST_Code", "gst_number", "gst",
                   "col_2", "field_b", "data_2", "tax_id", "reg_no"],
    "aadhaar_number": ["aadhaar_no", "aadhar", "uid_number", "aadhaar", "aadhar_no",
                       "col_3", "field_c", "data_3", "id_proof", "doc_no"],
    "ifsc_code": ["ifsc_code", "bank_ifsc", "IFSC", "ifsc", "bank_code",
                  "col_4", "field_d", "data_4", "branch_info", "code_1"],
    "indian_mobile": ["mobile_no", "phone_number", "contact", "mob", "phone",
                      "col_5", "field_e", "data_5", "number_1", "reach_at"],
    "indian_pincode": ["pincode", "pin_code", "zip", "postal_code", "pin",
                       "col_6", "field_f", "data_6", "area_code", "location_code"],
    "inr_currency": ["amount", "total_amt", "price", "salary", "payment_inr",
                     "col_7", "field_g", "data_7", "value", "figure"],
    "email_address": ["email", "email_id", "mail", "email_address", "e_mail",
                      "col_8", "field_h", "data_8", "contact_info", "login"],
    "url": ["website", "url", "link", "homepage", "web_url",
            "col_9", "field_i", "data_9", "resource", "path"],
    "date_formatted": ["dob", "joining_date", "start_date", "created_at", "birth_date",
                       "col_10", "field_j", "data_10", "when", "day"],
    "date_disguised": ["date_int", "date_code", "ymd", "datenum", "date_value",
                       "col_11", "field_k", "data_11", "period", "ref_date"],
    "timestamp": ["timestamp", "created_ts", "epoch", "unix_time", "ts",
                  "col_12", "field_l", "data_12", "time_val", "marker"],
    "percentage": ["pct", "completion", "accuracy", "pass_rate", "percent",
                   "col_13", "field_m", "data_13", "score", "metric_1"],
    "binary_flag": ["is_active", "flag", "status", "is_valid", "approved",
                    "col_14", "field_n", "data_14", "check", "toggle"],
    "id_column": ["id", "record_id", "sr_no", "customer_id", "uid",
                  "col_15", "field_o", "data_15", "key", "ref"],
    "ratio": ["ratio", "proportion", "weight", "probability", "share",
              "col_16", "field_p", "data_16", "factor", "coeff"],
    "count": ["count", "qty", "frequency", "num_orders", "total_count",
              "col_17", "field_q", "data_17", "tally", "occurrences"],
    "age": ["age", "age_years", "years_old", "age_yr", "patient_age",
            "col_18", "field_r", "data_18", "years", "duration"],
    "category_low_card": ["category", "gender", "region", "department", "tier",
                          "col_19", "field_s", "data_19", "group", "segment"],
    "free_text": ["remarks", "description", "comments", "feedback", "notes",
                  "col_20", "field_t", "data_20", "narrative", "body"],
}


def generate_training_dataset(
    columns_per_type: int = 150,
    rows_per_column: int = 200,
    seed: int = 42,
) -> tuple:
    """
    Generate a complete labelled training dataset.

    50% of columns use keyword-matching names, 50% use generic names.
    This forces the model to learn from data patterns, not just column names.

    Parameters
    ----------
    columns_per_type : int
        Number of synthetic columns to generate per semantic type.
    rows_per_column : int
        Number of rows in each synthetic column.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (list[pd.Series], list[str], list[str])
        (columns, labels, column_names)
    """
    random.seed(seed)
    np.random.seed(seed)

    all_columns = []
    all_labels = []
    all_names = []

    for sem_type, gen_func in GENERATORS.items():
        variant_names = VARIANT_COLUMN_NAMES.get(sem_type, [sem_type])
        for i in range(columns_per_type):
            # Vary column length more aggressively
            n_rows = random.randint(50, 500)

            series = gen_func(n=n_rows)

            # Cycle through variant names (first 5 are keyword-matching,
            # last 5 are generic — ensuring 50/50 split)
            col_name = variant_names[i % len(variant_names)]
            series.name = col_name

            all_columns.append(series)
            all_labels.append(sem_type)
            all_names.append(col_name)

    return all_columns, all_labels, all_names
