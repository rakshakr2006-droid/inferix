# Inferix — India-Aware Semantic Data Type Inferencer

[![PyPI version](https://img.shields.io/pypi/v/inferix.svg)](https://pypi.org/project/inferix/)
[![Python](https://img.shields.io/pypi/pyversions/inferix.svg)](https://pypi.org/project/inferix/)
[![Downloads](https://img.shields.io/pypi/dm/inferix.svg)](https://pypi.org/project/inferix/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-62%20passed-brightgreen.svg)]()

> Automatically detect what your CSV columns actually **mean** — PAN numbers, GST numbers, Aadhaar, IFSC codes, Indian mobile numbers, and 14 more types.

---

## What It Does

Most tools like pandas only detect basic types (`int`, `float`, `string`). **Inferix** goes deeper and detects the **semantic meaning** of each column.

| Column Name   | pandas says  | Inferix says     | Confidence |
|--------------|-------------|-----------------|------------|
| `cust_pan`    | `object`     | `pan_number`     | 94%        |
| `gst_code`    | `object`     | `gst_number`     | 97%        |
| `join_date`   | `int64`      | `date_disguised` | 88%        |
| `monthly_amt` | `float64`    | `inr_currency`   | 91%        |

## Supported Semantic Types (20)

### India-Specific (7)
| Type | Description |
|------|-------------|
| `pan_number` | Permanent Account Number |
| `gst_number` | GST Identification Number |
| `aadhaar_number` | Aadhaar UID (12-digit) |
| `ifsc_code` | Bank IFSC Code |
| `indian_mobile` | Indian Mobile Number (10-digit) |
| `indian_pincode` | Indian Postal PIN Code (6-digit) |
| `inr_currency` | Indian Rupee Amount |

### General (13)
| Type | Description |
|------|-------------|
| `email_address` | Email addresses |
| `url` | Web URLs |
| `date_formatted` | Dates in DD/MM/YYYY etc. |
| `date_disguised` | Dates stored as integers (YYYYMMDD) |
| `timestamp` | Unix timestamps |
| `percentage` | Percentage values |
| `binary_flag` | Yes/No, True/False, 0/1 |
| `id_column` | Sequential or UUID identifiers |
| `ratio` | Decimal values between 0 and 1 |
| `count` | Non-negative integer counts |
| `age` | Age values (0-120) |
| `category_low_card` | Low-cardinality categorical |
| `free_text` | Free-form text / remarks |

## Installation

```bash
pip install inferix
```

Or install from source:

```bash
git clone https://github.com/rakshakr2006-droid/inferix.git
cd inferix
pip install -e .
```

## Quick Start

### Train the Model (one-time setup)
```bash
python -m inferix.train
```

### Use It
```python
import pandas as pd
from inferix import infer

df = pd.read_csv("your_data.csv")
results = infer(df)
print(results)
```

**Output:**
```
  column_name   semantic_type   confidence  evidence
  cust_pan      pan_number      94%         regex_pan=0.94, name_match=yes
  gst_code      gst_number      97%         regex_gst=0.97, name_match=yes
  join_date     date_disguised  88%         all_int=True, name_match=yes
  monthly_amt   inr_currency    91%         regex_inr=0.85, name_match=yes
```

## Architecture

Inferix uses a **5-layer analysis pipeline** combining regex pattern matching, statistical profiling, and XGBoost classification:

```
Column Data --> [Layer 1: Syntactic] --> null%, unique%, dtype
             --> [Layer 2: Pattern]  --> regex match scores (12 patterns)
             --> [Layer 3: Stats]    --> mean, std, skew, entropy
             --> [Layer 4: Name]     --> column name keyword match
                           |
              [All 50 features combined]
                           |
              [Layer 5: XGBoost Classifier]
                           |
              Semantic Type + Confidence + Evidence
```

## Project Structure

```
inferix/
├── inferix/
│   ├── __init__.py          # Package init, public API
│   ├── infer.py             # Main infer() function
│   ├── patterns.py          # 12 Indian regex patterns
│   ├── features.py          # 50-feature extraction pipeline
│   ├── data_generator.py    # Synthetic training data
│   ├── train.py             # Model training script
│   └── model/
│       ├── inferix_model.json
│       └── label_encoder.pkl
├── tests/
│   ├── test_patterns.py     # 35 pattern tests
│   ├── test_features.py     # 19 feature tests
│   └── test_infer.py        # 8 inference tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v
```

## System Requirements

- Python 3.10+
- 8GB RAM (no GPU needed)
- Works completely offline after initial setup

## Why Inferix?

| Tool | Semantic Detection | India Types | ML-Based | Lightweight |
|------|:-:|:-:|:-:|:-:|
| **Inferix** | 20 types | 7 types | XGBoost | ~50MB |
| Sherlock (MIT) | 78 types | 0 | DNN | ~2GB, needs GPU |
| csv-detective | ~30 types | 0 | No | ~10MB |
| pandas | 0 | 0 | No | built-in |

## License

MIT License

## Acknowledgements

Inspired by [Sherlock](https://github.com/mitmedialab/sherlock-project) (MIT, 2019)
which detects 78 generic semantic types. Inferix fills the gap for
**India-specific** types that Sherlock cannot detect.
