"""
sanitizer.py — Strip PII from user queries before processing.

Patterns removed:
  - 12-digit PRAN numbers (may have spaces/dashes)
  - Aadhaar numbers (12-digit, same format)
  - Indian bank account numbers (9–18 digit sequences)
  - UPI IDs (user@bank patterns)

The sanitizer is conservative — it removes anything that looks like a
sensitive identifier and replaces it with a safe placeholder so the
query still parses grammatically.
"""

import re

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# PRAN / Aadhaar: 12 digits, optionally separated by spaces or dashes
_PRAN_AADHAAR = re.compile(r"\b(\d[\s\-]?){11}\d\b")

# Bank account: standalone 9-18 digit sequence not already caught above
_BANK_ACCOUNT = re.compile(r"\b\d{9,18}\b")

# UPI ID: something@something (alphanumeric)
_UPI_ID = re.compile(r"\b[\w.\-]+@[\w.\-]+\b")

# IFSC code: 4 letters + 0 + 6 alphanumeric
_IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sanitize(query: str) -> str:
    """
    Remove PII patterns from a user query and return the cleaned string.
    Replacements use bracketed placeholders so the sentence stays readable.
    """
    query = _PRAN_AADHAAR.sub("[ID REMOVED]", query)
    query = _BANK_ACCOUNT.sub("[ACCOUNT REMOVED]", query)
    query = _UPI_ID.sub("[UPI REMOVED]", query)
    query = _IFSC.sub("[IFSC REMOVED]", query)
    return query.strip()
