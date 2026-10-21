"""Shared paths and loaders. All scripts are run from the repository root."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # mock_llm.py lives at the repository root
    sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
POLICIES = ROOT / "data" / "policies"
DB = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"
STATE = ROOT / "state"
SQL = ROOT / "sql" / "queries.sql"

for _d in (CLEAN, OUT, STATE):
    _d.mkdir(parents=True, exist_ok=True)


def load_raw(name):
    """Raw CSV with ONLY blank cells treated as missing.

    Default pandas also turns the text 'N/A' into NaN, which would hide the planted
    unparseable amounts ('N/A', 'error'); keeping them as text lets us count them.
    """
    return pd.read_csv(RAW / f"{name}.csv", na_values=[""], keep_default_na=False)


def load_tickets():
    """Tickets with empty texts kept as empty strings (not NaN)."""
    return pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
