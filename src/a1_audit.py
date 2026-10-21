"""Part A task 2 - audit the raw data (shape, dtypes, missingness, specific audit figures)."""
import pandas as pd

from common import load_raw

tables = {n: load_raw(n) for n in ("customers", "products", "orders")}

for name, df in tables.items():
    print(f"\n=== {name}: shape {df.shape} ===")
    miss = df.isna().sum()
    rep = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": miss,
                        "missing_pct": (miss / len(df) * 100).round(2)})
    print(rep.to_string())

o, c = tables["orders"], tables["customers"]
amount_num = pd.to_numeric(o["order_amount"], errors="coerce")
age_num = pd.to_numeric(c["age"], errors="coerce")
gender_nonblank = c["gender"].dropna()
fig = {
    "raw orders.csv rows": len(o),
    "exact duplicate rows": int(o.duplicated().sum()),
    "rows with status pending": int((o["status"] == "pending").sum()),
    "distinct order_id": int(o["order_id"].nunique()),
    "non-numeric order_amount entries": int((amount_num.isna() & o["order_amount"].notna()).sum()),
    "negative delivery_minutes": int((o["delivery_minutes"] < 0).sum()),
    "blank delivery_minutes": int(o["delivery_minutes"].isna().sum()),
    "customers: blank age": int(c["age"].isna().sum()),
    "customers: age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "customers: distinct raw city spellings": int(c["city"].nunique()),
    "customers: distinct non-blank raw gender spellings": int(gender_nonblank.nunique()),
    "customers: blank gender": int(c["gender"].isna().sum()),
}
print("\n=== audit figures ===")
for k, v in fig.items():
    print(f"{k:55s} {v:>7,}")
