"""Part A task 5 - assert the reference checkpoints from tasks 3 and 4; print ALL CHECKPOINTS PASSED."""
import sqlite3

import pandas as pd

from common import CLEAN, DB

o = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
c = pd.read_csv(CLEAN / "customers.csv")
done = o[o["status"] == "completed"]
cancelled = o[o["status"] == "cancelled"]

checks = [
    ("orders: one row per order_id (12,309)", len(o) == 12309 and o["order_id"].is_unique),
    ("orders: 11,110 completed", len(done) == 11110),
    ("orders: 1,199 cancelled", len(cancelled) == 1199),
    ("orders: no pending rows", (o["status"] == "pending").sum() == 0),
    ("orders: 0 missing order_amount", o["order_amount"].isna().sum() == 0),
    ("orders: 0 missing delivery (completed)", done["delivery_minutes"].isna().sum() == 0),
    ("orders: 1,199 missing delivery (cancelled)", cancelled["delivery_minutes"].isna().sum() == 1199),
    ("orders: no negative delivery", (o["delivery_minutes"].dropna() >= 0).all()),
    ("orders: completed revenue 2,998,668.80", round(done["order_amount"].sum(), 2) == 2998668.80),
    ("customers: exactly 6 cities", c["city"].nunique() == 6),
    ("customers: city counts", c["city"].value_counts().to_dict() ==
     {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("customers: gender counts", c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("customers: 0 missing ages", c["age"].isna().sum() == 0),
    ("customers: ages within 18-60", c["age"].between(18, 60).all()),
]
with sqlite3.connect(DB) as con:
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    n_db = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    rev_db = con.execute("SELECT ROUND(SUM(order_amount),2) FROM orders WHERE status='completed'").fetchone()[0]
checks += [
    ("db: tables customers/products/orders", {"customers", "products", "orders"} <= tables),
    ("db: 12,309 order rows", n_db == 12309),
    ("db: completed revenue 2,998,668.80", rev_db == 2998668.80),
]
failed = [name for name, ok in checks if not ok]
for name in failed:
    print("FAIL  " + name)
assert not failed, f"failed checkpoints: {failed}"
print("ALL CHECKPOINTS PASSED")
