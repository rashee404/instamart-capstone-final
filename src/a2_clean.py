"""Part A tasks 3-5 - clean customers and orders, write data/clean/*.csv and data/instamart.db."""
import sqlite3

import pandas as pd

from common import CLEAN, DB, load_raw

TS_FMT = "%Y-%m-%d %H:%M:%S"
customers, products, orders = load_raw("customers"), load_raw("products"), load_raw("orders")

# ------------------------------------------------------------------ customers
print("=== CLEAN CUSTOMERS ===")
raw_cities = customers["city"].nunique()
customers["city"] = customers["city"].str.strip().str.title()
print(f"(a) city: {raw_cities} raw spellings -> {customers['city'].nunique()} cities")

def map_gender(g):
    if pd.isna(g):
        return "unknown"
    g = str(g).strip().lower()
    return "male" if g in ("male", "m") else "female" if g in ("female", "f") else "unknown"

customers["gender"] = customers["gender"].map(map_gender)
print("(b) gender mapped -> male/female/unknown")

age = pd.to_numeric(customers["age"], errors="coerce")
bad_age = (age < 10) | (age > 100)
age = age.mask(bad_age)
n_missing_before_fill = int(age.isna().sum())
customers["age"] = age.fillna(age.groupby(customers["city"]).transform("median"))
print(f"(c) age: {int(bad_age.sum())} impossible ages set to missing; {n_missing_before_fill} missing filled "
      f"with city median; missing now {int(customers['age'].isna().sum())}; range {customers['age'].min()}-{customers['age'].max()}")

customers["signup_date"] = pd.to_datetime(customers["signup_date"])
print("(d) signup_date converted to datetime")
print(customers["city"].value_counts().to_string())
print(customers["gender"].value_counts().to_string())

# ------------------------------------------------------------------ orders
print("\n=== CLEAN ORDERS ===")
n0 = len(orders)
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
print(f"1. order_ts -> datetime ({n0:,} values converted)")

orders = orders.drop_duplicates()
n1 = len(orders)
print(f"2. exact duplicates removed: {n0 - n1} rows ({n0:,} -> {n1:,})")

orders = orders.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
n2 = len(orders)
print(f"3. kept latest record per order_id: {n1 - n2} rows removed ({n1:,} -> {n2:,})")

amount = pd.to_numeric(orders["order_amount"], errors="coerce")
unparseable = int(amount.isna().sum())
price = orders["product_id"].map(products.set_index("product_id")["price_inr"])
computed = (price * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = amount.fillna(computed)
print(f"4. order_amount: {unparseable} unparseable values -> missing -> filled from product price; "
      f"missing now {int(orders['order_amount'].isna().sum())}")

neg = orders["delivery_minutes"] < 0
orders.loc[neg, "delivery_minutes"] = float("nan")
completed = orders["status"] == "completed"
cat = orders["product_id"].map(products.set_index("product_id")["category"])
med = orders["delivery_minutes"].where(completed).groupby(cat).transform("median")
fill_mask = completed & orders["delivery_minutes"].isna()
orders.loc[fill_mask, "delivery_minutes"] = med[fill_mask]
print(f"5. delivery_minutes: {int(neg.sum())} negatives -> missing; {int(fill_mask.sum())} missing completed orders "
      f"filled with category median; cancelled orders left missing "
      f"({int((orders['status'] == 'cancelled').sum())}, never delivered)")
print("   category medians (completed):", orders["delivery_minutes"].where(completed).groupby(cat).median().to_dict())

orders = orders.sort_values("order_id").reset_index(drop=True)
orders["delivery_minutes"] = orders["delivery_minutes"].astype("float64")

print(f"\nclean orders rows {len(orders):,}; status counts {orders['status'].value_counts().to_dict()}")
print(f"completed revenue INR {orders.loc[orders['status'] == 'completed', 'order_amount'].sum():,.2f}")

# ------------------------------------------------------------------ store
out_c, out_o = customers.copy(), orders.copy()
out_c["signup_date"] = out_c["signup_date"].dt.strftime("%Y-%m-%d")
out_o["order_ts"] = out_o["order_ts"].dt.strftime(TS_FMT)
out_c.to_csv(CLEAN / "customers.csv", index=False)
products.to_csv(CLEAN / "products.csv", index=False)
out_o.to_csv(CLEAN / "orders.csv", index=False)

if DB.exists():
    DB.unlink()
with sqlite3.connect(DB) as con:
    out_c.to_sql("customers", con, index=False)
    products.to_sql("products", con, index=False)
    out_o.to_sql("orders", con, index=False)
    con.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
print(f"\nwritten: {CLEAN}/customers.csv products.csv orders.csv and {DB}")
