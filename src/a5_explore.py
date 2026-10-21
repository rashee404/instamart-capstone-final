"""Part A task 7 - pandas cross-check of Q2 and exploratory statistics for completed order_amount."""
from itertools import combinations

import pandas as pd

from common import CLEAN

o = pd.read_csv(CLEAN / "orders.csv")
c = pd.read_csv(CLEAN / "customers.csv")
p = pd.read_csv(CLEAN / "products.csv")
done = o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id") \
                                     .merge(p[["product_id", "category"]], on="product_id")

# (a) pivot table with margins must equal the SQL total
pt = pd.pivot_table(done, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print("pivot_table grand total:", f"{pt.loc['All', 'All']:,.2f}  (SQL Q2 total 2,998,668.80)")
assert round(pt.loc["All", "All"], 2) == 2998668.80
print(pt.round(2).to_string())

agg = done.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                               n_orders=("order_amount", "count")).round(2)
print("\nper-city completed order_amount:\n", agg.to_string())

# (b) distribution statistics
x = done["order_amount"]
q1, med, q3 = x.quantile([.25, .5, .75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean {x.mean():.2f} median {med:.2f} skew {x.skew():.2f}")
print(f"Q1 {q1:.2f} Q3 {q3:.2f} IQR {iqr:.2f} lower fence {lo:.2f} upper fence {hi:.2f}")
print(f"outliers below lower fence: {int((x < lo).sum())}; above upper fence: {int((x > hi).sum())}")
cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = done[cols].corr()
print("\ncorrelation matrix:\n", corr.round(3).to_string())
pairs = {(a, b): corr.loc[a, b] for a, b in combinations(cols, 2)}
print("\npairs by |r|:")
for (a, b), r in sorted(pairs.items(), key=lambda kv: -abs(kv[1]))[:4]:
    print(f"  {a} & {b}: {r:+.3f}")
